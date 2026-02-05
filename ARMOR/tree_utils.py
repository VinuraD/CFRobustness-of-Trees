"""
Tree Adapter Layer for ARMOR

Normalizes tree formats from XGBoost, LightGBM, and AdaBoost into the
flat-array representation that TreeMCTS already consumes (children_left,
children_right, feature, threshold, value, n_node_samples).

This means zero changes to TreeMCTS internals.
"""

import json
import numpy as np
from dataclasses import dataclass

try:
    import xgboost
except ImportError:
    xgboost = None

try:
    import lightgbm
except ImportError:
    lightgbm = None

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


@dataclass
class TreeArrays:
    """Holds flat numpy arrays matching sklearn's tree.tree_.* interface."""
    children_left: np.ndarray    # (n_nodes,) int -- left child index, -1 for leaves
    children_right: np.ndarray   # (n_nodes,) int -- right child index, -1 for leaves
    feature: np.ndarray          # (n_nodes,) int -- split feature index, -2 for leaves
    threshold: np.ndarray        # (n_nodes,) float -- split threshold
    value: np.ndarray            # (n_nodes, 1, 2) float -- class probabilities/counts
    n_node_samples: np.ndarray   # (n_nodes,) int


class WrappedTree:
    """Duck-type compatible with sklearn DecisionTree for TreeMCTS."""

    def __init__(self, tree_arrays: TreeArrays):
        self.tree_ = tree_arrays


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_trees(model) -> list:
    """Return list of tree objects (sklearn or WrappedTree) from any supported model.

    Supported model types:
    - RandomForestClassifier  -> model.estimators_ (already sklearn trees)
    - AdaBoostClassifier      -> model.estimators_ (already sklearn trees)
    - XGBClassifier           -> parse JSON dump into WrappedTree list
    - LGBMClassifier          -> parse dump_model() into WrappedTree list
    """
    model_type = get_model_type(model)

    if model_type in ('random_forest', 'adaboost'):
        return list(model.estimators_)

    if model_type == 'xgboost':
        return _extract_xgboost_trees(model)

    if model_type == 'lightgbm':
        return _extract_lightgbm_trees(model)

    raise ValueError(f"Unsupported model type: {type(model)}")


def get_model_type(model) -> str:
    """Identify the model type string."""
    if isinstance(model, RandomForestClassifier):
        return 'random_forest'
    if isinstance(model, AdaBoostClassifier):
        return 'adaboost'
    if xgboost is not None and isinstance(model, xgboost.XGBClassifier):
        return 'xgboost'
    if lightgbm is not None and isinstance(model, lightgbm.LGBMClassifier):
        return 'lightgbm'
    raise ValueError(f"Unsupported model type: {type(model)}")


def get_n_estimators(model) -> int:
    """Return number of trees in the ensemble."""
    model_type = get_model_type(model)
    if model_type in ('random_forest', 'adaboost'):
        return len(model.estimators_)
    if model_type == 'xgboost':
        return len(model.get_booster().get_dump())
    if model_type == 'lightgbm':
        return model.booster_.num_trees()
    raise ValueError(f"Unsupported model type: {type(model)}")


# ---------------------------------------------------------------------------
# XGBoost tree extraction
# ---------------------------------------------------------------------------

def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def _extract_xgboost_trees(model) -> list:
    """Parse XGBoost JSON dump into list of WrappedTree objects."""
    booster = model.get_booster()
    dumps = booster.get_dump(dump_format='json')

    trees = []
    for dump_str in dumps:
        tree_json = json.loads(dump_str)
        wrapped = _parse_xgb_tree(tree_json)
        trees.append(wrapped)
    return trees


def _parse_xgb_tree(tree_json: dict) -> WrappedTree:
    """BFS a single XGBoost JSON tree and produce a WrappedTree."""
    # BFS to assign contiguous node IDs
    nodes_bfs = []
    queue = [tree_json]
    while queue:
        node = queue.pop(0)
        nodes_bfs.append(node)
        if 'children' in node:
            for child in node['children']:
                queue.append(child)

    n_nodes = len(nodes_bfs)

    # Create a mapping from original XGB node id to our contiguous index
    xgb_id_to_idx = {}
    for idx, node in enumerate(nodes_bfs):
        xgb_id_to_idx[node['nodeid']] = idx

    children_left = np.full(n_nodes, -1, dtype=np.int64)
    children_right = np.full(n_nodes, -1, dtype=np.int64)
    feature = np.full(n_nodes, -2, dtype=np.int64)
    threshold = np.full(n_nodes, -2.0, dtype=np.float64)
    value = np.zeros((n_nodes, 1, 2), dtype=np.float64)
    n_node_samples = np.ones(n_nodes, dtype=np.int64)

    for idx, node in enumerate(nodes_bfs):
        if 'leaf' in node:
            # Leaf node
            log_odds = node['leaf']
            p = float(_sigmoid(log_odds))
            value[idx, 0, 0] = 1.0 - p
            value[idx, 0, 1] = p
            n_node_samples[idx] = int(node.get('cover', 1))
        else:
            # Internal node
            split_name = node['split']
            # Parse feature index: "f5" -> 5, or try int directly
            if isinstance(split_name, str) and split_name.startswith('f'):
                feat_idx = int(split_name[1:])
            else:
                try:
                    feat_idx = int(split_name)
                except (ValueError, TypeError):
                    feat_idx = int(split_name.replace('f', ''))

            feature[idx] = feat_idx
            threshold[idx] = node['split_condition']

            # XGBoost: 'yes' is the branch taken when condition is true (< threshold)
            yes_id = node['yes']
            no_id = node['no']
            children_left[idx] = xgb_id_to_idx[yes_id]
            children_right[idx] = xgb_id_to_idx[no_id]

            cover = node.get('cover', 1)
            n_node_samples[idx] = int(cover)

            # Internal node value: use cover-weighted estimate
            log_odds = node.get('leaf', 0.0)
            p = float(_sigmoid(log_odds))
            value[idx, 0, 0] = 1.0 - p
            value[idx, 0, 1] = p

    arrays = TreeArrays(
        children_left=children_left,
        children_right=children_right,
        feature=feature,
        threshold=threshold,
        value=value,
        n_node_samples=n_node_samples,
    )
    return WrappedTree(arrays)


# ---------------------------------------------------------------------------
# LightGBM tree extraction
# ---------------------------------------------------------------------------

def _extract_lightgbm_trees(model) -> list:
    """Parse LightGBM dump_model() into list of WrappedTree objects."""
    model_dump = model.booster_.dump_model()
    tree_infos = model_dump['tree_info']

    trees = []
    for tree_info in tree_infos:
        root = tree_info['tree_structure']
        wrapped = _parse_lgb_tree(root)
        trees.append(wrapped)
    return trees


def _parse_lgb_tree(root: dict) -> WrappedTree:
    """Pre-order traversal of a LightGBM tree dict into a WrappedTree."""
    # First pass: count nodes via pre-order traversal
    node_list = []
    _lgb_preorder(root, node_list)

    n_nodes = len(node_list)
    children_left = np.full(n_nodes, -1, dtype=np.int64)
    children_right = np.full(n_nodes, -1, dtype=np.int64)
    feature_arr = np.full(n_nodes, -2, dtype=np.int64)
    threshold_arr = np.full(n_nodes, -2.0, dtype=np.float64)
    value_arr = np.zeros((n_nodes, 1, 2), dtype=np.float64)
    n_node_samples = np.ones(n_nodes, dtype=np.int64)

    # Build ID mapping: node dict id -> contiguous index
    node_to_idx = {id(node): idx for idx, node in enumerate(node_list)}

    for idx, node in enumerate(node_list):
        if 'leaf_value' in node:
            # Leaf node
            log_odds = node['leaf_value']
            p = float(_sigmoid(log_odds))
            value_arr[idx, 0, 0] = 1.0 - p
            value_arr[idx, 0, 1] = p
            n_node_samples[idx] = int(node.get('leaf_count', 1))
        else:
            # Internal node
            feature_arr[idx] = int(node['split_feature'])
            threshold_arr[idx] = float(node['threshold'])

            left_child = node['left_child']
            right_child = node['right_child']
            children_left[idx] = node_to_idx[id(left_child)]
            children_right[idx] = node_to_idx[id(right_child)]

            n_node_samples[idx] = int(node.get('internal_count', 1))

            # Internal node value estimate
            value_arr[idx, 0, 0] = 0.5
            value_arr[idx, 0, 1] = 0.5

    arrays = TreeArrays(
        children_left=children_left,
        children_right=children_right,
        feature=feature_arr,
        threshold=threshold_arr,
        value=value_arr,
        n_node_samples=n_node_samples,
    )
    return WrappedTree(arrays)


def _lgb_preorder(node: dict, node_list: list):
    """Collect LightGBM tree nodes in pre-order."""
    node_list.append(node)
    if 'left_child' in node:
        _lgb_preorder(node['left_child'], node_list)
    if 'right_child' in node:
        _lgb_preorder(node['right_child'], node_list)

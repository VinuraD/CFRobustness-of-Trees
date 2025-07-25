#!/usr/bin/env python3
"""
Feature Tweak Counterfactual Generation
Simplified version adapted for robustness analysis

This module implements the FeatureTweak algorithm for generating counterfactual explanations
on tree-based models. It's adapted from the original implementation to work directly with
sklearn models without the CARLA framework dependency.

The FeatureTweak algorithm works by:
1. Identifying paths in decision trees that lead to the desired class
2. Finding epsilon-satisfactory instances that slightly modify feature values
3. Selecting the counterfactual with minimum cost

Reference:
Tolomei, G., Silvestri, F., Haines, A., & Lalmas, M. (2017). 
Interpretable predictions of tree-based ensembles via actionable feature tweaking. 
KDD 2017.
"""

import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def _L1_cost_func(a, b):
    """The 1-norm ||a-b||_1"""
    return np.linalg.norm(a - b, ord=1)


def _L2_cost_func(a, b):
    """The 2-norm ||a-b||_2"""
    return np.linalg.norm(a - b, ord=2)


def search_path(tree, class_labels):
    """
    Return path index list containing [{leaf node id, inequality symbol, threshold, feature index}].

    Parameters
    ----------
    tree: sklearn.tree.DecisionTreeClassifier
        The classification tree.
    class_labels: list
        All the possible class labels.

    Returns
    -------
    path_info: dict
        Dictionary containing path information for each leaf node
    """

    def parse_tree(tree):
        """
        Parse sklearn decision tree to extract structure information.
        """
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold
        values = tree.tree_.value

        # leaf nodes ID
        leaf_nodes = np.where(children_left == -1)[0]

        # outcomes of leaf nodes
        leaf_values = values[leaf_nodes].reshape(len(leaf_nodes), len(class_labels))
        leaf_classes = np.argmax(leaf_values, axis=-1)

        # select the leaf nodes whose outcome is class 1 (target counterfactual class)
        leaf_nodes = leaf_nodes[np.where(leaf_classes == 1)[0]]

        return children_left, children_right, feature, threshold, leaf_nodes

    """ select leaf nodes whose outcome is target class (1) """
    children_left, children_right, feature, threshold, leaf_nodes = parse_tree(tree)

    """ search the path to the selected leaf node """
    paths = {}
    for leaf_node in leaf_nodes:
        """correspond leaf node to left and right parents"""
        child_node = leaf_node
        parent_node = -100  # initialize
        parents_left = []
        parents_right = []
        while parent_node != 0:
            if np.where(children_left == child_node)[0].shape == (0,):
                parent_left = -1
                parent_right = np.where(children_right == child_node)[0][0]
                parent_node = parent_right
            elif np.where(children_right == child_node)[0].shape == (0,):
                parent_right = -1
                parent_left = np.where(children_left == child_node)[0][0]
                parent_node = parent_left
            parents_left.append(parent_left)
            parents_right.append(parent_right)
            """ for next step """
            child_node = parent_node

        # nodes dictionary containing left parents and right parents
        paths[leaf_node] = (parents_left, parents_right)

    path_info = get_path_info(paths, threshold, feature)
    return path_info


def get_path_info(paths, threshold, feature):
    """
    Extract the path info from the parameters

    Parameters
    ----------
    paths: dict
        Paths through the tree from root to leaves.
    threshold: array of double
        threshold[i] holds the threshold for the internal node i.
    feature: array of int
        feature[i] holds the feature to split on, for the internal node i.

    Returns
    -------
    path_info: dict
        Dictionary where dict[i] contains node_id, inequality_symbol, threshold, and feature
    """
    path_info = {}
    for i in paths:
        node_ids = []  # node ids used in the current node
        inequality_symbols = []  # inequality symbols used in the current node
        thresholds = []  # thresholds used in the current node
        features = []  # features used in the current node
        parents_left, parents_right = paths[i]

        for idx in range(len(parents_left)):

            def do_appends(node_id):
                """helper function to reduce duplicate code"""
                node_ids.append(node_id)
                thresholds.append(threshold[node_id])
                features.append(feature[node_id])

            if parents_left[idx] != -1:
                """the child node is the left child of the parent"""
                node_id = parents_left[idx]  # node id
                inequality_symbols.append(0)  # left child: feature <= threshold
                do_appends(node_id)
            elif parents_right[idx] != -1:
                """the child node is the right child of the parent"""
                node_id = parents_right[idx]
                inequality_symbols.append(1)  # right child: feature > threshold
                do_appends(node_id)

            path_info[i] = {
                "node_id": node_ids,
                "inequality_symbol": inequality_symbols,
                "threshold": thresholds,
                "feature": features,
            }
    return path_info


class FeatureTweakSimple:
    """
    Simplified FeatureTweak implementation for robustness analysis.
    
    This version works directly with sklearn models without CARLA dependencies.
    
    Parameters
    ----------
    eps: float, default=0.1
        Epsilon value for creating satisfactory instances
    cost_func: callable, default=_L2_cost_func
        Cost function for measuring distance between original and counterfactual
    """
    
    def __init__(self, eps=0.1, cost_func=_L2_cost_func):
        self.eps = eps
        self.cost_func = cost_func
    
    def esatisfactory_instance(self, x: np.ndarray, path_info):
        """
        Return the epsilon satisfactory instance of x.

        Parameters
        ----------
        x: np.ndarray
            A single factual example.
        path_info: dict
            One path from the result of search_path(tree, class_labels)

        Returns
        -------
        np.ndarray
            Epsilon satisfactory instance
        """
        esatisfactory = copy.deepcopy(x)
        for i in range(len(path_info["feature"])):
            feature_idx = path_info["feature"][i]  # feature index
            threshold_value = path_info["threshold"][i]  # threshold in current node
            inequality_symbol = path_info["inequality_symbol"][i]  # inequality symbol
            
            if inequality_symbol == 0:  # left child: feature <= threshold
                esatisfactory[feature_idx] = threshold_value - self.eps
            elif inequality_symbol == 1:  # right child: feature > threshold
                esatisfactory[feature_idx] = threshold_value + self.eps
                
        return esatisfactory

    def feature_tweaking(self, x: np.ndarray, model, class_labels: list, cf_label: int):
        """
        Perform feature tweaking on a single factual example.

        Parameters
        ----------
        x: np.ndarray
            A single factual example.
        model: sklearn model
            The trained model (RandomForestClassifier or DecisionTreeClassifier)
        class_labels: list
            List of possible class labels.
        cf_label: int
            What label the counterfactual should have.

        Returns
        -------
        np.ndarray
            Counterfactual example
        """
        x_out = copy.deepcopy(x)  # initialize output
        delta_mini = 10**3  # initialize cost

        # Handle different model types
        if isinstance(model, RandomForestClassifier):
            trees = model.estimators_
        elif isinstance(model, DecisionTreeClassifier):
            trees = [model]
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        for tree in trees:  # loop over individual trees
            # Check if this tree predicts differently from the ensemble for this instance
            tree_prediction = tree.predict(x.reshape(1, -1))[0]
            model_prediction = model.predict(x.reshape(1, -1))[0]
            
            if (model_prediction == tree_prediction and 
                tree_prediction != cf_label):
                
                paths_info = search_path(tree, class_labels)
                for key in paths_info:
                    """generate epsilon-satisfactory instance"""
                    path_info = paths_info[key]
                    es_instance = self.esatisfactory_instance(x, path_info)
                    
                    # Check if the epsilon-satisfactory instance achieves the target
                    tree_pred_es = tree.predict(es_instance.reshape(1, -1))[0]
                    cost = self.cost_func(x, es_instance)
                    
                    if (tree_pred_es == cf_label and cost < delta_mini):
                        x_out = es_instance
                        delta_mini = cost
                        
        return x_out

    def generate_counterfactuals(self, x_test, model, class_labels=[0, 1]):
        """
        Generate counterfactuals for a test set.
        
        Parameters
        ----------
        x_test: pd.DataFrame or np.ndarray
            Test instances
        model: sklearn model
            Trained model
        class_labels: list, default=[0, 1]
            List of class labels
            
        Returns
        -------
        cf_list: pd.DataFrame
            DataFrame with counterfactuals and success information
        success_rate: float
            Proportion of successful counterfactual generations
        """
        if isinstance(x_test, pd.DataFrame):
            x_test_array = x_test.values
            feature_names = x_test.columns
        else:
            x_test_array = x_test
            feature_names = [f'feature_{i}' for i in range(x_test_array.shape[1])]
        
        # Initialize results storage
        cf_list = pd.DataFrame(columns=list(feature_names) + ['cf_class', 'success'])
        
        successful_cfs = 0
        failed_cfs = 0
        
        for i in range(len(x_test_array)):
            query_instance = x_test_array[i]
            original_prediction = model.predict(query_instance.reshape(1, -1))[0]
            cf_label = 1 - original_prediction  # Flip the class
            
            try:
                # Generate counterfactual using FeatureTweak
                counterfactual = self.feature_tweaking(
                    query_instance, model, class_labels, cf_label
                )
                
                # Check if counterfactual actually flipped the prediction
                cf_prediction = model.predict(counterfactual.reshape(1, -1))[0]
                
                if cf_prediction == cf_label:
                    # Successful counterfactual
                    cf_row = list(counterfactual) + [cf_prediction, True]
                    successful_cfs += 1
                else:
                    # Failed to flip, use original with success=False
                    cf_row = list(query_instance) + [original_prediction, False]
                    failed_cfs += 1
                    
                cf_list.loc[i] = cf_row
                
            except Exception as e:
                # Failed to generate counterfactual, use original with success=False
                cf_row = list(query_instance) + [original_prediction, False]
                cf_list.loc[i] = cf_row
                failed_cfs += 1
        
        success_rate = successful_cfs / len(x_test_array) if len(x_test_array) > 0 else 0
        return cf_list, success_rate

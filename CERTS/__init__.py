"""
CERTS: Counterfactual Explanations via Robust Tree-Search

A counterfactual explanation method that generates robust counterfactuals
using MCTS-based candidate generation and perturbation ensemble validation.
"""

try:
    from .certs import CERTS
    from .mcts import TreeMCTS, MCTSState
    from .distance import heom_distance
    from .constraints import get_immutable_features, intersect_constraint
    from .tree_utils import extract_trees, get_model_type, WrappedTree
except ImportError:
    # Fallback for when running as script (not as package)
    from certs import CERTS
    from mcts import TreeMCTS, MCTSState
    from distance import heom_distance
    from constraints import get_immutable_features, intersect_constraint
    from tree_utils import extract_trees, get_model_type, WrappedTree

__all__ = [
    'CERTS',
    'TreeMCTS',
    'MCTSState',
    'heom_distance',
    'get_immutable_features',
    'intersect_constraint',
    'extract_trees',
    'get_model_type',
    'WrappedTree',
]

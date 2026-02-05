"""
ARMOR: Adversarially Robust Model-Optimized Recourse

A counterfactual explanation method that generates robust counterfactuals
using MCTS-based candidate generation and perturbation ensemble validation.
"""

try:
    from .armor import ARMOR
    from .mcts import TreeMCTS, MCTSState
    from .distance import heom_distance
    from .constraints import get_immutable_features, intersect_constraint
    from .tree_utils import extract_trees, get_model_type, WrappedTree
except ImportError:
    # Fallback for when running as script (not as package)
    from armor import ARMOR
    from mcts import TreeMCTS, MCTSState
    from distance import heom_distance
    from constraints import get_immutable_features, intersect_constraint
    from tree_utils import extract_trees, get_model_type, WrappedTree

__all__ = [
    'ARMOR',
    'TreeMCTS',
    'MCTSState',
    'heom_distance',
    'get_immutable_features',
    'intersect_constraint',
    'extract_trees',
    'get_model_type',
    'WrappedTree',
]

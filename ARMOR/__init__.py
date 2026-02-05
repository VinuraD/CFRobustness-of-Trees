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
except ImportError:
    # Fallback for when running as script (not as package)
    from armor import ARMOR
    from mcts import TreeMCTS, MCTSState
    from distance import heom_distance
    from constraints import get_immutable_features, intersect_constraint

__all__ = [
    'ARMOR',
    'TreeMCTS',
    'MCTSState',
    'heom_distance',
    'get_immutable_features',
    'intersect_constraint'
]

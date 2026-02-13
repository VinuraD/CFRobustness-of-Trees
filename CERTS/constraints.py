"""
CERTS Constraint Handling

Functions for managing feature constraints and immutable features.
"""

import numpy as np
from typing import Dict, Tuple, Set, Optional, Any, Union


def get_immutable_features(data_module):
    """
    Get indices of features that cannot be changed.

    Parameters
    ----------
    data_module : DataModule
        Data module containing feature metadata

    Returns
    -------
    list
        List of feature indices that are immutable
    """
    metadata = data_module.get_metadata()
    feature_actions = metadata['feature_actions']
    label_col = metadata['label_column']

    immutable = []
    feature_idx = 0
    for col, action in feature_actions.items():
        if col == label_col:
            continue
        # 'FIX' or similar indicates immutable
        if isinstance(action, str) and action.upper() in ['FIX', 'FIXED', 'IMMUTABLE']:
            immutable.append(feature_idx)
        feature_idx += 1

    return immutable


def get_feature_types_list(data_module):
    """
    Get feature types as a list (excluding label column).

    Parameters
    ----------
    data_module : DataModule
        Data module containing feature metadata

    Returns
    -------
    list
        List of feature types ('C', 'N', 'D', 'B') indexed by feature position
    """
    metadata = data_module.get_metadata()
    feature_types = metadata['feature_types']
    label_col = metadata['label_column']

    types_list = []
    for col, ftype in feature_types.items():
        if col == label_col:
            continue
        types_list.append(ftype)

    return types_list


def intersect_constraint(
    constraints: Dict[int, Any],
    feat: int,
    ftype: str,
    n_categories: int = None,
    lower: float = None,
    upper: float = None
) -> Optional[Dict[int, Any]]:
    """
    Intersect a new constraint with existing constraints.

    Parameters
    ----------
    constraints : dict
        Current constraints. For numerical features: {feat_idx: (lower, upper)}.
        For categorical: {feat_idx: set_of_valid_values}.
    feat : int
        Feature index
    ftype : str
        Feature type ('C' for categorical, 'N'/'D'/'B' for numerical)
    n_categories : int, optional
        Number of categories for categorical features
    lower : float, optional
        Lower bound (exclusive) for numerical features
    upper : float, optional
        Upper bound (inclusive) for numerical features

    Returns
    -------
    dict or None
        Updated constraints, or None if infeasible
    """
    new_constraints = constraints.copy()

    if ftype == 'C':
        # Categorical (label-encoded)
        if n_categories is None:
            n_categories = 100  # Default fallback

        existing = new_constraints.get(feat, set(range(n_categories)))

        if upper is not None:
            # x <= thresh means integer values {0, 1, ..., floor(thresh)}
            valid = set(range(int(upper) + 1))
            existing = existing & valid

        if lower is not None:
            # x > thresh means integer values {ceil(thresh+1), ...}
            valid = set(range(int(lower) + 1, n_categories))
            existing = existing & valid

        if not existing:
            return None  # Infeasible

        new_constraints[feat] = existing

    else:
        # Numerical (N, D, B)
        lo, hi = new_constraints.get(feat, (0.0, 1.0))  # MinMax scaled

        if lower is not None:
            lo = max(lo, lower + 1e-9)  # Strict inequality

        if upper is not None:
            hi = min(hi, upper)

        if lo >= hi:
            return None  # Infeasible

        new_constraints[feat] = (lo, hi)

    return new_constraints


def project_to_constraints(
    x: np.ndarray,
    constraints: Dict[int, Any],
    feature_types: list
) -> np.ndarray:
    """
    Project an instance into the constraint region.

    Parameters
    ----------
    x : np.ndarray
        Original instance
    constraints : dict
        Constraints to satisfy
    feature_types : list
        List of feature types

    Returns
    -------
    np.ndarray
        Projected instance satisfying constraints
    """
    x_projected = x.copy()

    for feat, constraint in constraints.items():
        if feat >= len(feature_types):
            continue

        ftype = feature_types[feat]

        if ftype == 'C':
            # Categorical: pick closest valid integer
            valid_set = constraint
            current = int(x_projected[feat])
            if current not in valid_set:
                x_projected[feat] = min(valid_set, key=lambda v: abs(v - current))
        else:
            # Numerical: clip to bounds
            lo, hi = constraint
            x_projected[feat] = np.clip(x_projected[feat], lo, hi)

    return x_projected


def check_constraints_satisfied(
    x: np.ndarray,
    constraints: Dict[int, Any],
    feature_types: list,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if an instance satisfies all constraints.

    Parameters
    ----------
    x : np.ndarray
        Instance to check
    constraints : dict
        Constraints to check against
    feature_types : list
        List of feature types
    tolerance : float, default=1e-6
        Numerical tolerance for bounds checking

    Returns
    -------
    bool
        True if all constraints are satisfied
    """
    for feat, constraint in constraints.items():
        if feat >= len(x) or feat >= len(feature_types):
            continue

        ftype = feature_types[feat]
        val = x[feat]

        if ftype == 'C':
            if int(round(val)) not in constraint:
                return False
        else:
            lo, hi = constraint
            if val < lo - tolerance or val > hi + tolerance:
                return False

    return True


def merge_constraints(
    constraints1: Dict[int, Any],
    constraints2: Dict[int, Any],
    feature_types: list
) -> Optional[Dict[int, Any]]:
    """
    Merge two sets of constraints.

    Parameters
    ----------
    constraints1 : dict
        First set of constraints
    constraints2 : dict
        Second set of constraints
    feature_types : list
        List of feature types

    Returns
    -------
    dict or None
        Merged constraints, or None if infeasible
    """
    merged = constraints1.copy()

    for feat, constraint2 in constraints2.items():
        if feat >= len(feature_types):
            continue

        ftype = feature_types[feat]

        if feat not in merged:
            merged[feat] = constraint2
        else:
            constraint1 = merged[feat]

            if ftype == 'C':
                # Intersection of sets
                intersection = constraint1 & constraint2
                if not intersection:
                    return None
                merged[feat] = intersection
            else:
                # Intersection of intervals
                lo1, hi1 = constraint1
                lo2, hi2 = constraint2
                new_lo = max(lo1, lo2)
                new_hi = min(hi1, hi2)
                if new_lo >= new_hi:
                    return None
                merged[feat] = (new_lo, new_hi)

    return merged

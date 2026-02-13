"""
CERTS Distance Metrics

Implements HEOM (Heterogeneous Euclidean-Overlap Metric) for mixed-type data.
"""

import numpy as np


def heom_distance(x1, x2, feature_types, label_encoders=None):
    """
    Heterogeneous Euclidean-Overlap Metric for mixed-type data.

    For categorical features (label-encoded integers):
        - Overlap distance: 0 if same, 1 if different

    For numerical features (assumed to be MinMax scaled to [0,1]):
        - Euclidean distance component

    Parameters
    ----------
    x1 : np.ndarray
        First instance
    x2 : np.ndarray
        Second instance
    feature_types : dict or list
        Feature type indicators. If dict, keys are feature names and values are types.
        If list, indexed by feature position.
        Types: 'C' = categorical, 'N'/'D'/'B' = numerical
    label_encoders : dict, optional
        Label encoders for categorical features (not used in distance calculation
        but kept for API consistency)

    Returns
    -------
    float
        HEOM distance between x1 and x2
    """
    dist_sq = 0.0

    # Handle both dict and list feature_types
    if isinstance(feature_types, dict):
        type_list = list(feature_types.values())
    else:
        type_list = feature_types

    for j, (v1, v2) in enumerate(zip(x1, x2)):
        if j >= len(type_list):
            # Default to numerical if feature type not specified
            ftype = 'N'
        else:
            ftype = type_list[j]

        if ftype == 'C':
            # Categorical (label-encoded integer): Overlap distance
            # 0 if same, 1 if different
            dist_sq += 0.0 if v1 == v2 else 1.0
        else:
            # Numerical (N, D, B) - already MinMax scaled to [0,1]
            dist_sq += (v1 - v2) ** 2

    return np.sqrt(dist_sq)


def l2_distance(x1, x2):
    """
    Standard L2 (Euclidean) distance.

    Parameters
    ----------
    x1 : np.ndarray
        First instance
    x2 : np.ndarray
        Second instance

    Returns
    -------
    float
        Euclidean distance between x1 and x2
    """
    return np.linalg.norm(x1 - x2, ord=2)


def l0_distance(x1, x2, threshold=1e-6):
    """
    L0 distance (number of changed features).

    Parameters
    ----------
    x1 : np.ndarray
        First instance
    x2 : np.ndarray
        Second instance
    threshold : float, default=1e-6
        Minimum difference to consider a feature changed

    Returns
    -------
    int
        Number of features that differ between x1 and x2
    """
    return np.sum(np.abs(x1 - x2) > threshold)


def l1_distance(x1, x2):
    """
    L1 (Manhattan) distance.

    Parameters
    ----------
    x1 : np.ndarray
        First instance
    x2 : np.ndarray
        Second instance

    Returns
    -------
    float
        Manhattan distance between x1 and x2
    """
    return np.linalg.norm(x1 - x2, ord=1)


def normalized_heom_distance(x1, x2, feature_types, label_encoders=None):
    """
    Normalized HEOM distance (divided by number of features).

    Parameters
    ----------
    x1 : np.ndarray
        First instance
    x2 : np.ndarray
        Second instance
    feature_types : dict or list
        Feature type indicators
    label_encoders : dict, optional
        Label encoders for categorical features

    Returns
    -------
    float
        Normalized HEOM distance
    """
    n_features = len(x1)
    if n_features == 0:
        return 0.0
    return heom_distance(x1, x2, feature_types, label_encoders) / np.sqrt(n_features)

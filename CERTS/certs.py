"""
CERTS: Counterfactual Explanations via Robust Tree-Search

Main class for generating robust counterfactual explanations using MCTS-based
candidate generation and perturbation ensemble validation.
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    import xgboost
except ImportError:
    xgboost = None
try:
    import lightgbm
except ImportError:
    lightgbm = None

try:
    from .tree_utils import extract_trees, get_model_type, get_n_estimators
except ImportError:
    from tree_utils import extract_trees, get_model_type, get_n_estimators

try:
    from .mcts import TreeMCTS, ForestMCTS
    from .distance import heom_distance, l2_distance, l0_distance
    from .constraints import (
        get_immutable_features,
        get_feature_types_list,
        project_to_constraints
    )
except ImportError:
    # Fallback for when running as script (not as package)
    from mcts import TreeMCTS, ForestMCTS
    from distance import heom_distance, l2_distance, l0_distance
    from constraints import (
        get_immutable_features,
        get_feature_types_list,
        project_to_constraints
    )


def get_opposite_class(original_pred, classes):
    """
    Get the opposite class for recourse.

    Parameters
    ----------
    original_pred : int
        Original prediction
    classes : array-like
        All available classes

    Returns
    -------
    int
        Target class (opposite of original)
    """
    assert len(classes) == 2, f"Binary classification required, got {len(classes)} classes"
    return classes[1] if original_pred == classes[0] else classes[0]


def margin_rf(model, x_cf, target_class):
    """
    Compute vote proportion for target class (RandomForest margin).

    Parameters
    ----------
    model : RandomForestClassifier
        Trained model
    x_cf : np.ndarray
        Counterfactual instance
    target_class : int
        Target class

    Returns
    -------
    float
        Margin (probability of target class), in [0, 1]
    """
    proba = model.predict_proba(x_cf.reshape(1, -1))[0]
    # Handle case where target_class might not be in classes_
    if target_class < len(proba):
        return proba[target_class]
    return 0.0


def margin_boost(model, x_cf, target_class):
    """
    Compute sigmoid of raw score for XGBoost/LightGBM.

    Parameters
    ----------
    model : XGBClassifier or LGBMClassifier
        Trained model
    x_cf : np.ndarray
        Counterfactual instance
    target_class : int
        Target class

    Returns
    -------
    float
        Margin shifted to [-0.5, 0.5]
    """
    proba = model.predict_proba(x_cf.reshape(1, -1))[0]
    if target_class < len(proba):
        return proba[target_class] - 0.5
    return -0.5


class CERTS:
    """
    Counterfactual Explanations via Robust Tree-Search

    Generates counterfactual explanations that are robust to model and data
    perturbations using MCTS-based candidate generation and ensemble validation.

    Parameters
    ----------
    tau : float, default=0.7
        Minimum consensus threshold for robustness
    n_ensemble : int, default=10
        Number of perturbed models in the validation ensemble
    mcts_budget : int, default=500
        Total MCTS budget (distributed across trees)
    n_candidates_per_tree : int, default=5
        Number of candidates to extract from each tree
    w_consensus : float, default=0.4
        Weight for consensus (robustness) term
    w_margin : float, default=0.3
        Weight for margin (confidence) term
    w_proximity : float, default=0.2
        Weight for proximity term
    w_sparsity : float, default=0.1
        Weight for sparsity term
    random_state : int, optional
        Random seed for reproducibility
    """

    def __init__(
        self,
        tau: float = 0.7,
        n_ensemble: int = 10,
        mcts_budget: int = 500,
        n_candidates_per_tree: int = 5,
        w_consensus: float = 0.4,
        w_margin: float = 0.3,
        w_proximity: float = 0.2,
        w_sparsity: float = 0.1,
        random_state: int = 42
    ):
        self.tau = tau
        self.n_ensemble = n_ensemble
        self.mcts_budget = mcts_budget
        self.n_candidates_per_tree = n_candidates_per_tree
        self.w_c = w_consensus
        self.w_m = w_margin
        self.w_p = w_proximity
        self.w_s = w_sparsity
        self.random_state = random_state

        # Will be set during fit
        self.ensemble = None
        self.feature_types = None
        self.label_encoders = None
        self.immutable_features = None
        self.classes_ = None
        self.X_train = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        base_model=None,
        feature_types: List[str] = None,
        label_encoders: Dict = None,
        immutable_features: List[int] = None
    ):
        """
        Fit CERTS by generating the perturbation ensemble.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            Training features
        y_train : np.ndarray or pd.Series
            Training labels
        base_model : sklearn model, optional
            Pre-trained base model. If None, will train a RandomForest.
        feature_types : list, optional
            List of feature types ('C', 'N', 'D', 'B')
        label_encoders : dict, optional
            Label encoders for categorical features
        immutable_features : list, optional
            List of immutable feature indices

        Returns
        -------
        self
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        # Convert to numpy if needed
        if isinstance(X_train, pd.DataFrame):
            self.X_train = X_train.values
        else:
            self.X_train = X_train

        if isinstance(y_train, pd.Series):
            y_train = y_train.values

        # Store classes
        self.classes_ = np.unique(y_train)
        assert len(self.classes_) == 2, f"Binary classification required, got {len(self.classes_)} classes"

        # Store feature info
        if feature_types is None:
            # Assume all numerical
            self.feature_types = ['N'] * self.X_train.shape[1]
        else:
            self.feature_types = feature_types

        self.label_encoders = label_encoders or {}
        self.immutable_features = immutable_features or []

        # Train base model if not provided
        if base_model is None:
            base_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.random_state
            )
            base_model.fit(self.X_train, y_train)

        self.base_model = base_model

        # Generate perturbation ensemble
        self.ensemble = self._generate_ensemble(self.X_train, y_train)

        return self

    def _generate_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> List:
        """
        Generate perturbation ensemble for robustness validation.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels

        Returns
        -------
        list
            List of perturbed models
        """
        models = []

        # Base model parameters (approximate)
        if hasattr(self.base_model, 'n_estimators'):
            base_n_estimators = self.base_model.n_estimators
        else:
            base_n_estimators = 100

        if hasattr(self.base_model, 'max_depth'):
            base_max_depth = self.base_model.max_depth or 5
        else:
            base_max_depth = 5

        # Perturbation bins (matching evaluation)
        perturbation_bins = [5, 10, 15, 20, 25, 30]  # Minor deletion percentages

        for k in range(self.n_ensemble):
            # Randomly pick a perturbation level
            bin_num = random.choice(perturbation_bins)
            remove_fraction = bin_num / 100.0
            remove_count = int(len(X_train) * remove_fraction)

            # Simple data perturbation: remove samples from beginning
            if remove_count > 0 and remove_count < len(X_train):
                X_pert = X_train[remove_count:]
                y_pert = y_train[remove_count:]
            else:
                X_pert = X_train
                y_pert = y_train

            # Also vary hyperparameters slightly
            n_est = max(10, base_n_estimators + random.randint(-30, 30))
            max_d = max(2, min(10, base_max_depth + random.randint(-1, 2)))

            rs = self.random_state + k if self.random_state else None
            m_k = self._create_model_like_base(n_est, max_d, rs)
            m_k.fit(X_pert, y_pert)
            models.append(m_k)

        return models

    def _create_model_like_base(self, n_estimators, max_depth, random_state):
        """Create a model of the same type as the base model with given hyperparameters."""
        try:
            model_type = get_model_type(self.base_model)
        except ValueError:
            model_type = 'random_forest'

        if model_type == 'xgboost' and xgboost is not None:
            return xgboost.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )
        elif model_type == 'lightgbm' and lightgbm is not None:
            return lightgbm.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1,
            )
        elif model_type == 'adaboost':
            try:
                return AdaBoostClassifier(
                    estimator=DecisionTreeClassifier(max_depth=max_depth),
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
            except TypeError:
                # Older sklearn uses base_estimator instead of estimator
                return AdaBoostClassifier(
                    base_estimator=DecisionTreeClassifier(max_depth=max_depth),
                    n_estimators=n_estimators,
                    random_state=random_state,
                )
        else:
            return RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
            )

    def _compute_consensus(
        self,
        x_cf: np.ndarray,
        target_class: int
    ) -> float:
        """
        Compute consensus across perturbation ensemble.

        Parameters
        ----------
        x_cf : np.ndarray
            Counterfactual instance
        target_class : int
            Target class

        Returns
        -------
        float
            Consensus score in [0, 1]
        """
        if not self.ensemble:
            return 1.0

        votes = 0
        for m_k in self.ensemble:
            pred = m_k.predict(x_cf.reshape(1, -1))[0]
            if pred == target_class:
                votes += 1

        return votes / len(self.ensemble)

    def _compute_margin(
        self,
        x_cf: np.ndarray,
        target_class: int
    ) -> float:
        """
        Compute margin on base model.

        Parameters
        ----------
        x_cf : np.ndarray
            Counterfactual instance
        target_class : int
            Target class

        Returns
        -------
        float
            Margin score
        """
        try:
            mt = get_model_type(self.base_model)
        except ValueError:
            mt = 'random_forest'
        if mt in ('xgboost', 'lightgbm'):
            return margin_boost(self.base_model, x_cf, target_class)
        return margin_rf(self.base_model, x_cf, target_class)

    def _compute_proximity(
        self,
        x_cf: np.ndarray,
        x_original: np.ndarray
    ) -> float:
        """
        Compute proximity (HEOM distance) between CF and original.

        Parameters
        ----------
        x_cf : np.ndarray
            Counterfactual instance
        x_original : np.ndarray
            Original instance

        Returns
        -------
        float
            Proximity score (lower is better)
        """
        return heom_distance(x_cf, x_original, self.feature_types, self.label_encoders)

    def _compute_sparsity(
        self,
        x_cf: np.ndarray,
        x_original: np.ndarray
    ) -> float:
        """
        Compute sparsity (L0 distance) between CF and original.

        Parameters
        ----------
        x_cf : np.ndarray
            Counterfactual instance
        x_original : np.ndarray
            Original instance

        Returns
        -------
        float
            Sparsity score (number of changed features)
        """
        return l0_distance(x_cf, x_original)

    def _score_candidate(
        self,
        x_cf: np.ndarray,
        x_original: np.ndarray,
        target_class: int
    ) -> Tuple[float, Dict]:
        """
        Score a counterfactual candidate using the full objective.

        Parameters
        ----------
        x_cf : np.ndarray
            Counterfactual candidate
        x_original : np.ndarray
            Original instance
        target_class : int
            Target class

        Returns
        -------
        tuple
            (total_score, dict_of_components)
        """
        consensus = self._compute_consensus(x_cf, target_class)
        margin = self._compute_margin(x_cf, target_class)
        proximity = self._compute_proximity(x_cf, x_original)
        sparsity = self._compute_sparsity(x_cf, x_original)

        # Normalize proximity and sparsity to [0, 1] range
        # Assume features are MinMax scaled, so max distance is sqrt(n_features)
        n_features = len(x_cf)
        max_distance = np.sqrt(n_features)
        norm_proximity = proximity / max_distance if max_distance > 0 else 0
        norm_sparsity = sparsity / n_features if n_features > 0 else 0

        # Compute total score
        # Higher is better for consensus and margin
        # Lower is better for proximity and sparsity
        score = (
            self.w_c * consensus +
            self.w_m * margin -
            self.w_p * norm_proximity -
            self.w_s * norm_sparsity
        )

        return score, {
            'consensus': consensus,
            'margin': margin,
            'proximity': proximity,
            'sparsity': sparsity,
            'norm_proximity': norm_proximity,
            'norm_sparsity': norm_sparsity
        }

    def _generate_single(
        self,
        x: np.ndarray,
        target_class: int
    ) -> Optional[np.ndarray]:
        """
        Generate a single counterfactual for one instance.

        Parameters
        ----------
        x : np.ndarray
            Original instance
        target_class : int
            Target class

        Returns
        -------
        np.ndarray or None
            Best counterfactual candidate, or None if none found
        """
        # Calculate MCTS budget per tree
        n_trees = get_n_estimators(self.base_model)
        budget_per_tree = max(10, self.mcts_budget // n_trees)

        # Run MCTS on the forest
        forest_mcts = ForestMCTS(
            model=self.base_model,
            target_class=target_class,
            x_original=x,
            feature_types=self.feature_types,
            label_encoders=self.label_encoders,
            immutable_features=self.immutable_features,
            budget_per_tree=budget_per_tree,
            exploration_weight=1.4,
            random_state=self.random_state
        )

        candidates = forest_mcts.search(n_candidates_per_tree=self.n_candidates_per_tree)

        if not candidates:
            return None

        # Filter candidates: must achieve target class on base model
        valid_candidates = []
        for x_cf in candidates:
            pred = self.base_model.predict(x_cf.reshape(1, -1))[0]
            if pred == target_class:
                valid_candidates.append(x_cf)

        if not valid_candidates:
            # Relax: try to find the best candidate even if not valid
            # This can happen when trees disagree
            valid_candidates = candidates

        # Score all valid candidates
        scored_candidates = []
        for x_cf in valid_candidates:
            score, components = self._score_candidate(x_cf, x, target_class)
            scored_candidates.append((x_cf, score, components))

        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select best candidate that meets consensus threshold
        for x_cf, score, components in scored_candidates:
            if components['consensus'] >= self.tau:
                return x_cf

        # If no candidate meets threshold, return best candidate anyway
        if scored_candidates:
            return scored_candidates[0][0]

        return None

    def generate_counterfactuals(
        self,
        x_test,
        model=None,
        class_labels: List = None
    ):
        """
        Generate counterfactuals for a test set.

        Matches the signature of FeatureTweakSimple.generate_counterfactuals.

        Parameters
        ----------
        x_test : pd.DataFrame or np.ndarray
            Test instances
        model : sklearn model, optional
            Trained model (if None, uses self.base_model)
        class_labels : list, optional
            List of class labels (default: [0, 1])

        Returns
        -------
        cf_list : pd.DataFrame
            DataFrame with feature columns + ['cf_class', 'success']
        success_rate : float
            Proportion of successful CF generations
        """
        if class_labels is None:
            if self.classes_ is not None:
                class_labels = list(self.classes_)
            else:
                class_labels = [0, 1]

        assert len(class_labels) == 2, "Binary classification only"

        # Use provided model or base model
        if model is not None:
            self.base_model = model
            # Rebuild ensemble for new model
            if self.X_train is not None:
                # Get y_train from stored data
                y_train = self.base_model.predict(self.X_train)
                self.ensemble = self._generate_ensemble(self.X_train, y_train)

        if isinstance(x_test, pd.DataFrame):
            x_test_array = x_test.values
            feature_names = x_test.columns.tolist()
        else:
            x_test_array = x_test
            feature_names = [f'feature_{i}' for i in range(x_test_array.shape[1])]

        # Initialize results
        cf_list = pd.DataFrame(columns=feature_names + ['cf_class', 'success'])
        successful_cfs = 0

        for i in range(len(x_test_array)):
            x = x_test_array[i]
            original_pred = self.base_model.predict(x.reshape(1, -1))[0]
            target_class = get_opposite_class(original_pred, class_labels)

            try:
                x_cf = self._generate_single(x, target_class)

                if x_cf is not None:
                    cf_pred = self.base_model.predict(x_cf.reshape(1, -1))[0]

                    if cf_pred == target_class:
                        cf_row = list(x_cf) + [cf_pred, True]
                        successful_cfs += 1
                    else:
                        # CF doesn't achieve target class
                        cf_row = list(x) + [original_pred, False]
                else:
                    # No CF found
                    cf_row = list(x) + [original_pred, False]

            except Exception as e:
                # Error during generation
                cf_row = list(x) + [original_pred, False]

            cf_list.loc[i] = cf_row

        success_rate = successful_cfs / len(x_test_array) if len(x_test_array) > 0 else 0
        return cf_list, success_rate

    def fit_and_generate(
        self,
        X_train,
        y_train,
        X_test,
        feature_types: List[str] = None,
        label_encoders: Dict = None,
        immutable_features: List[int] = None
    ):
        """
        Convenience method to fit and generate counterfactuals in one call.

        Parameters
        ----------
        X_train : np.ndarray or pd.DataFrame
            Training features
        y_train : np.ndarray or pd.Series
            Training labels
        X_test : np.ndarray or pd.DataFrame
            Test instances for CF generation
        feature_types : list, optional
            Feature types
        label_encoders : dict, optional
            Label encoders
        immutable_features : list, optional
            Immutable feature indices

        Returns
        -------
        cf_list : pd.DataFrame
            Counterfactuals
        success_rate : float
            Success rate
        """
        self.fit(
            X_train, y_train,
            feature_types=feature_types,
            label_encoders=label_encoders,
            immutable_features=immutable_features
        )

        return self.generate_counterfactuals(X_test)

"""
CERTS MCTS Implementation

Monte Carlo Tree Search for counterfactual candidate generation on decision trees.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, Tuple, Set, List, Optional, Any
from collections import defaultdict

try:
    from .tree_utils import extract_trees
except ImportError:
    from tree_utils import extract_trees


@dataclass
class MCTSState:
    """State representation for MCTS search on a decision tree."""
    node_id: int                           # Current node in tree
    constraints: Dict[int, Any] = field(default_factory=dict)  # feature_idx -> constraint
    is_terminal: bool = False
    leaf_class: int = None                 # If terminal, the leaf's prediction

    def __hash__(self):
        # Create a hashable representation of the state
        constraint_tuple = tuple(
            (k, tuple(v) if isinstance(v, set) else v)
            for k, v in sorted(self.constraints.items())
        )
        return hash((self.node_id, constraint_tuple, self.is_terminal, self.leaf_class))

    def __eq__(self, other):
        if not isinstance(other, MCTSState):
            return False
        return (
            self.node_id == other.node_id and
            self.constraints == other.constraints and
            self.is_terminal == other.is_terminal and
            self.leaf_class == other.leaf_class
        )


class TreeMCTS:
    """
    MCTS-based candidate generation for a single decision tree.

    Uses UCB1 for selection and random rollouts for simulation.
    """

    def __init__(
        self,
        tree,
        target_class: int,
        x_original: np.ndarray,
        feature_types: List[str],
        label_encoders: Dict = None,
        immutable_features: List[int] = None,
        budget: int = 500,
        exploration_weight: float = 1.4,
        random_state: int = None
    ):
        """
        Initialize TreeMCTS.

        Parameters
        ----------
        tree : sklearn DecisionTreeClassifier or estimator
            The decision tree to search
        target_class : int
            Target class for counterfactual
        x_original : np.ndarray
            Original instance to generate counterfactual for
        feature_types : list
            List of feature types ('C', 'N', 'D', 'B')
        label_encoders : dict, optional
            Label encoders for categorical features
        immutable_features : list, optional
            List of feature indices that cannot be changed
        budget : int, default=500
            Number of MCTS iterations
        exploration_weight : float, default=1.4
            UCB1 exploration parameter
        random_state : int, optional
            Random seed for reproducibility
        """
        self.tree = tree
        self.target_class = target_class
        self.x = x_original.copy()
        self.feature_types = feature_types
        self.label_encoders = label_encoders or {}
        self.immutable_features = set(immutable_features or [])
        self.budget = budget
        self.c = exploration_weight

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        # Tree structure
        self.children_left = tree.tree_.children_left
        self.children_right = tree.tree_.children_right
        self.feature = tree.tree_.feature
        self.threshold = tree.tree_.threshold
        self.value = tree.tree_.value
        self.n_node_samples = tree.tree_.n_node_samples

    def search(self, n_candidates: int = 5) -> List[np.ndarray]:
        """
        Run MCTS and return top-K candidate counterfactuals from this tree.

        Parameters
        ----------
        n_candidates : int, default=5
            Number of candidates to return

        Returns
        -------
        list
            List of candidate counterfactual arrays
        """
        root = MCTSState(node_id=0, constraints={}, is_terminal=False)

        # MCTS statistics
        visits = defaultdict(int)
        values = defaultdict(float)
        children = {}  # state -> [child_states]

        for _ in range(self.budget):
            # Selection + Expansion
            path, leaf_state = self._select_expand(root, visits, values, children)

            # Simulation (rollout to a leaf)
            terminal_state = self._simulate(leaf_state)

            # Backpropagation
            reward = self._evaluate(terminal_state)
            self._backpropagate(path, reward, visits, values)

        # Extract top-K candidates from explored states
        candidates = self._extract_candidates(root, children, values, visits, n_candidates)
        return candidates

    def _select_expand(
        self,
        state: MCTSState,
        visits: Dict,
        values: Dict,
        children: Dict
    ) -> Tuple[List[MCTSState], MCTSState]:
        """UCB1 selection with expansion."""
        path = [state]
        current = state

        while not current.is_terminal:
            if current not in children:
                # Expand: create children for left/right at split node
                children[current] = self._get_children(current)

            if not children[current]:
                break  # No valid children (constraints infeasible)

            # Check if all children have been visited at least once
            unvisited = [c for c in children[current] if visits[c] == 0]

            if unvisited:
                # Expand: visit an unvisited child
                best_child = random.choice(unvisited)
            else:
                # UCB1 selection
                best_child = max(
                    children[current],
                    key=lambda c: self._ucb1(c, current, visits, values)
                )

            path.append(best_child)
            current = best_child

        return path, current

    def _ucb1(
        self,
        child: MCTSState,
        parent: MCTSState,
        visits: Dict,
        values: Dict
    ) -> float:
        """Calculate UCB1 value for a child state."""
        if visits[child] == 0:
            return float('inf')

        exploitation = values[child] / visits[child]
        exploration = self.c * np.sqrt(np.log(visits[parent] + 1) / visits[child])

        return exploitation + exploration

    def _get_children(self, state: MCTSState) -> List[MCTSState]:
        """Generate child states for left/right branches."""
        node = state.node_id
        feat = self.feature[node]
        thresh = self.threshold[node]

        # Check if this is a leaf node
        if feat < 0 or self.children_left[node] == self.children_right[node]:
            # Leaf node
            leaf_class = int(np.argmax(self.value[node]))
            return [MCTSState(
                node_id=node,
                constraints=state.constraints.copy(),
                is_terminal=True,
                leaf_class=leaf_class
            )]

        # Check if feature is immutable
        if feat in self.immutable_features:
            # Can only go the direction consistent with x_original
            if self.x[feat] <= thresh:
                # Only left child is valid
                left_constraints = self._intersect_constraint(
                    state.constraints, feat, upper=thresh
                )
                if left_constraints is not None:
                    return [MCTSState(
                        node_id=self.children_left[node],
                        constraints=left_constraints,
                        is_terminal=False
                    )]
            else:
                # Only right child is valid
                right_constraints = self._intersect_constraint(
                    state.constraints, feat, lower=thresh
                )
                if right_constraints is not None:
                    return [MCTSState(
                        node_id=self.children_right[node],
                        constraints=right_constraints,
                        is_terminal=False
                    )]
            return []

        children = []

        # Left child: x[feat] <= thresh
        left_constraints = self._intersect_constraint(
            state.constraints, feat, upper=thresh
        )
        if left_constraints is not None:
            children.append(MCTSState(
                node_id=self.children_left[node],
                constraints=left_constraints,
                is_terminal=False
            ))

        # Right child: x[feat] > thresh
        right_constraints = self._intersect_constraint(
            state.constraints, feat, lower=thresh
        )
        if right_constraints is not None:
            children.append(MCTSState(
                node_id=self.children_right[node],
                constraints=right_constraints,
                is_terminal=False
            ))

        return children

    def _intersect_constraint(
        self,
        constraints: Dict[int, Any],
        feat: int,
        lower: float = None,
        upper: float = None
    ) -> Optional[Dict[int, Any]]:
        """Intersect new constraint with existing. Return None if infeasible."""
        new_constraints = {k: v.copy() if isinstance(v, set) else v
                          for k, v in constraints.items()}

        ftype = self.feature_types[feat] if feat < len(self.feature_types) else 'N'

        if ftype == 'C':
            # Categorical (label-encoded)
            # Determine number of categories
            n_cats = 100  # Default fallback
            for col, le in self.label_encoders.items():
                # Check if this encoder corresponds to this feature
                # This is a simplified check - in practice, you'd need proper mapping
                if hasattr(le, 'classes_'):
                    n_cats = max(n_cats, len(le.classes_))

            existing = new_constraints.get(feat, set(range(n_cats)))

            if upper is not None:
                # x <= thresh means integer values {0, 1, ..., floor(thresh)}
                valid = set(range(int(upper) + 1))
                existing = existing & valid

            if lower is not None:
                # x > thresh means integer values {ceil(thresh+1), ...}
                valid = set(range(int(lower) + 1, n_cats))
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

    def _simulate(self, state: MCTSState) -> MCTSState:
        """Rollout: random walk to a leaf."""
        if state.is_terminal:
            return state

        current = state
        max_depth = 50  # Prevent infinite loops

        for _ in range(max_depth):
            if current.is_terminal:
                return current

            children = self._get_children(current)
            if not children:
                # No valid children, create terminal state
                node = current.node_id
                if self.feature[node] < 0:
                    leaf_class = int(np.argmax(self.value[node]))
                else:
                    # Not a leaf but no valid children - use current prediction
                    leaf_class = int(np.argmax(self.value[node]))
                return MCTSState(
                    node_id=node,
                    constraints=current.constraints.copy(),
                    is_terminal=True,
                    leaf_class=leaf_class
                )

            current = random.choice(children)

        return current

    def _evaluate(self, terminal_state: MCTSState) -> float:
        """Reward: high if reaches target class with low feature changes."""
        if not terminal_state.is_terminal:
            return 0.0

        # Base reward: 1 if correct class, 0 otherwise
        class_reward = 1.0 if terminal_state.leaf_class == self.target_class else 0.0

        # Sparsity bonus: fewer constrained features = better
        n_constrained = len(terminal_state.constraints)
        sparsity_bonus = 1.0 / (1.0 + n_constrained)

        # Proximity bonus: prefer constraints that are closer to original values
        proximity_bonus = 0.0
        n_features_checked = 0
        for feat, constraint in terminal_state.constraints.items():
            if feat >= len(self.x):
                continue
            ftype = self.feature_types[feat] if feat < len(self.feature_types) else 'N'
            orig_val = self.x[feat]

            if ftype == 'C':
                # Categorical: bonus if original value is valid
                if int(orig_val) in constraint:
                    proximity_bonus += 1.0
            else:
                # Numerical: bonus based on how much of the range is available
                lo, hi = constraint
                range_size = hi - lo
                proximity_bonus += range_size  # Larger range = less restrictive
            n_features_checked += 1

        if n_features_checked > 0:
            proximity_bonus /= n_features_checked

        return class_reward + 0.3 * sparsity_bonus + 0.2 * proximity_bonus

    def _backpropagate(
        self,
        path: List[MCTSState],
        reward: float,
        visits: Dict,
        values: Dict
    ):
        """Backpropagate reward through the path."""
        for state in path:
            visits[state] += 1
            values[state] += reward

    def _project_to_candidate(self, state: MCTSState) -> np.ndarray:
        """Project x_original into the constraint region."""
        x_cf = self.x.copy()

        for feat, constraint in state.constraints.items():
            if feat >= len(x_cf):
                continue

            ftype = self.feature_types[feat] if feat < len(self.feature_types) else 'N'

            if ftype == 'C':
                # Pick closest valid integer
                valid_set = constraint
                current = int(x_cf[feat])
                if current not in valid_set:
                    x_cf[feat] = min(valid_set, key=lambda v: abs(v - current))
            else:
                lo, hi = constraint
                # Project to midpoint or clip
                if x_cf[feat] < lo:
                    x_cf[feat] = lo + 1e-6
                elif x_cf[feat] > hi:
                    x_cf[feat] = hi - 1e-6

        return x_cf

    def _extract_candidates(
        self,
        root: MCTSState,
        children: Dict,
        values: Dict,
        visits: Dict,
        n: int
    ) -> List[np.ndarray]:
        """Extract top-n candidates from MCTS exploration."""
        # Collect all terminal states that predict target class
        candidates = []

        def collect_terminals(state: MCTSState):
            if state.is_terminal and state.leaf_class == self.target_class:
                x_cf = self._project_to_candidate(state)
                score = values.get(state, 0) / max(visits.get(state, 1), 1)
                candidates.append((x_cf, score, state))

            if state in children:
                for child in children[state]:
                    collect_terminals(child)

        collect_terminals(root)

        # Remove duplicates (candidates that are very similar)
        unique_candidates = []
        for x_cf, score, state in candidates:
            is_duplicate = False
            for existing_cf, _, _ in unique_candidates:
                if np.allclose(x_cf, existing_cf, atol=1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append((x_cf, score, state))

        # Sort by score and return top-n
        unique_candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in unique_candidates[:n]]


class ForestMCTS:
    """
    MCTS-based candidate generation for ensemble of trees.
    """

    def __init__(
        self,
        model,
        target_class: int,
        x_original: np.ndarray,
        feature_types: List[str],
        label_encoders: Dict = None,
        immutable_features: List[int] = None,
        budget_per_tree: int = 100,
        exploration_weight: float = 1.4,
        random_state: int = None
    ):
        """
        Initialize ForestMCTS.

        Parameters
        ----------
        model : tree-based ensemble (RF, XGBoost, LightGBM, AdaBoost)
            The ensemble model
        target_class : int
            Target class for counterfactual
        x_original : np.ndarray
            Original instance
        feature_types : list
            List of feature types
        label_encoders : dict, optional
            Label encoders for categorical features
        immutable_features : list, optional
            List of immutable feature indices
        budget_per_tree : int, default=100
            MCTS budget per tree
        exploration_weight : float, default=1.4
            UCB1 exploration parameter
        random_state : int, optional
            Random seed
        """
        self.model = model
        self.target_class = target_class
        self.x = x_original
        self.feature_types = feature_types
        self.label_encoders = label_encoders or {}
        self.immutable_features = immutable_features or []
        self.budget_per_tree = budget_per_tree
        self.c = exploration_weight
        self.random_state = random_state

    def search(self, n_candidates_per_tree: int = 5) -> List[np.ndarray]:
        """
        Run MCTS on all trees and collect candidates.

        Parameters
        ----------
        n_candidates_per_tree : int, default=5
            Number of candidates to extract from each tree

        Returns
        -------
        list
            Combined list of candidate counterfactuals
        """
        all_candidates = []

        trees = extract_trees(self.model)
        for i, tree in enumerate(trees):
            tree_mcts = TreeMCTS(
                tree=tree,
                target_class=self.target_class,
                x_original=self.x,
                feature_types=self.feature_types,
                label_encoders=self.label_encoders,
                immutable_features=self.immutable_features,
                budget=self.budget_per_tree,
                exploration_weight=self.c,
                random_state=self.random_state + i if self.random_state else None
            )

            candidates = tree_mcts.search(n_candidates=n_candidates_per_tree)
            all_candidates.extend(candidates)

        # Deduplicate
        unique_candidates = []
        for x_cf in all_candidates:
            is_duplicate = False
            for existing in unique_candidates:
                if np.allclose(x_cf, existing, atol=1e-6):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_candidates.append(x_cf)

        return unique_candidates

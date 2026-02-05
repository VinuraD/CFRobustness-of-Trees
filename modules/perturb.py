import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm

class Perturbation:
    """
    Simple perturbation class for robustness testing.
    Works with the simplified DataModule.
    """
    
    def __init__(self, data_module):
        self.data_module = data_module
    
    def get_data(self, fold: int, raw_data: bool = True):
        """
        Get train and test data for specified fold.
        
        Args:
            fold: Fold number (0 to n_splits-1)
            raw_data: If True, return raw data. If False, return preprocessed data.
            
        Returns:
            train_df, test_df: Pandas DataFrames
        """
        return self.data_module.get_data(fold, raw_data)
    
    def get_metadata(self):
        """Get metadata for counterfactual generation."""
        return self.data_module.get_metadata()
    
    def perturb_data(self, train_data: pd.DataFrame, perturb_type: str, bin_number: int):
        """
        Apply data perturbation based on bin number.
        
        Args:
            train_data: Training DataFrame
            perturb_type: Type of perturbation ('minor_deletion', 'major_deletion', 'minor_addition', 'major_addition')
            bin_number: Bin number determining the perturbation level
            
        Returns:
            Perturbed training DataFrame (maintaining order)
        """
        if perturb_type == 'minor_deletion':
            return self._minor_deletion(train_data, bin_number)
        elif perturb_type == 'major_deletion':
            return self._major_deletion(train_data, bin_number)
        elif perturb_type == 'minor_addition':
            return self._minor_addition(train_data, bin_number)
        elif perturb_type == 'major_addition':
            return self._major_addition(train_data, bin_number)
        else:
            raise ValueError(f"Unknown perturbation type: {perturb_type}")
    
    def get_available_bins(self, perturb_type: str):
        """
        Get the number of available bins for a perturbation type.
        
        Args:
            perturb_type: Type of perturbation
            
        Returns:
            Number of available bins
        """
        if perturb_type in ['minor_deletion', 'minor_addition']:
            return 11  # Bins for 0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%, 40%, 45%, 50%
        elif perturb_type in ['major_deletion', 'major_addition']:
            return 2   # Bins 0-1
        else:
            raise ValueError(f"Unknown perturbation type: {perturb_type}")
    
    def perturb_model(self, train_data: pd.DataFrame, model_type: str, max_depth: int, n_estimators: int):
        """
        Train a model with perturbed hyperparameters using 80% of training data.
        
        Args:
            train_data: Training DataFrame
            model_type: Type of model ('xgboost', 'lightgbm', 'random_forest', 'adaboost')
            max_depth: Maximum depth parameter
            n_estimators: Number of estimators parameter
            
        Returns:
            Trained model with perturbed hyperparameters
        """
        # Use 80% of training data
        train_size = int(len(train_data) * 0.8)
        model_train_data = train_data.iloc[:train_size].reset_index(drop=True)
        
        # Get metadata for label column
        metadata = self.get_metadata()
        label_col = metadata['label_column']
        
        # Prepare features and target
        X_train = model_train_data.drop(columns=[label_col])
        y_train = model_train_data[label_col]
        
        # Create model based on type
        model = self._create_model(model_type, max_depth, n_estimators)
        
        # Train the model
        model.fit(X_train, y_train)
        
        return model
    
    def _minor_deletion(self, train_data: pd.DataFrame, bin_number: int):
        """
        Minor deletion: Remove bin_number% of data from the END (use last 50% for perturbations).
        Bin 0: Remove 0%, Bin 5: Remove 5%, ..., Bin 50: Remove 50%
        For perturbations, remove from the last 50% of data.
        """
        if bin_number == 0:
            return train_data.copy()

        # Use the last 50% of data for perturbations
        total_size = len(train_data)
        last_50_percent_start = total_size // 2
        last_50_percent = train_data.iloc[last_50_percent_start:]

        # Remove bin_number% from the last 50%
        remove_fraction = bin_number / 100.0
        remove_count = int(len(last_50_percent) * remove_fraction)

        if remove_count >= len(last_50_percent):
            remove_count = len(last_50_percent) - 1  # Keep at least 1 sample

        # Keep first 50% + remaining from last 50%
        first_50_percent = train_data.iloc[:last_50_percent_start]
        remaining_last_50 = last_50_percent.iloc[:-remove_count] if remove_count > 0 else last_50_percent

        result = pd.concat([first_50_percent, remaining_last_50], ignore_index=True)
        return result
    
    def _major_deletion(self, train_data: pd.DataFrame, bin_number: int):
        """
        Major deletion: Bin 0: Remove 0%, Bin 1: Remove 50%
        """
        if not 0 <= bin_number <= 1:
            raise ValueError(f"Major deletion bin must be 0 or 1, got {bin_number}")
        
        if bin_number == 0:
            return train_data.copy()
        else:  # bin_number == 1
            # Remove 50% from the beginning
            remove_count = len(train_data) // 2
            return train_data.iloc[remove_count:].reset_index(drop=True)
    
    def _minor_addition(self, train_data: pd.DataFrame, bin_number: int):
        """
        Minor addition: Start with 50% data, progressively add from the last 50%.
        Bin 0: Use 50%, Bin 5: Use 55%, ..., Bin 50: Use 100%
        Additions come from the last 50% of data.
        """
        total_size = len(train_data)
        first_50_percent = train_data.iloc[:total_size // 2]
        last_50_percent = train_data.iloc[total_size // 2:]

        if bin_number == 0:
            # Use only first 50%
            return first_50_percent.reset_index(drop=True)

        # Add bin_number% from the last 50%
        add_fraction = bin_number / 100.0
        add_count = int(len(last_50_percent) * add_fraction)
        add_count = min(add_count, len(last_50_percent))  # Don't exceed available data

        additional_data = last_50_percent.iloc[:add_count]
        result = pd.concat([first_50_percent, additional_data], ignore_index=True)
        return result
    
    def _major_addition(self, train_data: pd.DataFrame, bin_number: int):
        """
        Major addition: Bin 0: Use 50%, Bin 1: Use 100%
        """
        if not 0 <= bin_number <= 1:
            raise ValueError(f"Major addition bin must be 0 or 1, got {bin_number}")
        
        if bin_number == 0:
            # Use 50% of data
            keep_count = len(train_data) // 2
            return train_data.iloc[:keep_count].reset_index(drop=True)
        else:  # bin_number == 1
            # Use 100% of data
            return train_data.copy()
    
    def _create_model(self, model_type: str, max_depth: int, n_estimators: int):
        """Create a model with specified hyperparameters."""
        model_type = model_type.lower()
        
        try:
            if model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == 'xgboost':
                return xgb.XGBClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
            elif model_type == 'lightgbm':
                # LightGBM uses num_leaves instead of max_depth primarily
                # But max_depth is still supported
                return lightgbm.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    verbose=-1  # Suppress warnings
                )
            elif model_type == 'adaboost':
                # AdaBoost doesn't have max_depth parameter directly
                # It uses base_estimator's max_depth
                try:
                    from sklearn.tree import DecisionTreeClassifier
                    base_estimator = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                    return AdaBoostClassifier(
                        base_estimator=base_estimator,
                        n_estimators=n_estimators,
                        random_state=42
                    )
                except Exception as e:
                    print(f"Warning: AdaBoost with max_depth failed: {e}")
                    # Fallback: AdaBoost without max_depth specification
                    return AdaBoostClassifier(
                        n_estimators=n_estimators,
                        random_state=42
                    )
            elif model_type == 'catboost':
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=n_estimators,
                    depth=max_depth,
                    random_seed=42,
                    verbose=0
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}. "
                               f"Supported types: 'random_forest', 'xgboost', 'lightgbm', 'adaboost', 'catboost'")
        
        except ImportError as e:
            raise ImportError(f"Model {model_type} requires additional packages: {e}")
        except Exception as e:
            raise RuntimeError(f"Error creating {model_type} model: {e}")


# Simple usage example
if __name__ == "__main__":
    from data_module import DataModule
    
    print("=" * 60)
    print("NEW BIN-BASED PERTURBATION FRAMEWORK TEST")
    print("=" * 60)
    
    # Load dataset
    dm = DataModule("../data/German-Credit.csv", n_splits=5, random_state=42)
    perturbation = Perturbation(dm)
    
    # Get data for fold 0
    train, test = perturbation.get_data(fold=0, raw_data=True)
    print(f"Original data - Train: {train.shape}, Test: {test.shape}")
    
    print("\n--- DATA PERTURBATIONS ---")
    
    # Test minor deletion (bins 0-20)
    print("Minor deletion examples:")
    for bin_num in [0, 5, 10, 20]:
        perturbed_train = perturbation.perturb_data(train, 'minor_deletion', bin_num)
        removed_pct = bin_num
        print(f"  Bin {bin_num}: Remove {removed_pct}% -> {perturbed_train.shape[0]} samples")
    
    # Test major deletion (bins 0-1)
    print("\nMajor deletion examples:")
    for bin_num in [0, 1]:
        perturbed_train = perturbation.perturb_data(train, 'major_deletion', bin_num)
        removed_pct = 0 if bin_num == 0 else 50
        print(f"  Bin {bin_num}: Remove {removed_pct}% -> {perturbed_train.shape[0]} samples")
    
    # Test minor addition (bins 0-20)
    print("\nMinor addition examples:")
    for bin_num in [0, 5, 10, 20]:
        perturbed_train = perturbation.perturb_data(train, 'minor_addition', bin_num)
        use_pct = 80 + bin_num
        print(f"  Bin {bin_num}: Use {use_pct}% -> {perturbed_train.shape[0]} samples")
    
    # Test major addition (bins 0-1)
    print("\nMajor addition examples:")
    for bin_num in [0, 1]:
        perturbed_train = perturbation.perturb_data(train, 'major_addition', bin_num)
        use_pct = 50 if bin_num == 0 else 100
        print(f"  Bin {bin_num}: Use {use_pct}% -> {perturbed_train.shape[0]} samples")
    
    print("\n--- MODEL PERTURBATIONS ---")
    
    # Test model perturbations
    try:
        print("Testing Random Forest model creation...")
        rf_model = perturbation.perturb_model(train, 'random_forest', max_depth=5, n_estimators=100)
        print(f"Random Forest created successfully: {type(rf_model).__name__}")
        
        print("Testing XGBoost model creation...")
        xgb_model = perturbation.perturb_model(train, 'xgboost', max_depth=3, n_estimators=50)
        print(f"XGBoost created successfully: {type(xgb_model).__name__}")
        
    except ImportError as e:
        print(f"Some models not available: {e}")
    except Exception as e:
        print(f"Model creation error: {e}")
    
    # Show available bins for each perturbation type
    print("\n--- AVAILABLE BINS ---")
    for pert_type in ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']:
        bins = perturbation.get_available_bins(pert_type)
        print(f"{pert_type}: {bins} bins (0 to {bins-1})")
    
    print("\n" + "=" * 60)
    print("BIN-BASED PERTURBATION TEST COMPLETED!")
    print("=" * 60)
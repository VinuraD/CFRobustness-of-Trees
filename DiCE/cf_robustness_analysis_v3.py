#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (v3) - German Credit Dataset

This script evaluates the robustness of counterfactual explanations across two separate experiments:
1. Data perturbations - testing how changes in training data affect counterfactual validity
2. Model perturbations - testing how different model types and hyperparameters affect counterfactual validity

The workflow is:
1. Generate counterfactual explanations on unperturbed data with a baseline model
2. Run DATA PERTURBATION tests:
   - Train models with the same architecture on different perturbed datasets
   - Evaluate how valid the original counterfactuals remain
3. Run MODEL PERTURBATION tests:
   - Train different model types on the full unperturbed dataset
   - Evaluate how valid the original counterfactuals remain

This version uses the German Credit dataset which contains heterogeneous features (categorical and numerical).
This helps quantify the independent effects of data and model choices on counterfactual explanation stability.
"""

import sys
import os
import logging
from datetime import datetime
import contextlib
import io
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# DiCE ML imports
import dice_ml
from dice_ml.utils import helpers

from data_module import DataModule
from perturb import Perturbation

# Set up logging
def setup_logging():
    """Setup comprehensive logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cf_robustness_analysis_v3_{timestamp}.log"
    
    logger = logging.getLogger('CFRobustness')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # File handler with timestamps
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter('%(message)s')
    
    file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def log_print(*args, **kwargs):
    """Enhanced print function that logs to both console and file"""
    message = ' '.join(str(arg) for arg in args)
    logger = logging.getLogger('CFRobustness')
    logger.info(message)

def prepare_german_credit_features(df):
    """
    Prepare German Credit features for DiCE, handling categorical and numerical features
    
    Args:
        df: DataFrame with German Credit data
        
    Returns:
        continuous_features: List of continuous feature names
        categorical_features: List of categorical feature names
    """
    # Define categorical and continuous features based on German Credit dataset
    categorical_features = ['Sex', 'Housing', 'SavingAccounts', 'CheckingAccount', 'Purpose']
    
    # Numerical features (including ordinal Job which we'll treat as continuous)
    continuous_features = ['Age', 'Job', 'CreditAmount', 'Duration']
    
    # Ensure all expected features exist in the dataframe
    available_categorical = [f for f in categorical_features if f in df.columns]
    available_continuous = [f for f in continuous_features if f in df.columns]
    
    log_print(f"German Credit feature analysis:")
    log_print(f"  Categorical features: {available_categorical}")
    log_print(f"  Continuous features: {available_continuous}")
    
    return available_continuous, available_categorical

def generate_counterfactuals(x_test, model, dice_data, method='random', total_cfs=2):
    """
    Generate counterfactuals for test set using DiCE
    Returns counterfactuals with success information
    
    Args:
        x_test: Test data (without label)
        model: Trained model
        dice_data: DiCE data object
        method: DiCE method ('random', 'genetic', etc.)
        total_cfs: Number of counterfactuals to generate
        
    Returns:
        cf_list: DataFrame with counterfactuals and success flag
        success_rate: Proportion of successful generations
    """
    log_print(f"Generating counterfactuals using method: {method}")
    log_print(f"Test set size: {len(x_test)} samples")
    
    x_test = x_test.reset_index(drop=True)
    cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
    
    backend = 'sklearn'
    m = dice_ml.Model(model=model, backend=backend)
    exp = dice_ml.Dice(dice_data, m, method=method)
    
    successful_cfs = 0
    failed_cfs = 0
    
    for i in range(len(x_test)):
        query_instance = x_test[i:i+1]
        
        try:
            # Generate counterfactual
            dice_exp = exp.generate_counterfactuals(
                query_instance, 
                total_CFs=total_cfs, 
                desired_class="opposite", 
                verbose=False
            )
            
            # Extract the first counterfactual
            cf_result = dice_exp.cf_examples_list[0].final_cfs_df
            if len(cf_result) > 0:
                cf_values = cf_result.iloc[0].values
                cf_class = cf_values[-1]  # Last column should be the class
                cf_features = cf_values[:-1]  # All but last column
                
                # Store counterfactual
                cf_row = list(cf_features) + [cf_class, True]
                cf_list.loc[i] = cf_row
                successful_cfs += 1
            else:
                # No counterfactual generated, use original with success=False
                default_row = list(query_instance.iloc[0].values) + [0, False]  # Default cf_class to 0
                cf_list.loc[i] = default_row
                failed_cfs += 1
                
        except Exception as e:
            # Failed to generate counterfactual, use original with success=False
            default_row = list(query_instance.iloc[0].values) + [0, False]  # Default cf_class to 0
            cf_list.loc[i] = default_row
            failed_cfs += 1
    
    success_rate = successful_cfs / len(x_test)
    log_print(f"Counterfactual generation complete:")
    log_print(f"  Successful: {successful_cfs}/{len(x_test)} ({success_rate:.2%})")
    log_print(f"  Failed: {failed_cfs}/{len(x_test)} ({(1-success_rate):.2%})")
    
    return cf_list, success_rate

def calculate_comprehensive_metrics(model, cf_list, x_test, x_train):
    """
    Calculate comprehensive counterfactual evaluation metrics
    
    Args:
        model: Model to validate counterfactuals against
        cf_list: DataFrame with counterfactuals
        x_test: Original test data
        x_train: Training data for LOF calculation
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    # Get only successful counterfactuals
    successful_mask = cf_list['success'] == True
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals to validate")
        return {
            'validity': 0.0, 'flipped': 0, 'total': 0,
            'l2_distance': 0.0, 'l0_distance': 0.0, 'lof_score': 0.0
        }
    
    successful_cfs = cf_list[successful_mask]
    corresponding_originals = x_test[successful_mask]
    
    # Remove the success column for prediction
    cf_features = successful_cfs.drop(['success', 'cf_class'], axis=1)
    
    # 1. Validity: Predict on counterfactuals and originals
    cf_predictions = model.predict(cf_features)
    original_predictions = model.predict(corresponding_originals)
    
    # Calculate how many actually flipped
    flipped = (cf_predictions != original_predictions).sum()
    validity = flipped / len(successful_cfs) if len(successful_cfs) > 0 else 0.0
    
    # 2. L2 Distance (Euclidean distance)
    l2_distances = np.sqrt(np.sum((cf_features.values - corresponding_originals.values) ** 2, axis=1))
    avg_l2_distance = np.mean(l2_distances)
    
    # 3. L0 Distance (Number of changed features)
    l0_distances = np.sum(cf_features.values != corresponding_originals.values, axis=1)
    avg_l0_distance = np.mean(l0_distances)
    
    # 4. LOF Score (Local Outlier Factor)
    try:
        from sklearn.neighbors import LocalOutlierFactor
        # Combine training data with counterfactuals for LOF calculation
        combined_data = np.vstack([x_train.values, cf_features.values])
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        lof_scores = lof.fit_predict(combined_data)
        
        # Get LOF scores for counterfactuals (last part of combined_data)
        cf_lof_scores = lof_scores[-len(cf_features):]
        avg_lof_score = np.mean(cf_lof_scores)
    except Exception as e:
        log_print(f"Warning: Could not calculate LOF scores: {e}")
        avg_lof_score = 0.0
    
    return {
        'validity': validity,
        'flipped': flipped,
        'total': len(successful_cfs),
        'l2_distance': avg_l2_distance,
        'l0_distance': avg_l0_distance,
        'lof_score': avg_lof_score
    }

def calculate_validity(model, cf_list, x_test):
    """
    Backward compatibility wrapper for calculate_comprehensive_metrics
    """
    # Create dummy training data if not available
    dummy_train = x_test.copy()
    metrics = calculate_comprehensive_metrics(model, cf_list, x_test, dummy_train)
    return metrics['validity'], metrics['flipped'], metrics['total']


def save_counterfactuals_to_csv(cf_list, cf_method, dataset_name, fold_idx, cf_type="baseline"):
    """
    Save generated counterfactuals to CSV file
    
    Args:
        cf_list: DataFrame with counterfactuals
        cf_method: Name of the CF method (e.g., 'DiCE', 'CEML')
        dataset_name: Name of the dataset (e.g., 'Spambase', 'German-Credit')
        fold_idx: Fold number
        cf_type: Type of counterfactuals ('baseline', 'perturbed', etc.)
    """
    try:
        # Create counterfactuals directory if it doesn't exist
        cf_dir = os.path.join(os.path.dirname(__file__), '..', 'counterfactuals')
        os.makedirs(cf_dir, exist_ok=True)
        
        # Format filename: cf_method__dataset__cf_type__fold#.csv
        filename = f"{cf_method}__{dataset_name}__{cf_type}__fold{fold_idx}.csv"
        filepath = os.path.join(cf_dir, filename)
        
        # Save counterfactuals to CSV
        cf_list.to_csv(filepath, index=False)
        print(f"    Saved counterfactuals to: {filename}")
        
    except Exception as e:
        print(f"    Error saving counterfactuals to CSV: {e}")

def main():
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (v3) - GERMAN CREDIT DATASET")
    log_print("=" * 80)
    log_print(f"[LOG] Logging session to: {log_filename}")
    log_print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    # 1. Load dataset
    log_print("\n1. Loading German Credit dataset...")
    try:
        dm = DataModule("../data/German-Credit.csv", n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Dataset: German Credit")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
        # Display feature types
        log_print(f"Feature types:")
        for feature, ftype in metadata['feature_types'].items():
            log_print(f"  {feature}: {ftype}")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return
    
    # 2. Set analysis parameters for all 5 folds
    log_print("\n2. Analysis parameters:")
    log_print(f"  Number of folds: 5")
    
    # Print fold summary
    import io
    import contextlib
    
    # Capture the output from dm.print_fold_summary()
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dm.print_fold_summary()
    
    # Log each line of the captured output
    summary_output = f.getvalue()
    for line in summary_output.strip().split('\n'):
        if line.strip():
            log_print(line)
    
    # Define data perturbations to test
    # NOTE: Only deletion Bin 0 should match baseline (no data removed)
    # Addition Bin 0 uses less data: minor_addition=80%, major_addition=50%
    data_perturbations = [
        ('minor_deletion', [0, 5, 10, 15, 20]),  # Bin 0 = baseline (0% removed)
        ('major_deletion', [0, 1]),              # Bin 0 = baseline (0% removed)
        ('minor_addition', [0, 5, 10, 15, 20]),  # Bin 0 ≠ baseline (uses 80% of data)
        ('major_addition', [0, 1])               # Bin 0 ≠ baseline (uses 50% of data)
    ]
    log_print(f"  Data perturbation types: {[p[0] for p in data_perturbations]}")
    
    # Define model perturbations to test - NEW SYSTEMATIC APPROACH
    # Test max_depth variations (keep n_estimators fixed at baseline=100)
    # Test n_estimators variations (keep max_depth fixed at baseline=5)
    model_perturbations = []
    
    # Max depth study: Fix n_estimators=100, vary max_depth=[3,4,5,6]
    for max_depth in [3, 4, 5, 6]:
        model_perturbations.extend([
            ('random_forest', max_depth, 100),
            ('xgboost', max_depth, 100),
            ('lightgbm', max_depth, 100),
        ])
    
    # N_estimators study: Fix max_depth=5, vary n_estimators=[50,100,150,200]
    for n_estimators in [50, 100, 150, 200]:
        model_perturbations.extend([
            ('random_forest', 5, n_estimators),
            ('xgboost', 5, n_estimators),
            ('lightgbm', 5, n_estimators),
        ])

    # Add AdaBoost with n_estimators variations (fix max_depth=3 for base estimator)
    for n_estimators in [50, 100, 150, 200]:
        model_perturbations.extend([
            ('adaboost', 3, n_estimators),
        ])
    log_print(f"  Model perturbations: {len(model_perturbations)} configurations")
    for model_type, max_depth, n_estimators in model_perturbations:
        log_print(f"    - {model_type} (max_depth={max_depth}, n_estimators={n_estimators})")
    
    # Dictionary to store results across all folds
    all_fold_results = {
        'baseline_validity': [],
        'baseline_success_rate': [],
        'baseline_model_accuracy': [],
        'baseline_l2_distance': [],
        'baseline_l0_distance': [],
        'baseline_lof_score': [],
        'data_perturbations': {perturb_type: {bin_num: [] for bin_num in bins} 
                              for perturb_type, bins in data_perturbations},
        'model_perturbations': {f"{model_type}_{max_depth}_{n_estimators}": [] 
                               for model_type, max_depth, n_estimators in model_perturbations}
    }
    
    # 3. Run comprehensive analysis across all 5 folds
    log_print("\n3. Running comprehensive analysis across all 5 folds...")
    log_print("=" * 80)
    
    for fold in range(5):
        log_print(f"\n--- FOLD {fold} ANALYSIS ---")
        log_print("-" * 50)
        
        try:
            # Get unperturbed data for this fold
            train_raw, _ = perturbation.get_data(fold=fold, raw_data=True)
            train_processed, test_processed = perturbation.get_data(fold=fold, raw_data=False)
            
            log_print(f"Fold {fold} data shapes:")
            log_print(f"  Raw train: {train_raw.shape}")
            log_print(f"  Processed - Train: {train_processed.shape}, Test: {test_processed.shape}")
            
            # Setup for DiCE
            label_col = metadata['label_column']
            X_train = train_processed.drop(columns=[label_col])
            y_train = train_processed[label_col]
            X_test = test_processed.drop(columns=[label_col])
            y_test = test_processed[label_col]
            
            # Handle categorical labels if needed
            if y_train.dtype == 'object':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train)
                y_test = le.transform(y_test)
            
            log_print(f"  Training samples: {len(X_train)}")
            log_print(f"  Test samples: {len(X_test)}")
            log_print(f"  Features: {len(X_train.columns)}")
            log_print(f"  Class distribution - Train: {np.bincount(y_train)}")
            log_print(f"  Class distribution - Test: {np.bincount(y_test)}")
            
            # Train baseline model on unperturbed data
            log_print(f"\nTraining baseline model for fold {fold}...")
            baseline_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            baseline_model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
            test_acc = accuracy_score(y_test, baseline_model.predict(X_test))
            all_fold_results['baseline_model_accuracy'].append(test_acc)
            
            log_print(f"  Train accuracy: {train_acc:.4f}")
            log_print(f"  Test accuracy: {test_acc:.4f}")
            
            # Setup DiCE data object with heterogeneous features
            log_print(f"Setting up DiCE framework for fold {fold}...")
            
            # Create a combined dataset for DiCE
            train_data_with_label = train_processed.copy()
            
            # Determine continuous and categorical features for German Credit
            continuous_features, categorical_features = prepare_german_credit_features(train_data_with_label)
            
            # Create DiCE data object with proper feature specifications
            dice_data = dice_ml.Data(
                dataframe=train_data_with_label,
                continuous_features=continuous_features,
                outcome_name=label_col
            )
            
            # Generate counterfactuals using baseline model on unperturbed data
            log_print(f"Generating counterfactuals for fold {fold}...")
            
            # Create test data for counterfactual generation (without label)
            test_data_for_cf = X_test.copy()
            
            cf_list, success_rate = generate_counterfactuals(
                test_data_for_cf, 
                baseline_model, 
                dice_data, 
                method='random',
                total_cfs=2
            )
            
            # Save counterfactuals to CSV
            save_counterfactuals_to_csv(cf_list, "DiCE", "German-Credit", fold, "baseline")
            
            all_fold_results['baseline_success_rate'].append(success_rate)
            
            # Calculate comprehensive metrics for baseline model
            baseline_metrics = calculate_comprehensive_metrics(baseline_model, cf_list, X_test, X_train)
            all_fold_results['baseline_validity'].append(baseline_metrics['validity'])
            all_fold_results['baseline_l2_distance'].append(baseline_metrics['l2_distance'])
            all_fold_results['baseline_l0_distance'].append(baseline_metrics['l0_distance'])
            all_fold_results['baseline_lof_score'].append(baseline_metrics['lof_score'])
            
            log_print(f"  Success rate: {success_rate:.2%}")
            log_print(f"  Baseline validity: {baseline_metrics['validity']:.4f} ({baseline_metrics['flipped']}/{baseline_metrics['total']})")
            log_print(f"  Baseline L2 distance: {baseline_metrics['l2_distance']:.4f}")
            log_print(f"  Baseline L0 distance: {baseline_metrics['l0_distance']:.2f}")
            log_print(f"  Baseline LOF score: {baseline_metrics['lof_score']:.4f}")
            
            # Log baseline metrics in standardized format for visualization parsing
            log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics['validity']:.4f}, accuracy: {test_acc:.4f}, L2: {baseline_metrics['l2_distance']:.4f}, L0: {baseline_metrics['l0_distance']:.2f}, LOF: {baseline_metrics['lof_score']:.4f}")
            
            # Test counterfactuals on data perturbed models
            log_print(f"\nTesting data perturbations for fold {fold}...")
            
            for perturb_type, bins in data_perturbations:
                log_print(f"  {perturb_type}:")
                
                for bin_num in bins:
                    try:
                        # Get the raw unperturbed training data
                        train_raw_for_pert, _ = perturbation.get_data(fold=fold, raw_data=True)
                        
                        # Apply perturbation to the raw training data
                        perturbed_train_raw = perturbation.perturb_data(train_raw_for_pert, perturb_type, bin_num)
                        
                        # FIX: Apply the same preprocessing as baseline model
                        # The DataModule already has fitted preprocessors from baseline training
                        # We just need to apply them to the perturbed raw data
                        perturbed_train_processed = perturbation.data_module._preprocess_data(perturbed_train_raw)
                        
                        # Prepare features and target from PROCESSED data
                        perturbed_X_train = perturbed_train_processed.drop(columns=[label_col])
                        perturbed_y_train = perturbed_train_processed[label_col]
                        
                        # Handle categorical labels if needed
                        if perturbed_y_train.dtype == 'object':
                            le_pert = LabelEncoder()
                            perturbed_y_train = le_pert.fit_transform(perturbed_y_train)
                        
                        # Train model on perturbed PROCESSED data (same format as baseline)
                        perturbed_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                        perturbed_model.fit(perturbed_X_train, perturbed_y_train)
                        
                        # Evaluate perturbed model on test set (using processed test data)
                        perturbed_test_acc = accuracy_score(y_test, perturbed_model.predict(X_test))
                        
                        # Calculate comprehensive metrics for perturbed model
                        cf_metrics = calculate_comprehensive_metrics(perturbed_model, cf_list, X_test, perturbed_X_train)
                        
                        # Store results
                        all_fold_results['data_perturbations'][perturb_type][bin_num].append({
                            'validity': cf_metrics['validity'],
                            'model_accuracy': perturbed_test_acc,
                            'l2_distance': cf_metrics['l2_distance'],
                            'l0_distance': cf_metrics['l0_distance'],
                            'lof_score': cf_metrics['lof_score'],
                            'fold': fold
                        })
                        
                        if perturb_type in ['minor_deletion', 'major_deletion']:
                            remove_pct = bin_num if perturb_type == 'minor_deletion' else (0 if bin_num == 0 else 50)
                            log_print(f"    Bin {bin_num}: Remove {remove_pct}% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}, LOF: {cf_metrics['lof_score']:.4f}")
                        else:
                            use_pct = 80 + bin_num if perturb_type == 'minor_addition' else (50 if bin_num == 0 else 100)
                            log_print(f"    Bin {bin_num}: Use {use_pct}% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}, LOF: {cf_metrics['lof_score']:.4f}")
                        
                    except Exception as e:
                        log_print(f"    Error in {perturb_type} bin {bin_num}: {e}")
            
            # Test counterfactuals on model perturbations
            log_print(f"\nTesting model perturbations for fold {fold}...")
            
            for model_type, max_depth, n_estimators in model_perturbations:
                try:
                    # FIX: Use processed data for model perturbations too (same format as baseline)
                    train_processed_for_model, _ = perturbation.get_data(fold=fold, raw_data=False)
                    
                    # Extract features and target from processed data
                    model_X_train = train_processed_for_model.drop(columns=[label_col])
                    model_y_train = train_processed_for_model[label_col]
                    
                    # Handle categorical labels if needed
                    if model_y_train.dtype == 'object':
                        le_model = LabelEncoder()
                        model_y_train = le_model.fit_transform(model_y_train)
                    
                    # Create and train the model with different hyperparameters
                    if model_type == 'random_forest':
                        perturbed_model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                    elif model_type == 'xgboost':
                        import xgboost as xgb
                        perturbed_model = xgb.XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                    elif model_type == 'lightgbm':
                        import lightgbm
                        perturbed_model = lightgbm.LGBMClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42,
                            verbose=-1
                        )
                    elif model_type == 'adaboost':
                        from sklearn.ensemble import AdaBoostClassifier
                        from sklearn.tree import DecisionTreeClassifier
                        # Fix: Use 'estimator' instead of deprecated 'base_estimator'
                        try:
                            # Try new API first (scikit-learn >= 1.2)
                            base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                            perturbed_model = AdaBoostClassifier(
                                estimator=base_tree,
                                n_estimators=n_estimators,
                                random_state=42
                            )
                        except TypeError:
                            # Fallback to old API (scikit-learn < 1.2)
                            base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                            perturbed_model = AdaBoostClassifier(
                                base_estimator=base_tree,
                                n_estimators=n_estimators,
                                random_state=42
                            )
                    else:
                        continue  # Skip unknown model types
                    
                    # Train on processed data (same format as baseline)
                    perturbed_model.fit(model_X_train, model_y_train)
                    
                    # Evaluate model on test set
                    model_test_acc = accuracy_score(y_test, perturbed_model.predict(X_test))
                    
                    # Calculate comprehensive metrics for this model
                    cf_metrics = calculate_comprehensive_metrics(perturbed_model, cf_list, X_test, model_X_train)
                    
                    # Store results
                    model_key = f"{model_type}_{max_depth}_{n_estimators}"
                    all_fold_results['model_perturbations'][model_key].append({
                        'validity': cf_metrics['validity'],
                        'model_accuracy': model_test_acc,
                        'l2_distance': cf_metrics['l2_distance'],
                        'l0_distance': cf_metrics['l0_distance'],
                        'lof_score': cf_metrics['lof_score'],
                        'fold': fold
                    })
                    
                    log_print(f"  {model_type} ({max_depth}, {n_estimators}): validity: {cf_metrics['validity']:.4f}, accuracy: {model_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}, LOF: {cf_metrics['lof_score']:.4f}")
                    
                except Exception as e:
                    log_print(f"  Error with {model_type} ({max_depth}, {n_estimators}): {e}")
        
        except Exception as e:
            log_print(f"Error in fold {fold}: {e}")
            continue
    
    # 4. Generate comprehensive summary and analysis across all folds
    log_print("\n4. Generating comprehensive summary across all 5 folds...")
    log_print("=" * 80)
    
    try:
        # Calculate baseline statistics across folds
        baseline_validity_mean = np.mean(all_fold_results['baseline_validity'])
        baseline_validity_std = np.std(all_fold_results['baseline_validity'])
        baseline_success_mean = np.mean(all_fold_results['baseline_success_rate'])
        baseline_success_std = np.std(all_fold_results['baseline_success_rate'])
        baseline_accuracy_mean = np.mean(all_fold_results['baseline_model_accuracy'])
        baseline_accuracy_std = np.std(all_fold_results['baseline_model_accuracy'])
        baseline_l2_mean = np.mean(all_fold_results['baseline_l2_distance'])
        baseline_l2_std = np.std(all_fold_results['baseline_l2_distance'])
        baseline_l0_mean = np.mean(all_fold_results['baseline_l0_distance'])
        baseline_l0_std = np.std(all_fold_results['baseline_l0_distance'])
        baseline_lof_mean = np.mean(all_fold_results['baseline_lof_score'])
        baseline_lof_std = np.std(all_fold_results['baseline_lof_score'])
        
        log_print("\nBASELINE PERFORMANCE ACROSS ALL FOLDS:")
        log_print("-" * 80)
        log_print(f"Model Accuracy: {baseline_accuracy_mean:.4f} ± {baseline_accuracy_std:.4f}")
        log_print(f"CF Success Rate: {baseline_success_mean:.4f} ± {baseline_success_std:.4f}")
        log_print(f"CF Validity: {baseline_validity_mean:.4f} ± {baseline_validity_std:.4f}")
        log_print(f"CF L2 Distance: {baseline_l2_mean:.4f} ± {baseline_l2_std:.4f}")
        log_print(f"CF L0 Distance: {baseline_l0_mean:.4f} ± {baseline_l0_std:.4f}")
        log_print(f"CF LOF Score: {baseline_lof_mean:.4f} ± {baseline_lof_std:.4f}")
        log_print(f"Individual fold model accuracies: {[f'{acc:.4f}' for acc in all_fold_results['baseline_model_accuracy']]}")
        log_print(f"Individual fold CF success rates: {[f'{rate:.4f}' for rate in all_fold_results['baseline_success_rate']]}")
        log_print(f"Individual fold CF validities: {[f'{val:.4f}' for val in all_fold_results['baseline_validity']]}")
        
        # DATA PERTURBATION SUMMARY
        log_print("\nDATA PERTURBATION ROBUSTNESS SUMMARY:")
        log_print("-" * 150)
        log_print(f"{'Perturbation':<15} {'Bin':<5} {'Data':<12} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11} {'Per-Fold Validities'}")
        log_print("-" * 150)
        
        data_summary_results = {}
        for perturb_type, bins_data in all_fold_results['data_perturbations'].items():
            data_summary_results[perturb_type] = {}
            for bin_num, fold_results in bins_data.items():
                if fold_results:  # If we have results for this bin
                    validities = [r['validity'] for r in fold_results]
                    accuracies = [r['model_accuracy'] for r in fold_results]
                    l2_distances = [r['l2_distance'] for r in fold_results]
                    l0_distances = [r['l0_distance'] for r in fold_results]
                    lof_scores = [r['lof_score'] for r in fold_results]
                    
                    mean_validity = np.mean(validities)
                    std_validity = np.std(validities)
                    mean_accuracy = np.mean(accuracies)
                    std_accuracy = np.std(accuracies)
                    mean_l2 = np.mean(l2_distances)
                    mean_l0 = np.mean(l0_distances)
                    validity_delta = mean_validity - baseline_validity_mean
                    
                    data_summary_results[perturb_type][bin_num] = {
                        'mean_validity': mean_validity,
                        'std_validity': std_validity,
                        'mean_accuracy': mean_accuracy,
                        'std_accuracy': std_accuracy,
                        'mean_l2': mean_l2,
                        'mean_l0': mean_l0,
                        'validity_delta': validity_delta,
                        'validities': validities
                    }
                    
                    if perturb_type in ['minor_deletion', 'major_deletion']:
                        remove_pct = bin_num if perturb_type == 'minor_deletion' else (0 if bin_num == 0 else 50)
                        data_description = f"Remove {remove_pct}%"
                    else:
                        use_pct = 80 + bin_num if perturb_type == 'minor_addition' else (50 if bin_num == 0 else 100)
                        data_description = f"Use {use_pct}%"
                    
                    per_fold_str = [f'{v:.3f}' for v in validities]
                    log_print(f"{perturb_type:<15} {bin_num:<5} {data_description:<12} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:<+11.4f} {per_fold_str}")
        
        # MODEL PERTURBATION SUMMARY
        log_print("\nMODEL PERTURBATION ROBUSTNESS SUMMARY:")
        log_print("-" * 150)
        log_print(f"{'Model Configuration':<30} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11} {'Per-Fold Validities'}")
        log_print("-" * 150)
        
        model_summary_results = {}
        for model_key, fold_results in all_fold_results['model_perturbations'].items():
            if fold_results:  # If we have results for this model
                validities = [r['validity'] for r in fold_results]
                accuracies = [r['model_accuracy'] for r in fold_results]
                l2_distances = [r['l2_distance'] for r in fold_results]
                l0_distances = [r['l0_distance'] for r in fold_results]
                lof_scores = [r['lof_score'] for r in fold_results]
                
                mean_validity = np.mean(validities)
                std_validity = np.std(validities)
                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)
                mean_l2 = np.mean(l2_distances)
                mean_l0 = np.mean(l0_distances)
                validity_delta = mean_validity - baseline_validity_mean
                
                model_summary_results[model_key] = {
                    'mean_validity': mean_validity,
                    'std_validity': std_validity,
                    'mean_accuracy': mean_accuracy,
                    'std_accuracy': std_accuracy,
                    'mean_l2': mean_l2,
                    'mean_l0': mean_l0,
                    'validity_delta': validity_delta,
                    'validities': validities
                }
                
                per_fold_str = [f'{v:.3f}' for v in validities]
                log_print(f"{model_key:<30} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:<+11.4f} {per_fold_str}")
        
        # DETAILED FOLD-BY-FOLD ANALYSIS
        log_print("\n" + "=" * 80)
        log_print("DETAILED FOLD-BY-FOLD ANALYSIS")
        log_print("=" * 80)
        
        for fold in range(5):
            log_print(f"\nFOLD {fold} DETAILED RESULTS:")
            log_print("-" * 50)
            
            if fold < len(all_fold_results['baseline_validity']):
                log_print(f"Baseline - Validity: {all_fold_results['baseline_validity'][fold]:.4f}, "
                         f"Success Rate: {all_fold_results['baseline_success_rate'][fold]:.4f}, "
                         f"Model Accuracy: {all_fold_results['baseline_model_accuracy'][fold]:.4f}")
                
                # Data perturbations for this fold
                log_print("Data Perturbations:")
                for perturb_type, bins_data in all_fold_results['data_perturbations'].items():
                    for bin_num, fold_results in bins_data.items():
                        if len(fold_results) > fold:
                            result = fold_results[fold]
                            log_print(f"  {perturb_type} bin {bin_num}: validity={result['validity']:.4f}, accuracy={result['model_accuracy']:.4f}")
                
                # Model perturbations for this fold
                log_print("Model Perturbations:")
                for model_key, fold_results in all_fold_results['model_perturbations'].items():
                    if len(fold_results) > fold:
                        result = fold_results[fold]
                        log_print(f"  {model_key}: validity={result['validity']:.4f}, accuracy={result['model_accuracy']:.4f}")
        
        # Plot data perturbation results
        plt.figure(figsize=(12, 8))
        
        # Create a different marker for each perturbation type
        markers = {
            'minor_deletion': 'o',
            'major_deletion': 's',
            'minor_addition': '^',
            'major_addition': 'D'
        }
        
        for perturb_type, bin_results in data_summary_results.items():
            if not bin_results:
                continue
                
            # Extract data for plotting
            x_labels = []
            validities = []
            errors = []
            
            for bin_num in sorted(bin_results.keys()):
                result = bin_results[bin_num]
                
                if perturb_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = bin_num if perturb_type == 'minor_deletion' else (0 if bin_num == 0 else 50)
                    x_labels.append(f"{remove_pct}% removed")
                else:
                    use_pct = 80 + bin_num if perturb_type == 'minor_addition' else (50 if bin_num == 0 else 100)
                    x_labels.append(f"{use_pct}% used")
                
                validities.append(result['mean_validity'])
                errors.append(result['std_validity'])
            
            plt.errorbar(x_labels, validities, yerr=errors, marker=markers[perturb_type], 
                        label=perturb_type, linewidth=2, markersize=8, capsize=5)
        
        # Add baseline as horizontal line with error bars
        plt.axhline(y=baseline_validity_mean, color='red', linestyle='--', label='Baseline Validity')
        plt.fill_between(range(len(plt.gca().get_xticks())), 
                        baseline_validity_mean - baseline_validity_std,
                        baseline_validity_mean + baseline_validity_std,
                        color='red', alpha=0.2)
        
        plt.title('Counterfactual Explanation Robustness Across Data Perturbations (German Credit)\n5-Fold Cross-Validation with Error Bars', fontsize=14)
        plt.xlabel('Perturbation Level', fontsize=12)
        plt.ylabel('Counterfactual Validity', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        
        # Save data perturbation plot
        plt.tight_layout()
        data_plot_filename = f"cf_data_robustness_german_5fold_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(data_plot_filename)
        log_print(f"\nData perturbation plot saved to: {data_plot_filename}")
        
        # Plot model perturbation results
        plt.figure(figsize=(12, 8))
        
        # Group by model type
        model_types = set(key.split('_')[0] for key in model_summary_results.keys())
        
        for model_type in model_types:
            model_results = [(key, result) for key, result in model_summary_results.items() 
                           if key.startswith(model_type)]
            if not model_results:
                continue
                
            # Sort by n_estimators for better visualization
            model_results.sort(key=lambda x: int(x[0].split('_')[-1]))
            
            # Create x labels and extract data
            x_labels = []
            validities = []
            errors = []
            
            for key, result in model_results:
                parts = key.split('_')
                max_depth = parts[1]
                n_estimators = parts[2]
                x_labels.append(f"d={max_depth}, n={n_estimators}")
                validities.append(result['mean_validity'])
                errors.append(result['std_validity'])
            
            plt.errorbar(x_labels, validities, yerr=errors, marker='o', 
                        label=model_type, linewidth=2, markersize=8, capsize=5)
        
        # Add baseline as horizontal line with error bars
        plt.axhline(y=baseline_validity_mean, color='red', linestyle='--', label='Baseline Validity')
        plt.fill_between(range(len(plt.gca().get_xticks())), 
                        baseline_validity_mean - baseline_validity_std,
                        baseline_validity_mean + baseline_validity_std,
                        color='red', alpha=0.2)
        
        plt.title('Counterfactual Explanation Robustness Across Model Types (German Credit)\n5-Fold Cross-Validation with Error Bars', fontsize=14)
        plt.xlabel('Model Configuration', fontsize=12)
        plt.ylabel('Counterfactual Validity', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45)
        
        # Save model perturbation plot
        plt.tight_layout()
        model_plot_filename = f"cf_model_robustness_german_5fold_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(model_plot_filename)
        log_print(f"Model perturbation plot saved to: {model_plot_filename}")
        
    except Exception as e:
        log_print(f"Error generating summary visualizations: {e}")
    
    # 5. Final insights and statistical analysis
    log_print("\n" + "=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS INSIGHTS (GERMAN CREDIT DATASET - 5 FOLDS)")
    log_print("=" * 80)
    
    # Calculate average validity change for each perturbation type
    data_avg_changes = {}
    for perturb_type, bin_results in data_summary_results.items():
        if bin_results:  # Skip empty results
            # Skip bin 0 which is unperturbed for deletion types
            perturbed_results = [result for bin_num, result in bin_results.items() 
                               if not (perturb_type in ['minor_deletion', 'major_deletion'] and bin_num == 0)]
            if perturbed_results:
                avg_change = np.mean([result['validity_delta'] for result in perturbed_results])
                data_avg_changes[perturb_type] = avg_change
    
    # Calculate average validity change for each model type
    model_avg_changes = {}
    model_types = set(key.split('_')[0] for key in model_summary_results.keys())
    for model_type in model_types:
        type_results = [result for key, result in model_summary_results.items() 
                       if key.startswith(model_type)]
        if type_results:
            avg_change = np.mean([result['validity_delta'] for result in type_results])
            model_avg_changes[model_type] = avg_change
    
    # Report findings
    log_print("\n[INSIGHTS] DATA PERTURBATION INSIGHTS:")
    if data_avg_changes:
        most_robust_data = min(data_avg_changes.items(), key=lambda x: abs(x[1]))
        least_robust_data = max(data_avg_changes.items(), key=lambda x: abs(x[1]))
        
        log_print(f"• Most robust to: {most_robust_data[0]} (avg validity change: {most_robust_data[1]:+.4f})")
        log_print(f"• Least robust to: {least_robust_data[0]} (avg validity change: {least_robust_data[1]:+.4f})")
        
        # Overall data robustness score (average absolute change across all data perturbations)
        data_robustness_score = 1 - np.mean([abs(change) for change in data_avg_changes.values()])
        log_print(f"• Overall data perturbation robustness score: {data_robustness_score:.4f} (higher is better)")
    
    log_print("\n🤖 MODEL PERTURBATION INSIGHTS:")
    if model_avg_changes:
        most_robust_model = min(model_avg_changes.items(), key=lambda x: abs(x[1]))
        least_robust_model = max(model_avg_changes.items(), key=lambda x: abs(x[1]))
        
        log_print(f"• Most robust to: {most_robust_model[0]} (avg validity change: {most_robust_model[1]:+.4f})")
        log_print(f"• Least robust to: {least_robust_model[0]} (avg validity change: {least_robust_model[1]:+.4f})")
        
        # Overall model robustness score
        model_robustness_score = 1 - np.mean([abs(change) for change in model_avg_changes.values()])
        log_print(f"• Overall model perturbation robustness score: {model_robustness_score:.4f} (higher is better)")
    
    log_print("\n🏛️ GERMAN CREDIT DATASET INSIGHTS:")
    log_print(f"• Dataset contains heterogeneous features: {len(continuous_features)} continuous, {len(categorical_features)} categorical")
    log_print(f"• Categorical features: {categorical_features}")
    log_print(f"• Continuous features: {continuous_features}")
    log_print(f"• Baseline counterfactual success rate: {baseline_success_mean:.2%} ± {baseline_success_std:.3f}")
    log_print(f"• Baseline counterfactual validity: {baseline_validity_mean:.4f} ± {baseline_validity_std:.3f}")
    log_print(f"• Baseline model accuracy: {baseline_accuracy_mean:.4f} ± {baseline_accuracy_std:.3f}")
    log_print(f"• Baseline L2 distance: {baseline_l2_mean:.4f} ± {baseline_l2_std:.4f}")
    log_print(f"• Baseline L0 distance: {baseline_l0_mean:.4f} ± {baseline_l0_std:.4f}")
    log_print(f"• Baseline LOF score: {baseline_lof_mean:.4f} ± {baseline_lof_std:.4f}")
    
    # Statistical significance analysis
    log_print("\n📈 STATISTICAL ANALYSIS (5-FOLD CROSS-VALIDATION):")
    log_print(f"• Confidence intervals calculated from {len(all_fold_results['baseline_validity'])} independent folds")
    log_print(f"• All error bars represent ±1 standard deviation across folds")
    
    # Best/worst performing configurations
    if data_summary_results:
        all_data_configs = []
        for perturb_type, bin_results in data_summary_results.items():
            for bin_num, result in bin_results.items():
                all_data_configs.append((f"{perturb_type}_bin_{bin_num}", result['mean_validity'], result['std_validity']))
        
        if all_data_configs:
            best_data_config = max(all_data_configs, key=lambda x: x[1])
            worst_data_config = min(all_data_configs, key=lambda x: x[1])
            
            log_print(f"• Best data perturbation: {best_data_config[0]} (validity: {best_data_config[1]:.4f} ± {best_data_config[2]:.3f})")
            log_print(f"• Worst data perturbation: {worst_data_config[0]} (validity: {worst_data_config[1]:.4f} ± {worst_data_config[2]:.3f})")
    
    if model_summary_results:
        all_model_configs = [(key, result['mean_validity'], result['std_validity']) 
                           for key, result in model_summary_results.items()]
        
        if all_model_configs:
            best_model_config = max(all_model_configs, key=lambda x: x[1])
            worst_model_config = min(all_model_configs, key=lambda x: x[1])
            
            log_print(f"• Best model configuration: {best_model_config[0]} (validity: {best_model_config[1]:.4f} ± {best_model_config[2]:.3f})")
            log_print(f"• Worst model configuration: {worst_model_config[0]} (validity: {worst_model_config[1]:.4f} ± {worst_model_config[2]:.3f})")
    
    # Experiment summary
    total_data_experiments = sum(len(bins) for _, bins in data_perturbations) * 5  # 5 folds
    total_model_experiments = len(model_perturbations) * 5  # 5 folds
    total_experiments = total_data_experiments + total_model_experiments
    
    log_print(f"\n🔬 EXPERIMENT SUMMARY:")
    log_print(f"• Total data perturbation experiments: {total_data_experiments} ({sum(len(bins) for _, bins in data_perturbations)} configs × 5 folds)")
    log_print(f"• Total model perturbation experiments: {total_model_experiments} ({len(model_perturbations)} configs × 5 folds)")
    log_print(f"• Total experiments conducted: {total_experiments}")
    log_print(f"• All results aggregated with mean ± standard deviation across 5 folds")
    
    # End timing
    end_time = datetime.now()
    log_print(f"\n{'='*80}")
    log_print("🏁 COMPREHENSIVE COUNTERFACTUAL ROBUSTNESS ANALYSIS COMPLETED!")
    log_print(f"{'='*80}")
    log_print(f"[TIME] Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"[LOG] Complete analysis saved to: {log_filename}")
    log_print(f"{'='*80}")
    
    # Summary stats
    total_experiments = (
        sum(len(bins) for _, bins in data_perturbations) +  # Data perturbations
        len(model_perturbations)                           # Model perturbations
    )
    log_print(f"Total experiments run: {total_experiments}")
    log_print(f"Total visualizations created: 2")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print("❌ ANALYSIS TERMINATED DUE TO UNEXPECTED ERROR")
        print(f"{'='*80}")
        print(f"[ERROR] {type(e).__name__}: {str(e)}")
        print(f"[TIME] Error occurred at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        raise

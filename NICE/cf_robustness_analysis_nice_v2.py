#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (NICE v2) - Spambase Dataset

This script evaluates the robustness of counterfactual explanations using the NICE algorithm across two separate experiments:
1. Data perturbations - testing how changes in training data affect counterfactual validity
2. Model perturbations - testing how different model types and hyperparameters affect counterfactual validity

The workflow is:
1. Generate counterfactual explanations using NICE on unperturbed data with a baseline model
2. Run DATA PERTURBATION tests:
   - Train models with the same architecture on different perturbed datasets
   - Evaluate how valid the original counterfactuals remain
3. Run MODEL PERTURBATION tests:
   - Train different model types on the full unperturbed dataset
   - Evaluate how valid the original counterfactuals remain

This version uses the NICE algorithm for counterfactual generation and the Spambase dataset (all continuous features).
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
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# NICE imports
from nice import NICE

from data_module import DataModule
from perturb import Perturbation

# Set up logging
def setup_logging():
    """Setup comprehensive logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cf_robustness_analysis_nice_v2_{timestamp}.log"
    
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

def prepare_spambase_features(x_train):
    """
    Prepare Spambase features for NICE
    Spambase has all continuous features
    
    Args:
        x_train: Training features DataFrame
        
    Returns:
        cat_feat: List of categorical feature indices (empty for Spambase)
        num_feat: List of numerical feature indices (all features for Spambase)
    """
    # Spambase has all continuous/numerical features
    num_feat = list(range(len(x_train.columns)))
    cat_feat = []  # No categorical features in Spambase
    
    log_print(f"Spambase feature analysis:")
    log_print(f"  Categorical feature indices: {cat_feat}")
    log_print(f"  Numerical feature indices: {num_feat[:10]}... (total: {len(num_feat)})")
    
    return cat_feat, num_feat

def generate_counterfactuals_nice(x_test, x_train, y_train, model, cat_feat, num_feat):
    """
    Generate counterfactual explanations using NICE algorithm
    """
    log_print(f"Generating counterfactuals using NICE algorithm")
    log_print(f"Test set size: {len(x_test)} samples")
    
    try:
        # Convert to numpy arrays for NICE
        X_train_array = np.array(x_train)
        y_train_array = np.array(y_train)
        
        # Create predict function that preserves feature names to avoid warnings
        def predict_fn(x):
            # Convert numpy array back to DataFrame with original feature names if available
            if hasattr(x_train, 'columns'):
                x_df = pd.DataFrame(x, columns=x_train.columns)
                return model.predict_proba(x_df)
            else:
                return model.predict_proba(x)
        
        # Initialize NICE explainer
        nice_explainer = NICE(
            X_train=X_train_array,
            predict_fn=predict_fn,
            y_train=y_train_array,
            cat_feat=cat_feat,
            num_feat=num_feat,
            distance_metric='HEOM',  # Heterogeneous Euclidean-Overlap Metric for mixed data
            num_normalization='minmax',
            optimization='proximity',
            justified_cf=True
        )
        
        # Initialize result DataFrame
        cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        successful_cfs = 0
        failed_cfs = 0
        
        for i in range(len(x_test)):
            try:
                # Generate counterfactual for this instance
                instance = x_test.iloc[i:i+1].values
                cf = nice_explainer.explain(instance)
                
                if cf is not None and len(cf) > 0:
                    # Get the first counterfactual
                    cf_instance = cf[0]
                    
                    # Predict class for the counterfactual using feature names to avoid warnings
                    if hasattr(x_train, 'columns'):
                        cf_df = pd.DataFrame(cf_instance.reshape(1, -1), columns=x_train.columns)
                        cf_class = model.predict(cf_df)[0]
                    else:
                        cf_class = model.predict(cf_instance.reshape(1, -1))[0]
                    
                    # Store counterfactual
                    cf_row = list(cf_instance) + [cf_class, True]
                    cf_list.loc[i] = cf_row
                    successful_cfs += 1
                else:
                    # No counterfactual generated, use original with success=False
                    default_row = list(x_test.iloc[i].values) + [y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], False]
                    cf_list.loc[i] = default_row
                    failed_cfs += 1
                    
            except Exception as e:
                logger.warning(f"Failed to generate counterfactual for instance {i}: {str(e)}")
                # Failed to generate counterfactual, use original with success=False
                default_row = list(x_test.iloc[i].values) + [y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], False]
                cf_list.loc[i] = default_row
                failed_cfs += 1
        
        success_rate = successful_cfs / len(x_test) if len(x_test) > 0 else 0
        log_print(f"NICE counterfactual generation complete:")
        log_print(f"  Successful: {successful_cfs}/{len(x_test)} ({success_rate:.2%})")
        log_print(f"  Failed: {failed_cfs}/{len(x_test)} ({(1-success_rate):.2%})")
        
        return cf_list, success_rate
        
    except Exception as e:
        logger.error(f"Error in NICE counterfactual generation: {str(e)}")
        # Return empty DataFrame with proper structure
        cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        return cf_list, 0.0

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
        # Combine training data with counterfactuals for LOF calculation
        combined_data = np.vstack([x_train.values, cf_features.values])
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1)
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

def main():
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (NICE v2) - SPAMBASE DATASET")
    log_print("=" * 80)
    log_print(f"[LOG] Logging session to: {log_filename}")
    log_print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    # 1. Load dataset
    log_print("\n1. Loading dataset...")
    try:
        dm = DataModule("../data/Spambase.csv", n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Dataset: Spambase")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return
    
    # 2. Set analysis parameters for all 5 folds
    log_print("\n2. Analysis parameters:")
    log_print(f"  Number of folds: 5")
    log_print(f"  Counterfactual method: NICE")
    
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
    data_perturbations = [
        ('minor_deletion', [0, 5, 10, 15, 20]),  # Bin 0 = baseline (0% removed)
        ('major_deletion', [0, 1]),              # Bin 0 = baseline (0% removed)
        ('minor_addition', [0, 5, 10, 15, 20]),  # Bin 0 ≠ baseline (uses 80% of data)
        ('major_addition', [0, 1])               # Bin 0 ≠ baseline (uses 50% of data)
    ]
    log_print(f"  Data perturbation types: {[p[0] for p in data_perturbations]}")
    
    # Define model perturbations to test
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

    # Add AdaBoost with n_estimators variations
    for n_estimators in [50, 100, 150, 200]:
        model_perturbations.extend([
            ('adaboost', 3, n_estimators),
        ])
    log_print(f"  Model perturbations: {len(model_perturbations)} configurations")
    
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
            
            # Setup for NICE
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
            
            # Prepare features for NICE
            cat_feat, num_feat = prepare_spambase_features(X_train)
            
            # Train baseline model on unperturbed data
            log_print(f"\nTraining baseline model for fold {fold}...")
            baseline_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            baseline_model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
            test_acc = accuracy_score(y_test, baseline_model.predict(X_test))
            all_fold_results['baseline_model_accuracy'].append(test_acc)
            
            log_print(f"  Train accuracy: {train_acc:.4f}")
            log_print(f"  Test accuracy: {test_acc:.4f}")
            
            # Generate counterfactuals using NICE
            log_print(f"Generating counterfactuals using NICE for fold {fold}...")
            
            cf_list, success_rate = generate_counterfactuals_nice(
                X_test, X_train, y_train, baseline_model, cat_feat, num_feat
            )
            
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
                        
                        # Apply the same preprocessing as baseline model
                        perturbed_train_processed = perturbation.data_module._preprocess_data(perturbed_train_raw)
                        
                        # Prepare features and target from PROCESSED data
                        perturbed_X_train = perturbed_train_processed.drop(columns=[label_col])
                        perturbed_y_train = perturbed_train_processed[label_col]
                        
                        # Handle categorical labels if needed
                        if perturbed_y_train.dtype == 'object':
                            le_pert = LabelEncoder()
                            perturbed_y_train = le_pert.fit_transform(perturbed_y_train)
                        
                        # Train model on perturbed PROCESSED data
                        perturbed_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                        perturbed_model.fit(perturbed_X_train, perturbed_y_train)
                        
                        # Evaluate perturbed model on test set
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
                    # Use processed data for model perturbations
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
                        try:
                            base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                            perturbed_model = AdaBoostClassifier(
                                estimator=base_tree,
                                n_estimators=n_estimators,
                                random_state=42
                            )
                        except TypeError:
                            base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
                            perturbed_model = AdaBoostClassifier(
                                base_estimator=base_tree,
                                n_estimators=n_estimators,
                                random_state=42
                            )
                    else:
                        continue
                    
                    # Train on processed data
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

    # 4. Generate summary and visualization across all folds
    log_print("\n4. Generating comprehensive summary of NICE counterfactual robustness...")
    log_print("=" * 80)
    
    log_print("\nFOLD-BY-FOLD RESULTS AND AGGREGATES:")
    log_print("-" * 100)
    
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
    
    log_print("\nBASELINE PERFORMANCE ACROSS ALL FOLDS (NICE):")
    log_print("-" * 80)
    log_print(f"Model Accuracy: {baseline_accuracy_mean:.4f} ± {baseline_accuracy_std:.4f}")
    log_print(f"CF Success Rate: {baseline_success_mean:.4f} ± {baseline_success_std:.4f}")
    log_print(f"CF Validity: {baseline_validity_mean:.4f} ± {baseline_validity_std:.4f}")
    log_print(f"CF L2 Distance: {baseline_l2_mean:.4f} ± {baseline_l2_std:.4f}")
    log_print(f"CF L0 Distance: {baseline_l0_mean:.4f} ± {baseline_l0_std:.4f}")
    log_print(f"CF LOF Score: {baseline_lof_mean:.4f} ± {baseline_lof_std:.4f}")

    # DATA PERTURBATION SUMMARY
    log_print("\nDATA PERTURBATION ROBUSTNESS SUMMARY (NICE):")
    log_print("-" * 150)
    log_print(f"{'Perturbation':<15} {'Bin':<5} {'Data':<12} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11}")
    log_print("-" * 150)
    
    data_summary_results = {}
    for perturb_type, bins_data in all_fold_results['data_perturbations'].items():
        data_summary_results[perturb_type] = {}
        for bin_num, fold_results in bins_data.items():
            if fold_results:
                validities = [r['validity'] for r in fold_results]
                accuracies = [r['model_accuracy'] for r in fold_results]
                l2_distances = [r['l2_distance'] for r in fold_results]
                l0_distances = [r['l0_distance'] for r in fold_results]
                
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
                    'validity_delta': validity_delta
                }
                
                if perturb_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = bin_num if perturb_type == 'minor_deletion' else (0 if bin_num == 0 else 50)
                    data_description = f"Remove {remove_pct}%"
                else:
                    use_pct = 80 + bin_num if perturb_type == 'minor_addition' else (50 if bin_num == 0 else 100)
                    data_description = f"Use {use_pct}%"
                
                log_print(f"{perturb_type:<15} {bin_num:<5} {data_description:<12} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:<+11.4f}")

    # MODEL PERTURBATION SUMMARY
    log_print("\nMODEL PERTURBATION ROBUSTNESS SUMMARY (NICE):")
    log_print("-" * 150)
    log_print(f"{'Model Configuration':<30} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11}")
    log_print("-" * 150)
    
    model_summary_results = {}
    for model_key, fold_results in all_fold_results['model_perturbations'].items():
        if fold_results:
            validities = [r['validity'] for r in fold_results]
            accuracies = [r['model_accuracy'] for r in fold_results]
            l2_distances = [r['l2_distance'] for r in fold_results]
            l0_distances = [r['l0_distance'] for r in fold_results]
            
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
                'validity_delta': validity_delta
            }
            
            log_print(f"{model_key:<30} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:<+11.4f}")

    # 5. Final insights and completion
    log_print("\n" + "=" * 80)
    log_print("NICE COUNTERFACTUAL ROBUSTNESS ANALYSIS COMPLETED!")
    log_print("=" * 80)
    
    # End timing
    end_time = datetime.now()
    log_print(f"\n{'='*80}")
    log_print("🏁 COMPREHENSIVE NICE COUNTERFACTUAL ROBUSTNESS ANALYSIS COMPLETED!")
    log_print(f"{'='*80}")
    log_print(f"[TIME] Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"[LOG] Complete analysis saved to: {log_filename}")
    log_print(f"🧠 Used NICE algorithm for counterfactual generation")
    log_print(f"{'='*80}")

if __name__ == "__main__":
    main() 
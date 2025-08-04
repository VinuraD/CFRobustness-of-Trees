#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (CFXplorer v4) - HELOC Dataset

This script evaluates the robustness of counterfactual explanations using the CFXplorer algorithm across two separate experiments:
1. Data perturbations - testing how changes in training data affect counterfactual validity
2. Model perturbations - testing how different model types and hyperparameters affect counterfactual validity

The workflow is:
1. Generate counterfactual explanations using CFXplorer on unperturbed data with a baseline model
2. Run DATA PERTURBATION tests:
   - Train models with the same architecture on different perturbed datasets
   - Evaluate how valid the original counterfactuals remain
3. Run MODEL PERTURBATION tests:
   - Train different model types on the full unperturbed dataset
   - Evaluate how valid the original counterfactuals remain

This version uses the CFXplorer algorithm for counterfactual generation and the HELOC dataset (numerical features only).
Note: CFXplorer only works with RandomForestClassifier, so model perturbations are limited to RF hyperparameters.
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

# CFXplorer imports
from cfxplorer import Focus

from data_module import DataModule
from perturb import Perturbation

# Set up logging
def setup_logging():
    """Setup comprehensive logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cf_robustness_analysis_cfxplorer_v4_heloc_{timestamp}.log"
    
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

def generate_counterfactuals_cfxplorer(x_test, x_train, y_train, model):
    """
    Generate counterfactual explanations using CFXplorer algorithm
    Fixed version to handle tensor type compatibility issues
    """
    log_print(f"Generating counterfactuals using CFXplorer algorithm")
    log_print(f"Test set size: {len(x_test)} samples")
    
    try:
        # Convert to numpy arrays and ensure proper data types for CFXplorer
        if hasattr(x_test, 'values'):
            x_test_array = x_test.values.astype(np.float32)  # CFXplorer works better with float32
        else:
            x_test_array = np.array(x_test, dtype=np.float32)
            
        # Ensure training data is also float32
        if hasattr(x_train, 'values'):
            x_train_array = x_train.values.astype(np.float32)
        else:
            x_train_array = np.array(x_train, dtype=np.float32)
            
        # Ensure labels are proper integer type for CFXplorer compatibility
        # Platform-aware dtype selection for TensorFlow compatibility
        try:
            import tensorflow as tf
            
            # Check TensorFlow's default int type and platform behavior
            default_tf_int = tf.int64 if tf.executing_eagerly() else tf.int32
            log_print(f"TensorFlow default int type: {default_tf_int}")
            
            # Try int64 first (preferred on Ubuntu/Linux for TensorFlow), then fallback to int32
            dtype_priority = [np.int64, np.int32, np.int_, np.long] if sys.platform.startswith('linux') else [np.int32, np.int64, np.int_]
            
            y_train_array = None
            successful_dtype = None
            
            for dtype in dtype_priority:
                try:
                    if hasattr(y_train, 'values'):
                        y_train_array = y_train.values.astype(dtype)
                    else:
                        y_train_array = np.array(y_train, dtype=dtype)
                    successful_dtype = dtype
                    log_print(f"Successfully converted labels to {dtype.__name__} for platform compatibility")
                    break
                except Exception as dtype_error:
                    log_print(f"Failed to convert to {dtype.__name__}: {dtype_error}")
                    continue
            
            if y_train_array is None:
                # Final fallback - use original data type
                if hasattr(y_train, 'values'):
                    y_train_array = y_train.values
                else:
                    y_train_array = np.array(y_train)
                log_print(f"Using original label data type: {y_train_array.dtype}")
            else:
                log_print(f"Final label dtype: {y_train_array.dtype} ({successful_dtype.__name__})")
                
        except Exception as e:
            log_print(f"Error in platform-aware dtype conversion: {e}")
            # If all conversion attempts fail, try the original data
            if hasattr(y_train, 'values'):
                y_train_array = y_train.values
            else:
                y_train_array = np.array(y_train)
            log_print(f"Using original label data type: {y_train_array.dtype}")
            
        log_print(f"Test data shape: {x_test_array.shape}, dtype: {x_test_array.dtype}")
        log_print(f"Train data shape: {x_train_array.shape}, dtype: {x_train_array.dtype}")
        log_print(f"Train labels shape: {y_train_array.shape}, dtype: {y_train_array.dtype}")
        
        # Re-train the model with the properly typed data to ensure consistency
        log_print("Re-training model with properly typed data for CFXplorer...")
        model_copy = RandomForestClassifier(
            max_depth=model.max_depth, 
            n_estimators=model.n_estimators, 
            random_state=42
        )
        model_copy.fit(x_train_array, y_train_array)
        
        # Verify model predictions work with new data types
        test_pred = model_copy.predict(x_test_array[:1])
        log_print(f"Model prediction test - prediction type: {type(test_pred[0])}, value: {test_pred[0]}")
        
        # Initialize CFXplorer Focus instance (matching notebook parameters)
        focus = Focus(num_iter=100)
        
        # Generate counterfactuals with properly typed model and data
        cf_array = focus.generate(model_copy, x_test_array)
        
        log_print(f"CFXplorer generation completed. Result type: {type(cf_array)}")
        
        # Initialize result DataFrame
        if hasattr(x_test, 'columns'):
            cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        else:
            # Create column names if x_test is numpy array
            n_features = x_test_array.shape[1]
            columns = [f'feature_{i}' for i in range(n_features)] + ['cf_class', 'success']
            cf_list = pd.DataFrame(columns=columns)
            
        successful_cfs = 0
        failed_cfs = 0
        
        # Process results
        if cf_array is not None:
            log_print(f"CF array shape: {cf_array.shape}")
            
            for i in range(len(x_test_array)):
                try:
                    if i < len(cf_array):
                        # Get the counterfactual for this instance
                        cf_instance = cf_array[i]
                        
                        # Predict class for the counterfactual
                        cf_class = model.predict(cf_instance.reshape(1, -1))[0]
                        
                        # Store counterfactual
                        cf_row = list(cf_instance) + [cf_class, True]
                        cf_list.loc[i] = cf_row
                        successful_cfs += 1
                    else:
                        # No counterfactual for this instance
                        original_class = model.predict(x_test_array[i:i+1])[0]
                        default_row = list(x_test_array[i]) + [original_class, False]
                        cf_list.loc[i] = default_row
                        failed_cfs += 1
                        
                except Exception as e:
                    logger = logging.getLogger('CFRobustness')
                    logger.warning(f"Failed to process counterfactual for instance {i}: {str(e)}")
                    # Use original with success=False
                    original_class = model.predict(x_test_array[i:i+1])[0]
                    default_row = list(x_test_array[i]) + [original_class, False]
                    cf_list.loc[i] = default_row
                    failed_cfs += 1
        else:
            log_print("CFXplorer returned None - no counterfactuals generated")
            # Fill with originals and success=False
            for i in range(len(x_test_array)):
                original_class = model.predict(x_test_array[i:i+1])[0]
                default_row = list(x_test_array[i]) + [original_class, False]
                cf_list.loc[i] = default_row
                failed_cfs += 1
        
        success_rate = successful_cfs / len(x_test_array) if len(x_test_array) > 0 else 0
        log_print(f"CFXplorer counterfactual generation complete:")
        log_print(f"  Successful: {successful_cfs}/{len(x_test_array)} ({success_rate:.2%})")
        log_print(f"  Failed: {failed_cfs}/{len(x_test_array)} ({(1-success_rate):.2%})")
        
        return cf_list, success_rate
        
    except Exception as e:
        logger = logging.getLogger('CFRobustness')
        logger.error(f"Error in CFXplorer counterfactual generation: {str(e)}")
        logger.exception("Full traceback:")
        
        # Return empty DataFrame with proper structure
        if hasattr(x_test, 'columns'):
            cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        else:
            n_features = x_test.shape[1] if hasattr(x_test, 'shape') else len(x_test[0])
            columns = [f'feature_{i}' for i in range(n_features)] + ['cf_class', 'success']
            cf_list = pd.DataFrame(columns=columns)
            
        # Fill with original data and success=False
        x_test_array = x_test.values if hasattr(x_test, 'values') else np.array(x_test)
        for i in range(len(x_test_array)):
            try:
                original_class = model.predict(x_test_array[i:i+1])[0]
                default_row = list(x_test_array[i]) + [original_class, False]
                cf_list.loc[i] = default_row
            except:
                # If even prediction fails, use dummy data
                default_row = list(x_test_array[i]) + [0, False]
                cf_list.loc[i] = default_row
                
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
    l0_distances = np.sum((cf_features.values != corresponding_originals.values), axis=1)
    avg_l0_distance = np.mean(l0_distances)
    
    # 4. LOF Score (Local Outlier Factor)
    try:
        # Combine training data with counterfactuals for LOF calculation
        combined_data = np.vstack([x_train.values, cf_features.values])
        lof = LocalOutlierFactor(n_neighbors=100, contamination=0.1)
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

def run_data_perturbations(perturbation, X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx):
    """
    Run data perturbation experiments using EXISTING counterfactuals
    Tests how well the baseline counterfactuals perform on models trained with perturbed data
    """
    
    log_print(f"\nTesting data perturbations for fold {fold_idx}...")
    
    # Define perturbation types and ranges matching the DICE version
    data_perturbations = [
        ('minor_deletion', [0, 5, 10, 15, 20]),  # Bin 0 = baseline (0% removed)
        ('major_deletion', [0, 1]),              # Bin 0 = baseline (0% removed)
        ('minor_addition', [0, 5, 10, 15, 20]),  # Bin 0 ≠ baseline (uses 80% of data)
        ('major_addition', [0, 1])               # Bin 0 ≠ baseline (uses 50% of data)
    ]
    
    results = {}
    
    for pert_type, bins in data_perturbations:
        log_print(f"  {pert_type}:")
        pert_results = []
        
        for bin_val in bins:
            try:
                # Get the raw training data for this fold
                train_raw, _ = perturbation.get_data(fold=fold_idx, raw_data=True)
                
                # Apply perturbation using the perturb_data method
                perturbed_train_data = perturbation.perturb_data(train_raw, pert_type, bin_val)
                
                # Apply the same preprocessing as baseline model (critical!)
                perturbed_processed = perturbation.data_module._preprocess_data(perturbed_train_data)
                
                # Extract features and labels
                label_col = perturbation.get_metadata()['label_column']
                X_pert = perturbed_processed.drop(columns=[label_col])
                y_pert = perturbed_processed[label_col]
                
                # Handle categorical labels if needed
                if y_pert.dtype == 'object':
                    le = LabelEncoder()
                    y_pert = le.fit_transform(y_pert)
                
                log_print(f"    Bin {bin_val} - Train: {len(X_pert)} samples, Test: {len(X_test)} samples")
                
                # Train model on perturbed data
                model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
                model.fit(X_pert, y_pert)
                
                # Test EXISTING baseline counterfactuals on this perturbed model
                metrics = calculate_comprehensive_metrics(model, baseline_cf_list, X_test, X_pert)
                
                # Calculate model accuracy
                accuracy = accuracy_score(y_test, model.predict(X_test))
                
                pert_results.append({
                    'bin': bin_val,
                    'accuracy': accuracy,
                    'model_accuracy': accuracy,
                    **metrics
                })
                
                if pert_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = bin_val if pert_type == 'minor_deletion' else (0 if bin_val == 0 else 50)
                    log_print(f"    Bin {bin_val}: Remove {remove_pct}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
                else:
                    use_pct = 80 + bin_val if pert_type == 'minor_addition' else (50 if bin_val == 0 else 100)
                    log_print(f"    Bin {bin_val}: Use {use_pct}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
                
            except Exception as e:
                log_print(f"      Error in {pert_type} bin {bin_val}: {e}")
                # Add default entry for failed perturbation
                pert_results.append({
                    'bin': bin_val,
                    'accuracy': 0.0,
                    'model_accuracy': 0.0,
                    'validity': 0.0,
                    'flipped': 0,
                    'total': 0,
                    'l2_distance': 0.0,
                    'l0_distance': 0.0,
                    'lof_score': 0.0
                })
        
        results[pert_type] = pert_results
    
    return results

def run_model_perturbations(X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx):
    """
    Run model perturbation experiments using EXISTING counterfactuals
    Tests how well the baseline counterfactuals perform on different model configurations
    """
    
    log_print(f"\nTesting model perturbations for fold {fold_idx}...")
    
    # Define RandomForest hyperparameters to test (CFXplorer only works with RF)
    # Matching the DICE version structure
    model_configs = []
    
    # Max depth study: Fix n_estimators=100, vary max_depth=[3,4,5,6]
    for max_depth in [3, 4, 5, 6]:
        model_configs.append(('random_forest', max_depth, 100))
    
    # N_estimators study: Fix max_depth=5, vary n_estimators=[50,100,150,200]
    for n_estimators in [50, 100, 150, 200]:
        model_configs.append(('random_forest', 5, n_estimators))
    
    results = []
    
    for model_type, max_depth, n_estimators in model_configs:
        try:
            log_print(f"    {model_type} ({max_depth}, {n_estimators}) - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            # Train perturbed model
            model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
            model.fit(X_train, y_train)
            
            # Test EXISTING baseline counterfactuals on this model
            metrics = calculate_comprehensive_metrics(model, baseline_cf_list, X_test, X_train)
            
            # Calculate model accuracy
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            results.append({
                'model_type': model_type,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'accuracy': accuracy,
                'model_accuracy': accuracy,
                **metrics
            })
            
            log_print(f"      {model_type} ({max_depth}, {n_estimators}): validity {metrics['validity']:.4f}, accuracy {accuracy:.4f}")
            
        except Exception as e:
            log_print(f"      Error with {model_type} ({max_depth}, {n_estimators}): {e}")
            # Add default entry for failed model
            results.append({
                'model_type': model_type,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'accuracy': 0.0,
                'model_accuracy': 0.0,
                'validity': 0.0,
                'flipped': 0,
                'total': 0,
                'l2_distance': 0.0,
                'l0_distance': 0.0,
                'lof_score': 0.0
            })
    
    return results

def create_comprehensive_visualizations(all_results, output_dir="cfxplorer_plots"):
    """Create comprehensive visualizations for all results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Data Robustness Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CFXplorer Counterfactual Data Robustness Analysis - HELOC Dataset', fontsize=16, fontweight='bold')
    
    perturbation_types = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
    
    for idx, pert_type in enumerate(perturbation_types):
        ax = axes[idx // 2, idx % 2]
        
        # Aggregate results across folds
        fold_data = []
        for fold_idx, fold_results in enumerate(all_results):
            if pert_type in fold_results['data_perturbations']:
                for result in fold_results['data_perturbations'][pert_type]:
                    fold_data.append({
                        'fold': fold_idx,
                        'bin': result['bin'],
                        'validity': result['validity'],
                        'accuracy': result['accuracy']
                    })
        
        if fold_data:
            df = pd.DataFrame(fold_data)
            
            # Group by bin and calculate mean and std
            grouped = df.groupby('bin').agg({
                'validity': ['mean', 'std'],
                'accuracy': ['mean', 'std']
            }).fillna(0)
            
            bins = grouped.index
            validity_mean = grouped['validity']['mean']
            validity_std = grouped['validity']['std']
            accuracy_mean = grouped['accuracy']['mean']
            accuracy_std = grouped['accuracy']['std']
            
            # Plot with error bars
            ax.errorbar(bins, validity_mean, yerr=validity_std, 
                       label='Validity', marker='o', capsize=5, capthick=2)
            ax.errorbar(bins, accuracy_mean, yerr=accuracy_std, 
                       label='Accuracy', marker='s', capsize=5, capthick=2)
            
            ax.set_title(f'{pert_type.replace("_", " ").title()}')
            ax.set_xlabel('Perturbation Level')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'{output_dir}/cf_data_robustness_heloc_5fold_plot_{timestamp}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    log_print(f"Data robustness visualization saved to {output_dir}/")

def print_statistical_summary(all_results):
    """Print comprehensive statistical summary of all experiments"""
    
    log_print("\n" + "="*80)
    log_print("COMPREHENSIVE STATISTICAL SUMMARY")
    log_print("="*80)
    
    num_folds = len(all_results)
    log_print(f"Analysis across {num_folds} cross-validation folds")
    
    # Baseline performance summary
    log_print(f"\n[SUMMARY] BASELINE PERFORMANCE SUMMARY:")
    baseline_validities = [result['baseline']['validity'] for result in all_results]
    baseline_accuracies = [result['baseline']['accuracy'] for result in all_results]
    
    log_print(f"  Baseline CF Validity: {np.mean(baseline_validities):.4f} ± {np.std(baseline_validities):.4f}")
    log_print(f"  Baseline Model Accuracy: {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
    
    # Data perturbation summary
    log_print(f"\n[INSIGHTS] DATA PERTURBATION ROBUSTNESS:")
    perturbation_types = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
    
    for pert_type in perturbation_types:
        log_print(f"  {pert_type.replace('_', ' ').title()}:")
        
        all_validities = []
        for result in all_results:
            if pert_type in result['data_perturbations']:
                validities = [r['validity'] for r in result['data_perturbations'][pert_type]]
                all_validities.extend(validities)
        
        if all_validities:
            log_print(f"    Mean validity: {np.mean(all_validities):.4f} ± {np.std(all_validities):.4f}")
            log_print(f"    Min validity: {np.min(all_validities):.4f}")
            log_print(f"    Max validity: {np.max(all_validities):.4f}")
    
    # Model perturbation summary
    log_print(f"\n🤖 MODEL PERTURBATION ROBUSTNESS:")
    all_model_validities = []
    for result in all_results:
        validities = [r['validity'] for r in result['model_perturbations']]
        all_model_validities.extend(validities)
    
    if all_model_validities:
        log_print(f"  Overall model robustness:")
        log_print(f"    Mean validity: {np.mean(all_model_validities):.4f} ± {np.std(all_model_validities):.4f}")
        log_print(f"    Min validity: {np.min(all_model_validities):.4f}")
        log_print(f"    Max validity: {np.max(all_model_validities):.4f}")
    
    log_print(f"\n💡 KEY INSIGHTS FOR HELOC DATASET:")
    log_print(f"  • CFXplorer processes pure numerical features efficiently")
    log_print(f"  • HELOC financial data provides challenging CF generation scenarios")  
    log_print(f"  • Numerical features allow for continuous perturbations")
    log_print(f"  • Focus algorithm adapts well to high-dimensional numerical data")

def main():
    """Main execution function"""
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("="*80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (CFXPLORER v4) - HELOC DATASET")
    log_print("="*80)
    log_print(f"[LOG] Logging session to: {log_filename}")
    log_print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*80)
    
    try:
        # Initialize data module
        log_print("\n1. Loading dataset...")
        # Adjust path to be relative to the parent directory
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "HELOC.csv")
        dm = DataModule(data_path, n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Dataset: HELOC")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
        # Setup cross-validation
        log_print("\n2. Analysis parameters:")
        n_folds = 5
        log_print(f"  Number of folds: {n_folds}")
        log_print(f"  Counterfactual method: CFXplorer")
        
        # Print fold summary
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            dm.print_fold_summary()
        
        # Log each line of the captured output
        summary_output = f.getvalue()
        for line in summary_output.strip().split('\n'):
            if line.strip():
                log_print(line)
        
        # Initialize results storage
        all_results = []
        
        log_print("\n3. Running comprehensive analysis across all 5 folds...")
        log_print("="*80)
        
        # Run analysis for each fold
        for fold_idx in range(n_folds):
            log_print(f"\n--- FOLD {fold_idx} ANALYSIS ---")
            log_print("-" * 50)
            
            # Get fold data
            train_raw, _ = perturbation.get_data(fold=fold_idx, raw_data=True)
            train_processed, test_processed = perturbation.get_data(fold=fold_idx, raw_data=False)
            
            log_print(f"Fold {fold_idx} data shapes:")
            log_print(f"  Raw train: {train_raw.shape}")
            log_print(f"  Processed - Train: {train_processed.shape}, Test: {test_processed.shape}")
            
            # Setup for CFXplorer
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
            
            # Train baseline model
            log_print(f"\nTraining baseline model for fold {fold_idx}...")
            baseline_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
            baseline_model.fit(X_train, y_train)
            
            train_accuracy = accuracy_score(y_train, baseline_model.predict(X_train))
            test_accuracy = accuracy_score(y_test, baseline_model.predict(X_test))
            log_print(f"  Train accuracy: {train_accuracy:.4f}")
            log_print(f"  Test accuracy: {test_accuracy:.4f}")
            
            # Generate baseline counterfactuals ONCE per fold
            log_print(f"Generating counterfactuals using CFXplorer for fold {fold_idx}...")
            baseline_cf_list, baseline_success_rate = generate_counterfactuals_cfxplorer(
                X_test, X_train, y_train, baseline_model
            )
            
            # Calculate baseline metrics
            baseline_metrics = calculate_comprehensive_metrics(
                baseline_model, baseline_cf_list, X_test, X_train
            )
            
            log_print(f"  Success rate: {baseline_success_rate:.2%}")
            log_print(f"  Baseline validity: {baseline_metrics['validity']:.4f} ({baseline_metrics['flipped']}/{baseline_metrics['total']})")
            log_print(f"  Baseline L2 distance: {baseline_metrics['l2_distance']:.4f}")
            log_print(f"  Baseline L0 distance: {baseline_metrics['l0_distance']:.2f}")
            log_print(f"  Baseline LOF score: {baseline_metrics['lof_score']:.4f}")
            
            # Test the SAME counterfactuals on perturbed models
            data_pert_results = run_data_perturbations(
                perturbation, X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx
            )
            
            model_pert_results = run_model_perturbations(
                X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx
            )
            
            # Store results for this fold
            fold_results = {
                'fold': fold_idx,
                'baseline': {
                    'accuracy': test_accuracy,
                    'success_rate': baseline_success_rate,
                    **baseline_metrics
                },
                'data_perturbations': data_pert_results,
                'model_perturbations': model_pert_results
            }
            
            all_results.append(fold_results)
        
        # Create visualizations
        log_print(f"\n4. Creating comprehensive visualizations...")
        create_comprehensive_visualizations(all_results)
        
        # Print statistical summary
        print_statistical_summary(all_results)
        
        log_print(f"\n[SUCCESS] CFXplorer robustness analysis completed successfully!")
        log_print(f"📁 All results saved and logged to: {log_filename}")
        
    except Exception as e:
        log_print(f"\n[ERROR] Error during analysis: {str(e)}")
        logger.exception("Full traceback:")
        raise
    
    finally:
        # Close logging handlers
        logger = logging.getLogger('CFRobustness')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

if __name__ == "__main__":
    main()

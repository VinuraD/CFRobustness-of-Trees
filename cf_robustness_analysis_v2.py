#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (v2)

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

This helps quantify the independent effects of data and model choices on counterfactual explanation stability.
"""

import sys
import os
import logging
from datetime import datetime
import contextlib
import io
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

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
    log_filename = f"cf_robustness_analysis_v2_{timestamp}.log"
    
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

def calculate_validity(model, cf_list, x_test):
    """
    Calculate validity: how many CFs actually flip the predicted class with the given model
    
    Args:
        model: Model to validate counterfactuals against
        cf_list: DataFrame with counterfactuals
        x_test: Original test data
        
    Returns:
        validity: Proportion of counterfactuals that flip the class
    """
    # Get only successful counterfactuals
    successful_mask = cf_list['success'] == True
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals to validate")
        return 0.0, 0, 0
    
    successful_cfs = cf_list[successful_mask]
    corresponding_originals = x_test[successful_mask]
    
    # Remove the success column for prediction
    cf_features = successful_cfs.drop(['success', 'cf_class'], axis=1)
    
    # Predict on counterfactuals and originals
    cf_predictions = model.predict(cf_features)
    original_predictions = model.predict(corresponding_originals)
    
    # Calculate how many actually flipped
    flipped = (cf_predictions != original_predictions).sum()
    validity = flipped / len(successful_cfs) if len(successful_cfs) > 0 else 0.0
    
    return validity, flipped, len(successful_cfs)

def main():
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (v2)")
    log_print("=" * 80)
    log_print(f"📝 Logging session to: {log_filename}")
    log_print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    # 1. Load dataset
    log_print("\n1. Loading dataset...")
    try:
        dm = DataModule("data/Spambase.csv", n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Dataset: Spambase")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return
    
    # 2. Set analysis parameters
    fold = 0
    log_print("\n2. Analysis parameters:")
    log_print(f"  Fold: {fold}")
    
    # Define data perturbations to test
    data_perturbations = [
        ('minor_deletion', [0, 5, 10, 15, 20]),
        ('major_deletion', [0, 1]),
        ('minor_addition', [0, 5, 10, 15, 20]),
        ('major_addition', [0, 1])
    ]
    log_print(f"  Data perturbation types: {[p[0] for p in data_perturbations]}")
    
    # Define model perturbations to test
    model_perturbations = [
        # Model type, max_depth, n_estimators
        ('random_forest', 3, 50),
        ('random_forest', 5, 100),
        ('random_forest', 10, 200),
        ('xgboost', 3, 50),
        ('xgboost', 6, 100),
        ('lightgbm', 4, 75),
        ('adaboost', 2, 30)
    ]
    log_print(f"  Model perturbations: {len(model_perturbations)} configurations")
    for model_type, max_depth, n_estimators in model_perturbations:
        log_print(f"    - {model_type} (max_depth={max_depth}, n_estimators={n_estimators})")
    
    # Dictionary to store results for each perturbation
    results = {
        'data_perturbations': {},
        'model_perturbations': {}
    }
    
    # 3. Get unperturbed data (Bin 0)
    log_print("\n3. Preparing unperturbed data (fold 0)...")
    try:
        train_raw, _ = perturbation.get_data(fold=fold, raw_data=True)
        train_processed, test_processed = perturbation.get_data(fold=fold, raw_data=False)
        
        log_print(f"Unperturbed data shapes:")
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
        
        log_print(f"Data preparation complete:")
        log_print(f"  Training samples: {len(X_train)}")
        log_print(f"  Test samples: {len(X_test)}")
        log_print(f"  Features: {len(X_train.columns)}")
        log_print(f"  Class distribution - Train: {np.bincount(y_train)}")
        log_print(f"  Class distribution - Test: {np.bincount(y_test)}")
        
    except Exception as e:
        log_print(f"Error preparing unperturbed data: {e}")
        return
    
    # 4. Train baseline model on unperturbed data
    log_print("\n4. Training baseline model on unperturbed data...")
    try:
        baseline_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        baseline_model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
        test_acc = accuracy_score(y_test, baseline_model.predict(X_test))
        
        log_print(f"Baseline model (RandomForest, max_depth=5, n_estimators=100):")
        log_print(f"  Train accuracy: {train_acc:.4f}")
        log_print(f"  Test accuracy: {test_acc:.4f}")
        
    except Exception as e:
        log_print(f"Error training baseline model: {e}")
        return
    
    # 5. Setup DiCE data object
    log_print("\n5. Setting up DiCE framework...")
    try:
        # Create a combined dataset for DiCE
        train_data_with_label = train_processed.copy()
        
        # Determine continuous and categorical features for DiCE
        # For Spambase, most features are likely continuous
        continuous_features = X_train.columns.tolist()  # Assume all are continuous for now
        categorical_features = []  # Update if there are categorical features
        
        # Create DiCE data object
        dice_data = dice_ml.Data(
            dataframe=train_data_with_label,
            continuous_features=continuous_features,
            outcome_name=label_col
        )
        
        log_print(f"DiCE setup complete:")
        log_print(f"  Continuous features: {len(continuous_features)}")
        log_print(f"  Categorical features: {len(categorical_features)}")
        
    except Exception as e:
        log_print(f"Error setting up DiCE: {e}")
        return
    
    # 6. Generate counterfactuals using baseline model on unperturbed data
    log_print("\n6. Generating counterfactuals using baseline model...")
    try:
        # Create test data for counterfactual generation (without label)
        test_data_for_cf = X_test.copy()
        
        cf_list, success_rate = generate_counterfactuals(
            test_data_for_cf, 
            baseline_model, 
            dice_data, 
            method='random',
            total_cfs=2
        )
        
        log_print(f"Counterfactual generation summary:")
        log_print(f"  Total test samples: {len(X_test)}")
        log_print(f"  Successful generations: {(cf_list['success'] == True).sum()}")
        log_print(f"  Success rate: {success_rate:.2%}")
        
        # Validate the counterfactuals on the baseline model
        baseline_validity, flipped, total = calculate_validity(baseline_model, cf_list, X_test)
        log_print(f"Baseline validity: {baseline_validity:.4f} ({flipped}/{total})")
        
    except Exception as e:
        log_print(f"Error generating counterfactuals: {e}")
        return
    
    # 7. Test counterfactuals on data perturbed models
    log_print("\n7. Testing counterfactual robustness across data perturbations...")
    log_print("=" * 80)
    
    # For storing result summaries
    data_validity_results = {}
    
    # Function to evaluate one data perturbation type
    def evaluate_data_perturbation_type(perturb_type, bins):
        log_print(f"\n📊 {perturb_type.upper()} PERTURBATION ANALYSIS")
        log_print("-" * 60)
        
        results = []
        
        for bin_num in bins:
            try:
                # Get the raw unperturbed training data
                train_raw, _ = perturbation.get_data(fold=fold, raw_data=True)
                
                # Apply perturbation to the raw training data
                perturbed_train_raw = perturbation.perturb_data(train_raw, perturb_type, bin_num)
                
                # Process the perturbed data
                if perturb_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = bin_num if perturb_type == 'minor_deletion' else (0 if bin_num == 0 else 50)
                    log_print(f"Bin {bin_num}: Remove {remove_pct}% -> {perturbed_train_raw.shape[0]} samples")
                else:  # 'minor_addition', 'major_addition'
                    use_pct = 80 + bin_num if perturb_type == 'minor_addition' else (50 if bin_num == 0 else 100)
                    log_print(f"Bin {bin_num}: Use {use_pct}% -> {perturbed_train_raw.shape[0]} samples")
                
                # Prepare features and target
                label_col = metadata['label_column']
                perturbed_X_train = perturbed_train_raw.drop(columns=[label_col])
                perturbed_y_train = perturbed_train_raw[label_col]
                
                # Handle categorical labels if needed
                if perturbed_y_train.dtype == 'object':
                    le = LabelEncoder()
                    perturbed_y_train = le.fit_transform(perturbed_y_train)
                
                # Train model on perturbed data (using same params as baseline)
                perturbed_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                perturbed_model.fit(perturbed_X_train, perturbed_y_train)
                
                # Evaluate perturbed model on test set
                perturbed_test_acc = accuracy_score(y_test, perturbed_model.predict(X_test))
                
                # Calculate validity of original counterfactuals on perturbed model
                cf_validity, cf_flipped, cf_total = calculate_validity(perturbed_model, cf_list, X_test)
                
                # Store results
                result = {
                    'perturb_type': perturb_type,
                    'bin': bin_num,
                    'validity': cf_validity,
                    'flipped': cf_flipped,
                    'total': cf_total,
                    'accuracy': perturbed_test_acc,
                    'train_samples': perturbed_train_raw.shape[0]
                }
                
                results.append(result)
                
                log_print(f"  Model accuracy: {perturbed_test_acc:.4f}")
                log_print(f"  Counterfactual validity: {cf_validity:.4f} ({cf_flipped}/{cf_total})")
                log_print(f"  Validity change from baseline: {cf_validity - baseline_validity:+.4f}")
                log_print("-" * 40)
                
            except Exception as e:
                log_print(f"Error in {perturb_type} bin {bin_num}: {e}")
        
        return results
    
    # Test all data perturbation types
    for perturb_type, bins in data_perturbations:
        results = evaluate_data_perturbation_type(perturb_type, bins)
        data_validity_results[perturb_type] = results
    
    # 8. Test counterfactuals on model perturbations
    log_print("\n8. Testing counterfactual robustness across model perturbations...")
    log_print("=" * 80)    
    model_validity_results = []
    log_print("\n🤖 MODEL PERTURBATION ANALYSIS")
    log_print("-" * 60)
    
    for model_type, max_depth, n_estimators in model_perturbations:
        try:
            log_print(f"Model: {model_type} (max_depth={max_depth}, n_estimators={n_estimators})")
            
            # Get the raw unperturbed training data (full dataset)
            train_raw, _ = perturbation.get_data(fold=fold, raw_data=True)
            
            # Create and train the model on full unperturbed data
            perturbed_model = perturbation.perturb_model(
                train_raw, 
                model_type=model_type, 
                max_depth=max_depth, 
                n_estimators=n_estimators
            )
            
            # Evaluate model on test set
            model_test_acc = accuracy_score(y_test, perturbed_model.predict(X_test))
            
            # Calculate validity of original counterfactuals on this model
            cf_validity, cf_flipped, cf_total = calculate_validity(perturbed_model, cf_list, X_test)
            
            # Store results
            result = {
                'model_type': model_type,
                'max_depth': max_depth,
                'n_estimators': n_estimators,
                'validity': cf_validity,
                'flipped': cf_flipped,
                'total': cf_total,
                'accuracy': model_test_acc
            }
            
            model_validity_results.append(result)
            
            log_print(f"  Model accuracy: {model_test_acc:.4f}")
            log_print(f"  Counterfactual validity: {cf_validity:.4f} ({cf_flipped}/{cf_total})")
            log_print(f"  Validity change from baseline: {cf_validity - baseline_validity:+.4f}")
            log_print("-" * 40)
            
        except Exception as e:
            log_print(f"Error with model {model_type} (max_depth={max_depth}, n_estimators={n_estimators}): {e}")    # 9. Generate summary and visualization
    log_print("\n9. Generating summary of counterfactual robustness...")
    log_print("=" * 80)
    
    try:
        # Summary table for data perturbations
        log_print("\nDATA PERTURBATION ROBUSTNESS SUMMARY:")
        log_print("-" * 100)
        log_print(f"{'Perturbation':<15} {'Bin':<5} {'Data':<12} {'Train Samples':<15} {'Model Acc':<12} {'CF Validity':<12} {'Validity Δ':<12} {'Flipped/Total':<15}")
        log_print("-" * 100)
        
        for perturb_type, results in data_validity_results.items():
            for result in results:
                if perturb_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = result['bin'] if perturb_type == 'minor_deletion' else (0 if result['bin'] == 0 else 50)
                    data_description = f"Remove {remove_pct}%"
                else:  # 'minor_addition', 'major_addition'
                    use_pct = 80 + result['bin'] if perturb_type == 'minor_addition' else (50 if result['bin'] == 0 else 100)
                    data_description = f"Use {use_pct}%"
                
                log_print(f"{perturb_type:<15} {result['bin']:<5} {data_description:<12} {result['train_samples']:<15} {result['accuracy']:.4f}       {result['validity']:.4f}       {result['validity'] - baseline_validity:+.4f}       {result['flipped']}/{result['total']}")
        
        # Summary table for model perturbations
        log_print("\nMODEL PERTURBATION ROBUSTNESS SUMMARY:")
        log_print("-" * 100)
        log_print(f"{'Model Type':<15} {'Max Depth':<10} {'N Estimators':<15} {'Model Acc':<12} {'CF Validity':<12} {'Validity Δ':<12} {'Flipped/Total':<15}")
        log_print("-" * 100)
        
        for result in model_validity_results:
            log_print(f"{result['model_type']:<15} {result['max_depth']:<10} {result['n_estimators']:<15} {result['accuracy']:.4f}       {result['validity']:.4f}       {result['validity'] - baseline_validity:+.4f}       {result['flipped']}/{result['total']}")
        
        # Plot data perturbation results
        plt.figure(figsize=(12, 8))
        
        # Create a different marker for each perturbation type
        markers = {
            'minor_deletion': 'o',
            'major_deletion': 's',
            'minor_addition': '^',
            'major_addition': 'D'
        }
        
        for perturb_type, results in data_validity_results.items():
            if not results:
                continue
                
            # Extract data for plotting
            if perturb_type in ['minor_deletion', 'major_deletion']:
                x_labels = [f"{r['bin']}% removed" if perturb_type == 'minor_deletion' else ('0%' if r['bin'] == 0 else '50%') for r in results]
            else:
                x_labels = [f"{80+r['bin']}% used" if perturb_type == 'minor_addition' else ('50%' if r['bin'] == 0 else '100%') for r in results]
                
            validities = [r['validity'] for r in results]
            
            plt.plot(x_labels, validities, marker=markers[perturb_type], label=perturb_type, linewidth=2, markersize=8)
        
        # Add baseline as horizontal line
        plt.axhline(y=baseline_validity, color='red', linestyle='--', label='Baseline Validity')
        
        plt.title('Counterfactual Explanation Robustness Across Data Perturbations', fontsize=14)
        plt.xlabel('Perturbation Level', fontsize=12)
        plt.ylabel('Counterfactual Validity', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        
        # Save data perturbation plot
        plt.tight_layout()
        data_plot_filename = f"cf_data_robustness_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(data_plot_filename)
        log_print(f"\nData perturbation plot saved to: {data_plot_filename}")
        
        # Plot model perturbation results
        plt.figure(figsize=(12, 8))
        
        # Group by model type
        model_types = set(r['model_type'] for r in model_validity_results)
        
        for model_type in model_types:
            model_results = [r for r in model_validity_results if r['model_type'] == model_type]
            if not model_results:
                continue
                
            # Sort by n_estimators for better visualization
            model_results.sort(key=lambda x: x['n_estimators'])
            
            # Create x labels
            x_labels = [f"d={r['max_depth']}, n={r['n_estimators']}" for r in model_results]
            validities = [r['validity'] for r in model_results]
            
            plt.plot(x_labels, validities, marker='o', label=model_type, linewidth=2, markersize=8)
        
        # Add baseline as horizontal line
        plt.axhline(y=baseline_validity, color='red', linestyle='--', label='Baseline Validity')
        
        plt.title('Counterfactual Explanation Robustness Across Model Types', fontsize=14)
        plt.xlabel('Model Configuration', fontsize=12)
        plt.ylabel('Counterfactual Validity', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        plt.xticks(rotation=45)
          # Save model perturbation plot
        plt.tight_layout()
        model_plot_filename = f"cf_model_robustness_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(model_plot_filename)
        log_print(f"Model perturbation plot saved to: {model_plot_filename}")
        
    except Exception as e:
        log_print(f"Error generating summary visualizations: {e}")
    
    # 10. Final insights
    log_print("\n" + "=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS INSIGHTS")
    log_print("=" * 80)
    
    # Calculate average validity change for each perturbation type
    data_avg_changes = {}
    for perturb_type, results in data_validity_results.items():
        if results:  # Skip empty results
            # Skip bin 0 which is unperturbed for deletion types
            perturbed_results = [r for r in results if not (perturb_type in ['minor_deletion', 'major_deletion'] and r['bin'] == 0)]
            if perturbed_results:
                avg_change = np.mean([r['validity'] - baseline_validity for r in perturbed_results])
                data_avg_changes[perturb_type] = avg_change
    
    # Calculate average validity change for each model type
    model_avg_changes = {}
    model_types = set(r['model_type'] for r in model_validity_results)
    for model_type in model_types:
        type_results = [r for r in model_validity_results if r['model_type'] == model_type]
        if type_results:
            avg_change = np.mean([r['validity'] - baseline_validity for r in type_results])
            model_avg_changes[model_type] = avg_change
    
    # Report findings
    log_print("\n📊 DATA PERTURBATION INSIGHTS:")
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
      # End timing
    end_time = datetime.now()
    log_print(f"\n{'='*80}")
    log_print("🏁 COMPREHENSIVE COUNTERFACTUAL ROBUSTNESS ANALYSIS COMPLETED!")
    log_print(f"{'='*80}")
    log_print(f"🕒 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Complete analysis saved to: {log_filename}")
    log_print(f"{'='*80}")
    
    # Summary stats
    total_experiments = (
        sum(len(bins) for _, bins in data_perturbations) +  # Data perturbations
        len(model_perturbations)                           # Model perturbations
    )
    log_print(f"Total experiments run: {total_experiments}")
    log_print(f"Total visualizations created: 2")

if __name__ == "__main__":
    main()

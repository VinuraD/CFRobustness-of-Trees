#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis

This script evaluates the robustness of counterfactual explanations across perturbed datasets.
The workflow is:
1. Generate counterfactual explanations on unperturbed data
2. Train models on perturbed versions of the dataset
3. Evaluate how valid the original counterfactuals remain when tested against models trained on perturbed data

This helps quantify the stability of counterfactual explanations under data perturbations.
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
    log_filename = f"cf_robustness_analysis_{timestamp}.log"
    
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
        return 0.0
    
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
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS")
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
    log_print(f"  Perturbation types: minor_deletion, major_deletion, minor_addition, major_addition")
    
    # Dictionary to store results for each perturbation
    results = {
        'minor_deletion': {},
        'major_deletion': {},
        'minor_addition': {},
        'major_addition': {}
    }
    
    # 3. Get unperturbed data (Bin 0)
    log_print("\n3. Preparing unperturbed data (fold 0)...")
    try:
        train_processed, test_processed = perturbation.get_data(fold=fold, raw_data=False)
        
        log_print(f"Unperturbed data shapes:")
        log_print(f"  Train: {train_processed.shape}, Test: {test_processed.shape}")
        
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
        baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        baseline_model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
        test_acc = accuracy_score(y_test, baseline_model.predict(X_test))
        
        log_print(f"Baseline model performance:")
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
    
    # 7. Test counterfactuals on perturbed data models
    log_print("\n7. Testing counterfactual robustness across perturbed datasets...")
    log_print("=" * 80)
    
    # For storing result summaries
    validity_results = {
        'minor_deletion': [],
        'major_deletion': [],
        'minor_addition': [],
        'major_addition': []
    }
    
    # Function to evaluate one perturbation type
    def evaluate_perturbation_type(perturb_type, bins):
        log_print(f"\n📊 {perturb_type.upper()} PERTURBATION ANALYSIS")
        log_print("-" * 60)
        
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
                
                # Need to preprocess this perturbed data
                # We'll use the DataModule for this, but we need to replace its data
                dm_perturbed = DataModule("data/Spambase.csv", n_splits=5, random_state=42)
                
                # Train a model on the perturbed data
                label_col = metadata['label_column']
                perturbed_X_train = perturbed_train_raw.drop(columns=[label_col])
                perturbed_y_train = perturbed_train_raw[label_col]
                
                # Handle categorical labels if needed
                if perturbed_y_train.dtype == 'object':
                    le = LabelEncoder()
                    perturbed_y_train = le.fit_transform(perturbed_y_train)
                
                # Train model on perturbed data
                perturbed_model = RandomForestClassifier(n_estimators=100, random_state=42)
                perturbed_model.fit(perturbed_X_train, perturbed_y_train)
                
                # Evaluate perturbed model on test set
                perturbed_test_acc = accuracy_score(y_test, perturbed_model.predict(X_test))
                
                # Calculate validity of original counterfactuals on perturbed model
                cf_validity, cf_flipped, cf_total = calculate_validity(perturbed_model, cf_list, X_test)
                
                # Store results
                validity_results[perturb_type].append({
                    'bin': bin_num,
                    'validity': cf_validity,
                    'flipped': cf_flipped,
                    'total': cf_total,
                    'accuracy': perturbed_test_acc,
                    'train_samples': perturbed_train_raw.shape[0]
                })
                
                log_print(f"  Model accuracy: {perturbed_test_acc:.4f}")
                log_print(f"  Counterfactual validity: {cf_validity:.4f} ({cf_flipped}/{cf_total})")
                log_print(f"  Validity change from baseline: {cf_validity - baseline_validity:+.4f}")
                log_print("-" * 40)
                
            except Exception as e:
                log_print(f"Error in {perturb_type} bin {bin_num}: {e}")
    
    # Test all perturbation types
    # Minor deletion: bins 0, 5, 10, 15, 20
    evaluate_perturbation_type('minor_deletion', [0, 5, 10, 15, 20])
    
    # Major deletion: bins 0, 1
    evaluate_perturbation_type('major_deletion', [0, 1])
    
    # Minor addition: bins 0, 5, 10, 15, 20
    evaluate_perturbation_type('minor_addition', [0, 5, 10, 15, 20])
    
    # Major addition: bins 0, 1
    evaluate_perturbation_type('major_addition', [0, 1])
    
    # 8. Generate summary and visualization
    log_print("\n8. Generating summary of counterfactual robustness...")
    log_print("=" * 80)
    
    try:
        # Create summary table
        summary_rows = []
        
        for perturb_type, results in validity_results.items():
            for result in results:
                if perturb_type in ['minor_deletion', 'major_deletion']:
                    remove_pct = result['bin'] if perturb_type == 'minor_deletion' else (0 if result['bin'] == 0 else 50)
                    data_description = f"Remove {remove_pct}%"
                else:  # 'minor_addition', 'major_addition'
                    use_pct = 80 + result['bin'] if perturb_type == 'minor_addition' else (50 if result['bin'] == 0 else 100)
                    data_description = f"Use {use_pct}%"
                
                summary_rows.append({
                    'Perturbation': perturb_type,
                    'Bin': result['bin'],
                    'Data': data_description,
                    'Train Samples': result['train_samples'],
                    'Model Accuracy': result['accuracy'],
                    'CF Validity': result['validity'],
                    'Validity Change': result['validity'] - baseline_validity,
                    'Flipped/Total': f"{result['flipped']}/{result['total']}"
                })
        
        # Create DataFrame for easy display
        summary_df = pd.DataFrame(summary_rows)
        
        # Display summary table
        log_print("\nCOUNTERFACTUAL ROBUSTNESS SUMMARY:")
        log_print("-" * 100)
        
        # Format and print the table rows
        log_print(f"{'Perturbation':<15} {'Bin':<5} {'Data':<12} {'Train Samples':<15} {'Model Acc':<12} {'CF Validity':<12} {'Validity Δ':<12} {'Flipped/Total':<15}")
        log_print("-" * 100)
        
        for _, row in summary_df.iterrows():
            log_print(f"{row['Perturbation']:<15} {row['Bin']:<5} {row['Data']:<12} {row['Train Samples']:<15} {row['Model Accuracy']:.4f}       {row['CF Validity']:.4f}       {row['Validity Change']:+.4f}       {row['Flipped/Total']:<15}")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        
        # Create a different marker for each perturbation type
        markers = {
            'minor_deletion': 'o',
            'major_deletion': 's',
            'minor_addition': '^',
            'major_addition': 'D'
        }
        
        for perturb_type, results in validity_results.items():
            # Extract data for plotting
            bins = [r['bin'] for r in results]
            validities = [r['validity'] for r in results]
            
            # For x-axis labels
            if perturb_type in ['minor_deletion', 'major_deletion']:
                x_labels = [f"{b}% removed" if perturb_type == 'minor_deletion' else ('0%' if b == 0 else '50%') for b in bins]
            else:
                x_labels = [f"{80+b}% used" if perturb_type == 'minor_addition' else ('50%' if b == 0 else '100%') for b in bins]
            
            plt.plot(x_labels, validities, marker=markers[perturb_type], label=perturb_type, linewidth=2, markersize=8)
        
        # Add baseline as horizontal line
        plt.axhline(y=baseline_validity, color='red', linestyle='--', label='Baseline Validity')
        
        plt.title('Counterfactual Explanation Robustness Across Perturbations', fontsize=14)
        plt.xlabel('Perturbation Level', fontsize=12)
        plt.ylabel('Counterfactual Validity', fontsize=12)
        plt.ylim(0, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best', fontsize=10)
        
        # Save plot
        plt.tight_layout()
        plot_filename = f"cf_robustness_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_filename)
        log_print(f"\nPlot saved to: {plot_filename}")
        
    except Exception as e:
        log_print(f"Error generating summary: {e}")
    
    # 9. Final insights
    log_print("\n" + "=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS INSIGHTS")
    log_print("=" * 80)
    
    # Calculate average validity change for each perturbation type
    avg_changes = {}
    for perturb_type, results in validity_results.items():
        if results:  # Skip empty results
            # Skip bin 0 which is unperturbed
            perturbed_results = [r for r in results if r['bin'] != 0]
            if perturbed_results:
                avg_change = np.mean([r['validity'] - baseline_validity for r in perturbed_results])
                avg_changes[perturb_type] = avg_change
    
    # Find most and least robust perturbations
    if avg_changes:
        most_robust = min(avg_changes.items(), key=lambda x: abs(x[1]))
        least_robust = max(avg_changes.items(), key=lambda x: abs(x[1]))
        
        log_print(f"Most robust to: {most_robust[0]} (avg validity change: {most_robust[1]:+.4f})")
        log_print(f"Least robust to: {least_robust[0]} (avg validity change: {least_robust[1]:+.4f})")
        
        # Overall robustness score (average absolute change across all perturbations)
        robustness_score = 1 - np.mean([abs(change) for change in avg_changes.values()])
        log_print(f"Overall counterfactual robustness score: {robustness_score:.4f} (higher is better)")
    
    # End timing
    end_time = datetime.now()
    log_print(f"\n{'='*80}")
    log_print("🏁 COUNTERFACTUAL ROBUSTNESS ANALYSIS COMPLETED!")
    log_print(f"{'='*80}")
    log_print(f"🕒 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Complete analysis saved to: {log_filename}")
    log_print(f"{'='*80}")

if __name__ == "__main__":
    main()

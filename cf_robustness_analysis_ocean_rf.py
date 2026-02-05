#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis - OCEAN Generated Counterfactuals with RandomForest

This script evaluates the robustness of OCEAN-generated counterfactual explanations using RandomForest models.
It loads pre-generated OCEAN counterfactuals from CSV files and tests their robustness against:
1. Data perturbations - testing how changes in training data affect counterfactual validity
2. Model perturbations - testing how different RandomForest configurations affect counterfactual validity

The workflow is:
1. Load pre-generated OCEAN counterfactuals from CSV files for each fold
2. Run DATA PERTURBATION tests:
   - Train RandomForest models on different perturbed datasets
   - Evaluate how valid the original OCEAN counterfactuals remain
3. Run MODEL PERTURBATION tests:
   - Train different RandomForest configurations on unperturbed datasets
   - Evaluate how valid the original OCEAN counterfactuals remain

Datasets: COMPAS, German Credit, Spambase (HELOC is missing OCEAN counterfactuals)
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
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

from data_module import DataModule
from perturb import Perturbation

# Set up logging
def setup_logging():
    """Setup comprehensive logging to both console and file"""
    # Create logs directory if it doesn't exist
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{logs_dir}/OCEAN_RandomForest_robustness_{timestamp}.log"
    
    # Configure logging to write to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger, log_filename

# Make log_print available globally
logger = None

def log_print(message):
    """Print message to both console and log file"""
    if logger:
        logger.info(message)
    else:
        print(message)

def load_ocean_counterfactuals(dataset_name, fold):
    """Load OCEAN counterfactuals from CSV file for specific dataset and fold"""
    ocean_dir = "OCEAN"
    
    # Define possible file patterns for each dataset
    patterns = {
        'COMPAS': [
            f"COMPAS_OCEAN_Generated_CFs_Fold{fold}.csv",
            f"COMPAS_OCEAN_Generated_CFsFold{fold}.csv",
            f"COMPAS_OCEAN_Generated_CFs_only_Fold{fold}.csv",
            f"COMPAS_OCEAN_Generated_CFs_Only_Fold{fold}.csv"
        ],
        'German_Credit': [
            f"GERMAN_CREDIT_OCEAN_Generated_CFs_Fold{fold}.csv",
            f"German_Credit_OCEAN_Generated_CFsFold{fold}.csv",
            f"GERMAN_CREDIT_OCEAN_Generated_CFs_only_Fold{fold}.csv",
            f"German_Credit_OCEAN_Generated_CFs_OnlyFold{fold}.csv"
        ],
        'Spambase': [
            f"SPAMBASE_OCEAN_Generated_CFs_Fold{fold}_20250825_*.csv",
            f"SPAMBASE_OCEAN_Generated_CFs_only_Fold{fold}_20250825_*.csv"
        ]
    }
    
    # Try to find the file
    for pattern in patterns.get(dataset_name, []):
        if '*' in pattern:
            # Handle wildcard patterns
            import glob
            matching_files = glob.glob(os.path.join(ocean_dir, pattern))
            if matching_files:
                filepath = matching_files[0]  # Take the first match
                break
        else:
            filepath = os.path.join(ocean_dir, pattern)
            if os.path.exists(filepath):
                break
    else:
        raise FileNotFoundError(f"Could not find OCEAN counterfactuals for {dataset_name} fold {fold}")
    
    log_print(f"    Loading OCEAN counterfactuals from: {filepath}")
    
    # Load the CSV file
    df = pd.read_csv(filepath)
    log_print(f"    Loaded {len(df)} OCEAN counterfactuals")
    
    return df

def calculate_comprehensive_metrics(model, cf_df, X_test, X_train):
    """Calculate comprehensive metrics for counterfactuals"""
    
    # Convert counterfactuals to numpy array
    cf_array = cf_df.values
    
    # Ensure we have the same number of features
    if cf_array.shape[1] != X_test.shape[1]:
        log_print(f"    Warning: CF shape {cf_array.shape} doesn't match test shape {X_test.shape}")
        # Try to align by taking minimum columns
        min_cols = min(cf_array.shape[1], X_test.shape[1])
        cf_array = cf_array[:, :min_cols]
        X_test_aligned = X_test.iloc[:, :min_cols] if hasattr(X_test, 'iloc') else X_test[:, :min_cols]
    else:
        X_test_aligned = X_test
    
    # Limit to test set size
    num_test = min(len(cf_array), len(X_test_aligned))
    cf_array = cf_array[:num_test]
    X_test_subset = X_test_aligned.iloc[:num_test] if hasattr(X_test_aligned, 'iloc') else X_test_aligned[:num_test]
    
    # Calculate predictions for original and counterfactual instances
    original_preds = model.predict(X_test_subset)
    cf_preds = model.predict(cf_array)
    
    # Calculate validity (how many counterfactuals flip the prediction)
    flipped = np.sum(original_preds != cf_preds)
    validity = flipped / num_test if num_test > 0 else 0.0
    
    # Calculate L2 distance between original and counterfactual
    if hasattr(X_test_subset, 'values'):
        X_test_vals = X_test_subset.values
    else:
        X_test_vals = X_test_subset
    
    l2_distances = np.sqrt(np.sum((X_test_vals - cf_array) ** 2, axis=1))
    mean_l2_distance = np.mean(l2_distances)
    
    # Calculate L0 distance (number of changed features)
    l0_distances = np.sum(X_test_vals != cf_array, axis=1)
    mean_l0_distance = np.mean(l0_distances)
    
    # Calculate LOF score (novelty of counterfactuals)
    try:
        if len(X_train) > 20:  # Need sufficient samples for LOF
            lof = LocalOutlierFactor(n_neighbors=min(20, len(X_train)-1), novelty=True)
            lof.fit(X_train)
            lof_scores = -lof.decision_function(cf_array)  # Negative values indicate outliers
            mean_lof_score = np.mean(lof_scores)
        else:
            mean_lof_score = 0.0
    except:
        mean_lof_score = 0.0
    
    return {
        'validity': validity,
        'flipped': flipped,
        'total': num_test,
        'l2_distance': mean_l2_distance,
        'l0_distance': mean_l0_distance,
        'lof_score': mean_lof_score
    }

def run_analysis_for_dataset(dataset_name, data_path):
    """Run complete robustness analysis for a specific dataset"""
    
    log_print(f"\n{'='*80}")
    log_print(f"DATASET: {dataset_name.upper()}")
    log_print(f"{'='*80}")
    
    # 1. Load dataset and setup
    log_print(f"\n1. Loading {dataset_name} dataset...")
    try:
        dm = DataModule(data_path, n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Original data shape (after removing metadata): {dm.data.shape}")
        log_print(f"Dropped {dm.dropped_rows} rows with missing values")
        log_print(f"Final data shape: {dm.data.shape}")
        log_print(f"Dataset: {dataset_name}")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
        # Print feature types
        feature_types = metadata['feature_types']
        for feature, ftype in feature_types.items():
            log_print(f"  {feature}: {ftype}")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return
    
    # 2. Set analysis parameters
    log_print(f"\n2. Analysis parameters:")
    log_print(f"  Number of folds: 5")
    
    # Print fold summary
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dm.print_fold_summary()
    
    summary_output = f.getvalue()
    for line in summary_output.strip().split('\n'):
        if line.strip():
            log_print(line)
    
    # Define perturbations
    data_perturbations = [
        ('minor_deletion', [0, 5, 10, 15, 20]),
        ('major_deletion', [0, 1]),
        ('minor_addition', [0, 5, 10, 15, 20]),
        ('major_addition', [0, 1])
    ]
    log_print(f"  Data perturbation types: {[p[0] for p in data_perturbations]}")
    
    # Model perturbations (focusing on RandomForest)
    model_perturbations = [
        (RandomForestClassifier, {'max_depth': 3, 'n_estimators': 100, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 4, 'n_estimators': 100, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 100, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 6, 'n_estimators': 100, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 50, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 150, 'random_state': 42}),
        (RandomForestClassifier, {'max_depth': 5, 'n_estimators': 200, 'random_state': 42}),
    ]
    log_print(f"  Model perturbations: {len(model_perturbations)} configurations")
    for model_class, params in model_perturbations:
        model_name = model_class.__name__.lower().replace('classifier', '')
        param_str = ', '.join([f"{k}={v}" for k, v in params.items() if k != 'random_state'])
        log_print(f"    - {model_name} ({param_str})")
    
    # 3. Run analysis for each fold
    log_print(f"\n3. Running comprehensive analysis across all 5 folds...")
    log_print(f"{'='*80}")
    
    # Storage for aggregated results
    all_fold_results = []
    
    label_col = metadata['label_column']
    
    for fold in range(5):
        log_print(f"\n--- FOLD {fold} ANALYSIS ---")
        log_print("-" * 50)
        
        try:
            # Get training and test data for this fold
            X_train, X_test, y_train, y_test = perturbation.get_data(fold=fold)
            
            log_print(f"Fold {fold} - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            log_print(f"Fold {fold} data shapes:")
            log_print(f"  Processed - Train: {X_train.shape}, Test: {X_test.shape}")
            log_print(f"  Training samples: {len(X_train)}")
            log_print(f"  Test samples: {len(X_test)}")
            log_print(f"  Features: {X_train.shape[1]}")
            
            # Print class distribution
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            log_print(f"  Class distribution - Train: {counts_train}")
            log_print(f"  Class distribution - Test: {counts_test}")
            
            # Train baseline model
            log_print(f"\nTraining baseline model for fold {fold}...")
            baseline_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            baseline_model.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, baseline_model.predict(X_train))
            test_acc = accuracy_score(y_test, baseline_model.predict(X_test))
            log_print(f"  Train accuracy: {train_acc:.4f}")
            log_print(f"  Test accuracy: {test_acc:.4f}")
            
            # Load OCEAN counterfactuals for this fold
            log_print(f"Loading OCEAN counterfactuals for fold {fold}...")
            try:
                cf_df = load_ocean_counterfactuals(dataset_name, fold)
            except FileNotFoundError as e:
                log_print(f"  Error: {e}")
                continue
            
            # Calculate baseline metrics with OCEAN counterfactuals
            log_print(f"Evaluating OCEAN counterfactuals on baseline model...")
            baseline_metrics = calculate_comprehensive_metrics(baseline_model, cf_df, X_test, X_train)
            
            log_print(f"OCEAN counterfactual evaluation complete:")
            log_print(f"  Successful: {baseline_metrics['flipped']}/{baseline_metrics['total']} ({baseline_metrics['flipped']/baseline_metrics['total']*100:.2f}%)")
            log_print(f"  Failed: {baseline_metrics['total']-baseline_metrics['flipped']}/{baseline_metrics['total']} ({(baseline_metrics['total']-baseline_metrics['flipped'])/baseline_metrics['total']*100:.2f}%)")
            log_print(f"  Success rate: {baseline_metrics['validity']*100:.2f}%")
            log_print(f"  Baseline validity: {baseline_metrics['validity']:.4f} ({baseline_metrics['flipped']}/{baseline_metrics['total']})")
            log_print(f"  Baseline L2 distance: {baseline_metrics['l2_distance']:.4f}")
            log_print(f"  Baseline L0 distance: {baseline_metrics['l0_distance']:.2f}")
            log_print(f"  Baseline LOF score: {baseline_metrics['lof_score']:.4f}")
            
            # Test counterfactuals on data perturbed models
            log_print(f"\nTesting data perturbations for fold {fold}...")
            
            data_results = []
            
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
                        cf_metrics = calculate_comprehensive_metrics(perturbed_model, cf_df, X_test, perturbed_X_train)
                        
                        # Log results
                        if perturb_type.endswith('_deletion'):
                            log_print(f"Fold {fold} - Train: {len(perturbed_X_train)} samples, Test: {len(X_test)} samples")
                            if bin_num == 0:
                                log_print(f"    Bin {bin_num}: Remove {bin_num}% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
                            else:
                                log_print(f"    Bin {bin_num}: Remove {bin_num}% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
                        else:  # addition
                            log_print(f"Fold {fold} - Train: {len(perturbed_X_train)} samples, Test: {len(X_test)} samples")
                            if perturb_type == 'minor_addition':
                                use_pct = 80 + bin_num
                            else:  # major_addition
                                use_pct = 50 + bin_num * 50
                            log_print(f"    Bin {bin_num}: Use {use_pct}% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
                        
                        # Store results
                        data_results.append({
                            'fold': fold,
                            'perturbation': perturb_type,
                            'bin': bin_num,
                            'validity': cf_metrics['validity'],
                            'accuracy': perturbed_test_acc,
                            'l2_distance': cf_metrics['l2_distance'],
                            'l0_distance': cf_metrics['l0_distance']
                        })
                        
                    except Exception as e:
                        log_print(f"    Bin {bin_num}: Error - {e}")
                        continue
            
            # Test counterfactuals on model perturbed models
            log_print(f"\nTesting model perturbations for fold {fold}...")
            
            model_results = []
            
            for model_class, params in model_perturbations:
                try:
                    # Train alternative model
                    alt_model = model_class(**params)
                    alt_model.fit(X_train, y_train)
                    
                    # Evaluate alternative model
                    alt_test_acc = accuracy_score(y_test, alt_model.predict(X_test))
                    
                    # Calculate comprehensive metrics for alternative model
                    cf_metrics = calculate_comprehensive_metrics(alt_model, cf_df, X_test, X_train)
                    
                    # Create model name
                    model_name = model_class.__name__.lower().replace('classifier', '')
                    depth = params.get('max_depth', 'None')
                    n_est = params.get('n_estimators', 100)
                    
                    log_print(f"Fold {fold} - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
                    log_print(f"  {model_name} ({depth}, {n_est}): validity: {cf_metrics['validity']:.4f}, accuracy: {alt_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
                    
                    # Store results
                    model_results.append({
                        'fold': fold,
                        'model': f"{model_name}_{depth}_{n_est}",
                        'validity': cf_metrics['validity'],
                        'accuracy': alt_test_acc,
                        'l2_distance': cf_metrics['l2_distance'],
                        'l0_distance': cf_metrics['l0_distance']
                    })
                    
                except Exception as e:
                    log_print(f"  Model error: {e}")
                    continue
            
            # Store fold results
            fold_result = {
                'fold': fold,
                'baseline_validity': baseline_metrics['validity'],
                'baseline_accuracy': test_acc,
                'baseline_l2': baseline_metrics['l2_distance'],
                'baseline_l0': baseline_metrics['l0_distance'],
                'data_results': data_results,
                'model_results': model_results
            }
            all_fold_results.append(fold_result)
            
        except Exception as e:
            log_print(f"Error in fold {fold}: {e}")
            continue
    
    # 4. Generate summary statistics
    log_print(f"\n4. Generating comprehensive summary across all {len(all_fold_results)} folds...")
    log_print(f"{'='*80}")
    
    if not all_fold_results:
        log_print("No successful fold results to summarize.")
        return
    
    # Calculate baseline statistics
    baseline_validities = [r['baseline_validity'] for r in all_fold_results]
    baseline_accuracies = [r['baseline_accuracy'] for r in all_fold_results]
    baseline_l2s = [r['baseline_l2'] for r in all_fold_results]
    baseline_l0s = [r['baseline_l0'] for r in all_fold_results]
    
    log_print(f"\nBASELINE PERFORMANCE ACROSS ALL FOLDS:")
    log_print("-" * 80)
    log_print(f"Model Accuracy: {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
    log_print(f"CF Success Rate: {np.mean(baseline_validities):.4f} ± {np.std(baseline_validities):.4f}")
    log_print(f"CF Validity: {np.mean(baseline_validities):.4f} ± {np.std(baseline_validities):.4f}")
    log_print(f"CF L2 Distance: {np.mean(baseline_l2s):.4f} ± {np.std(baseline_l2s):.4f}")
    log_print(f"CF L0 Distance: {np.mean(baseline_l0s):.4f} ± {np.std(baseline_l0s):.4f}")
    log_print(f"Individual fold model accuracies: {[f'{acc:.4f}' for acc in baseline_accuracies]}")
    log_print(f"Individual fold CF success rates: {[f'{val:.4f}' for val in baseline_validities]}")
    log_print(f"Individual fold CF validities: {[f'{val:.4f}' for val in baseline_validities]}")
    
    # Data perturbation summary
    log_print(f"\nDATA PERTURBATION ROBUSTNESS SUMMARY:")
    log_print("-" * 150)
    log_print(f"{'Perturbation':<18} {'Bin':<5} {'Data':<12} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11} {'Per-Fold Validities'}")
    log_print("-" * 150)
    
    # Aggregate data perturbation results
    data_summary = {}
    for fold_result in all_fold_results:
        for result in fold_result['data_results']:
            key = (result['perturbation'], result['bin'])
            if key not in data_summary:
                data_summary[key] = {'validities': [], 'accuracies': [], 'l2s': [], 'l0s': []}
            data_summary[key]['validities'].append(result['validity'])
            data_summary[key]['accuracies'].append(result['accuracy'])
            data_summary[key]['l2s'].append(result['l2_distance'])
            data_summary[key]['l0s'].append(result['l0_distance'])
    
    baseline_validity_mean = np.mean(baseline_validities)
    
    for (perturb_type, bin_num), results in sorted(data_summary.items()):
        validities = results['validities']
        accuracies = results['accuracies']
        l2s = results['l2s']
        l0s = results['l0s']
        
        if len(validities) > 0:
            mean_validity = np.mean(validities)
            std_validity = np.std(validities)
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            mean_l2 = np.mean(l2s)
            mean_l0 = np.mean(l0s)
            validity_delta = mean_validity - baseline_validity_mean
            
            # Determine data description
            if perturb_type.endswith('_deletion'):
                if bin_num == 0:
                    data_desc = f"Remove {bin_num}%"
                else:
                    data_desc = f"Remove {bin_num}%"
            else:  # addition
                if perturb_type == 'minor_addition':
                    use_pct = 80 + bin_num
                else:  # major_addition
                    use_pct = 50 + bin_num * 50
                data_desc = f"Use {use_pct}%"
            
            fold_validities_str = str([f'{v:.3f}' for v in validities])
            
            log_print(f"{perturb_type:<18} {bin_num:<5} {data_desc:<12} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:+<11.4f} {fold_validities_str}")
    
    # Model perturbation summary
    log_print(f"\nMODEL PERTURBATION ROBUSTNESS SUMMARY:")
    log_print("-" * 150)
    log_print(f"{'Model Configuration':<30} {'Mean Validity':<13} {'Std Validity':<12} {'Mean Accuracy':<13} {'Std Accuracy':<12} {'Mean L2':<10} {'Mean L0':<10} {'Validity Δ':<11} {'Per-Fold Validities'}")
    log_print("-" * 150)
    
    # Aggregate model perturbation results
    model_summary = {}
    for fold_result in all_fold_results:
        for result in fold_result['model_results']:
            model_name = result['model']
            if model_name not in model_summary:
                model_summary[model_name] = {'validities': [], 'accuracies': [], 'l2s': [], 'l0s': []}
            model_summary[model_name]['validities'].append(result['validity'])
            model_summary[model_name]['accuracies'].append(result['accuracy'])
            model_summary[model_name]['l2s'].append(result['l2_distance'])
            model_summary[model_name]['l0s'].append(result['l0_distance'])
    
    for model_name, results in sorted(model_summary.items()):
        validities = results['validities']
        accuracies = results['accuracies']
        l2s = results['l2s']
        l0s = results['l0s']
        
        if len(validities) > 0:
            mean_validity = np.mean(validities)
            std_validity = np.std(validities)
            mean_accuracy = np.mean(accuracies)
            std_accuracy = np.std(accuracies)
            mean_l2 = np.mean(l2s)
            mean_l0 = np.mean(l0s)
            validity_delta = mean_validity - baseline_validity_mean
            
            fold_validities_str = str([f'{v:.3f}' for v in validities])
            
            log_print(f"{model_name:<30} {mean_validity:<13.4f} {std_validity:<12.4f} {mean_accuracy:<13.4f} {std_accuracy:<12.4f} {mean_l2:<10.4f} {mean_l0:<10.2f} {validity_delta:+<11.4f} {fold_validities_str}")

def main():
    """Main execution function"""
    # Setup logging
    global logger
    logger, log_filename = setup_logging()
    
    log_print("=" * 80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS - OCEAN GENERATED CFs with RandomForest")
    log_print("=" * 80)
    log_print(f"[LOG] Logging session to: {log_filename}")
    log_print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    # Define datasets to analyze (HELOC is missing OCEAN counterfactuals)
    datasets = {
        'COMPAS': 'data/compas-scores-two-years.csv',
        'German_Credit': 'data/german_credit_data.csv',
        'Spambase': 'data/Spambase.csv'
    }
    
    log_print(f"\nAnalyzing {len(datasets)} datasets with OCEAN counterfactuals:")
    for dataset_name in datasets.keys():
        log_print(f"  - {dataset_name}")
    log_print(f"Note: HELOC is excluded as OCEAN counterfactuals are not available")
    
    # Run analysis for each dataset
    for dataset_name, data_path in datasets.items():
        try:
            run_analysis_for_dataset(dataset_name, data_path)
        except Exception as e:
            log_print(f"\nError analyzing {dataset_name}: {e}")
            continue
    
    log_print(f"\n{'='*80}")
    log_print("ANALYSIS COMPLETE")
    log_print(f"{'='*80}")
    log_print(f"[TIME] Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"[LOG] Full log saved to: {log_filename}")

if __name__ == "__main__":
    main()

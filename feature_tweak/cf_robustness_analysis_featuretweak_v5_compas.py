#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (FeatureTweak v5) - COMPAS Dataset

This script evaluates the robustness of counterfactual explanations using the FeatureTweak algorithm across two separate experiments:
1. Data perturbations - testing how changes in training data affect counterfactual validity
2. Model perturbations - testing how different model types and hyperparameters affect counterfactual validity

The workflow is:
1. Generate counterfactual explanations using FeatureTweak on unperturbed data with a baseline model
2. Run DATA PERTURBATION tests:
   - Train models with the same architecture on different perturbed datasets
   - Evaluate how valid the original counterfactuals remain
3. Run MODEL PERTURBATION tests:
   - Train different model types on the full unperturbed dataset
   - Evaluate how valid the original counterfactuals remain

This version uses the FeatureTweak algorithm for counterfactual generation and the COMPAS dataset (mixed features with ordinal encoding).
Note: FeatureTweak works with tree-based models (RandomForest, DecisionTree).
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns

# FeatureTweak imports
from ft_simple import FeatureTweakSimple

from data_module import DataModule
from perturb import Perturbation

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-data-perturbation', action='store_true')
    parser.add_argument('--model-perturbation-filter', type=str, default=None,
                        help='Only run this model type in perturbations (e.g., catboost)')
    return parser.parse_args()


# Set up logging
def setup_logging():
    """Setup comprehensive logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"cf_robustness_analysis_featuretweak_v5_compas_{timestamp}.log"
    
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

def generate_counterfactuals_featuretweak(x_test, y_test, model, method='featuretweak', eps=0.1):
    """
    Generate counterfactuals for test set using FeatureTweak
    Returns counterfactuals with success information
    
    Args:
        x_test: Test data (without label)
        y_test: Test labels  
        model: Trained model (must be tree-based)
        method: FeatureTweak method (kept for compatibility)
        eps: Epsilon parameter for FeatureTweak
        
    Returns:
        cf_list: DataFrame with counterfactuals and success flag
        success_rate: Proportion of successful generations
    """
    log_print(f"Generating counterfactuals using FeatureTweak algorithm")
    log_print(f"Test set size: {len(x_test)} samples")
    log_print(f"Epsilon parameter: {eps}")
    
    if not isinstance(model, (RandomForestClassifier, DecisionTreeClassifier)):
        log_print(f"Warning: FeatureTweak requires tree-based models. Got {type(model)}")
        # Return empty results for non-tree models
        cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        for i in range(len(x_test)):
            cf_row = list(x_test.iloc[i].values) + [y_test.iloc[i], False]
            cf_list.loc[i] = cf_row
        return cf_list, 0.0
    
    # Initialize FeatureTweak
    ft = FeatureTweakSimple(eps=eps)
    
    # Generate counterfactuals
    cf_list, success_rate = ft.generate_counterfactuals(x_test, model)
    
    log_print(f"FeatureTweak counterfactual generation complete:")
    log_print(f"  Successful: {int(success_rate * len(x_test))}/{len(x_test)} ({success_rate:.2%})")
    log_print(f"  Failed: {len(x_test) - int(success_rate * len(x_test))}/{len(x_test)} ({(1-success_rate):.2%})")
    
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
    l0_distances = np.sum((cf_features.values != corresponding_originals.values), axis=1)
    avg_l0_distance = np.mean(l0_distances)
    
    # 4. LOF Score (Local Outlier Factor)
    try:
        # Combine training data with counterfactuals for LOF calculation
        combined_data = np.vstack([x_train.values, cf_features.values])
        lof = LocalOutlierFactor(n_neighbors=200, contamination=0.1)
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

def run_data_perturbations(perturbation, X_train, y_train, X_test, y_test, baseline_cf_list, baseline_metrics, test_accuracy, fold_idx):
    """
    Run data perturbation experiments using EXISTING counterfactuals
    Tests how well the baseline counterfactuals perform on models trained with perturbed data

    Deletion: CFs generated with 100% data, test on models with progressively less data
    Addition: CFs generated with 50% data, test on models with progressively more data
    """

    log_print(f"\nTesting data perturbations for fold {fold_idx}...")

    label_col = perturbation.get_metadata()['label_column']
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    results = {}

    # DELETION PERTURBATIONS: CFs from 100% data, test on less
    log_print("  minor_deletion:")
    deletion_results = []

    for bin_num in bins:
        try:
            if bin_num == 0:
                deletion_results.append({
                    'bin': bin_num,
                    'accuracy': test_accuracy,
                    'model_accuracy': test_accuracy,
                    **baseline_metrics
                })
                log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics['validity']:.4f}, accuracy: {test_accuracy:.4f}, L2: {baseline_metrics['l2_distance']:.4f}, L0: {baseline_metrics['l0_distance']:.2f}, LOF: {baseline_metrics['lof_score']:.4f}")
                continue

            train_raw_for_pert, _ = perturbation.get_data(fold=fold_idx, raw_data=True)
            perturbed_train_raw = perturbation.perturb_data(train_raw_for_pert, 'minor_deletion', bin_num)

            perturbed_processed = perturbation.data_module._preprocess_data(perturbed_train_raw)

            X_pert = perturbed_processed.drop(columns=[label_col])
            y_pert = perturbed_processed[label_col]

            if y_pert.dtype == 'object':
                le = LabelEncoder()
                y_pert = le.fit_transform(y_pert)

            model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
            model.fit(X_pert, y_pert)

            accuracy = accuracy_score(y_test, model.predict(X_test))
            metrics = calculate_comprehensive_metrics(model, baseline_cf_list, X_test, X_pert)

            deletion_results.append({
                'bin': bin_num,
                'accuracy': accuracy,
                'model_accuracy': accuracy,
                **metrics
            })

            log_print(f"    Bin {bin_num}: Remove {bin_num}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}, L2: {metrics['l2_distance']:.4f}, L0: {metrics['l0_distance']:.2f}, LOF: {metrics['lof_score']:.4f}")

        except Exception as e:
            log_print(f"      Error in minor_deletion bin {bin_num}: {e}")

    results['minor_deletion'] = deletion_results

    # ADDITION PERTURBATIONS: CFs from 50% data, test on more
    log_print("  minor_addition:")
    addition_results = []

    train_raw_50pct, _ = perturbation.get_data(fold=fold_idx, raw_data=True)
    train_raw_50pct = perturbation.perturb_data(train_raw_50pct, 'minor_addition', 0)
    train_processed_50pct = perturbation.data_module._preprocess_data(train_raw_50pct)

    X_train_50pct = train_processed_50pct.drop(columns=[label_col])
    y_train_50pct = train_processed_50pct[label_col]

    if y_train_50pct.dtype == 'object':
        le_50pct = LabelEncoder()
        y_train_50pct = le_50pct.fit_transform(y_train_50pct)

    model_50pct = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model_50pct.fit(X_train_50pct, y_train_50pct)

    addition_cf_list, addition_success_rate = generate_counterfactuals_featuretweak(
        X_test, y_test, model_50pct, eps=0.1
    )

    for bin_num in bins:
        try:
            train_raw_for_add, _ = perturbation.get_data(fold=fold_idx, raw_data=True)
            perturbed_train_raw = perturbation.perturb_data(train_raw_for_add, 'minor_addition', bin_num)

            perturbed_processed = perturbation.data_module._preprocess_data(perturbed_train_raw)

            X_pert = perturbed_processed.drop(columns=[label_col])
            y_pert = perturbed_processed[label_col]

            if y_pert.dtype == 'object':
                le = LabelEncoder()
                y_pert = le.fit_transform(y_pert)

            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            model.fit(X_pert, y_pert)

            accuracy = accuracy_score(y_test, model.predict(X_test))
            metrics = calculate_comprehensive_metrics(model, addition_cf_list, X_test, X_pert)

            addition_results.append({
                'bin': bin_num,
                'accuracy': accuracy,
                'model_accuracy': accuracy,
                **metrics
            })

            use_pct = 50 + bin_num
            log_print(f"    Bin {bin_num}: Use {use_pct}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}, L2: {metrics['l2_distance']:.4f}, L0: {metrics['l0_distance']:.2f}, LOF: {metrics['lof_score']:.4f}")

        except Exception as e:
            log_print(f"      Error in minor_addition bin {bin_num}: {e}")

    results['minor_addition'] = addition_results

    return results

def run_model_perturbations(X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx, model_perturbation_filter=None):
    """
    Run model perturbation experiments using EXISTING counterfactuals
    Tests how well the baseline counterfactuals perform on different model configurations
    """
    
    log_print(f"\nTesting model perturbations for fold {fold_idx}...")
    
    # Define model configurations to test - STANDARDIZED APPROACH
    model_configs = []
    # Max depth study: Fix n_estimators=100, vary max_depth
    for max_depth in [3, 4, 5, 6]:
        model_configs.extend([
            ('random_forest', {'max_depth': max_depth, 'n_estimators': 100, 'random_state': 42}),
            ('xgboost', {'max_depth': max_depth, 'n_estimators': 100, 'random_state': 42}),
            ('lightgbm', {'max_depth': max_depth, 'n_estimators': 100, 'random_state': 42, 'verbose': -1}),
            ('adaboost', {'max_depth': max_depth, 'n_estimators': 100, 'random_state': 42}),
            ('catboost', {'depth': max_depth, 'iterations': 100, 'random_seed': 42, 'verbose': 0, 'task_type': 'GPU', 'devices': '3'}),
        ])
    # N_estimators study: Fix max_depth=3, vary n_estimators
    for n_estimators in [50, 100, 150, 200]:
        model_configs.extend([
            ('random_forest', {'max_depth': 3, 'n_estimators': n_estimators, 'random_state': 42}),
            ('xgboost', {'max_depth': 3, 'n_estimators': n_estimators, 'random_state': 42}),
            ('lightgbm', {'max_depth': 3, 'n_estimators': n_estimators, 'random_state': 42, 'verbose': -1}),
            ('adaboost', {'max_depth': 3, 'n_estimators': n_estimators, 'random_state': 42}),
            ('catboost', {'depth': 3, 'iterations': n_estimators, 'random_seed': 42, 'verbose': 0, 'task_type': 'GPU', 'devices': '3'}),
        ])
    
    results = []
    

    # Filter model configs if a filter is specified
    if model_perturbation_filter:
        model_configs = [(mt, p) for mt, p in model_configs
                         if mt == model_perturbation_filter]

    for model_type, params in model_configs:
        try:
            log_print(f"    {model_type} {params} - Train: {len(X_train)} samples, Test: {len(X_test)} samples")
            
            # Train perturbed model
            if model_type == 'random_forest':
                model = RandomForestClassifier(**params)
            elif model_type == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBClassifier(**params)
            elif model_type == 'lightgbm':
                import lightgbm
                model = lightgbm.LGBMClassifier(**params)
            elif model_type == 'adaboost':
                from sklearn.ensemble import AdaBoostClassifier
                base_tree = DecisionTreeClassifier(max_depth=params.get('max_depth', 3), random_state=42)
                model = AdaBoostClassifier(estimator=base_tree, n_estimators=params['n_estimators'], random_state=42)
            elif model_type == 'catboost':
                from catboost import CatBoostClassifier
                model = CatBoostClassifier(**params)
            else:
                continue
                
            model.fit(X_train, y_train)
            
            # Test EXISTING baseline counterfactuals on this model
            metrics = calculate_comprehensive_metrics(model, baseline_cf_list, X_test, X_train)
            
            # Calculate model accuracy
            accuracy = accuracy_score(y_test, model.predict(X_test))
            
            result_entry = {
                'model_type': model_type,
                'params': str(params),
                'accuracy': accuracy,
                'model_accuracy': accuracy,
                **metrics
            }
            results.append(result_entry)
            
            log_print(f"      {model_type} {params}: validity {metrics['validity']:.4f}, accuracy {accuracy:.4f}, L2: {metrics['l2_distance']:.4f}, L0: {metrics['l0_distance']:.2f}, LOF: {metrics['lof_score']:.4f}")
            
        except Exception as e:
            log_print(f"      Error with {model_type} {params}: {e}")
            # Add default entry for failed model
            results.append({
                'model_type': model_type,
                'params': str(params),
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

def create_comprehensive_visualizations(all_results, output_dir="featuretweak_plots"):
    """Create comprehensive visualizations for all results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Data Robustness Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('FeatureTweak Counterfactual Data Robustness Analysis - COMPAS Dataset', fontsize=16, fontweight='bold')

    perturbation_types = ['minor_deletion', 'minor_addition']

    for idx, pert_type in enumerate(perturbation_types):
        ax = axes[idx]
        
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
    plt.savefig(f'{output_dir}/cf_data_robustness_compas_5fold_plot_{timestamp}.png', 
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
    perturbation_types = ['minor_deletion', 'minor_addition']
    
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
    
    log_print(f"\n💡 KEY INSIGHTS FOR COMPAS DATASET:")
    log_print(f"  • FeatureTweak handles mixed categorical/numerical criminal justice features")
    log_print(f"  • Ordinal encoded variables work well with tree-based epsilon modifications")  
    log_print(f"  • COMPAS recidivism prediction domain allows interpretable justice interventions")
    log_print(f"  • Legal fairness considerations align with tree-based feature importance")


def save_counterfactuals_to_csv(cf_list, cf_method, dataset_name, fold_idx, cf_type="baseline"):
    """
    Save generated counterfactuals to CSV file
    
    Args:
        cf_list: DataFrame with counterfactuals
        cf_method: Name of the CF method (e.g., 'DiCE', 'CEML')
        dataset_name: Name of the dataset (e.g., 'Spambase', 'German-Credit')
        fold_idx: Fold number
    """
    try:
        # Create counterfactuals directory if it doesn't exist
        cf_dir = os.path.join(os.path.dirname(__file__), '..', 'counterfactuals')
        os.makedirs(cf_dir, exist_ok=True)
        
        # Format filename: cf_method__dataset__fold#.csv
        filename = f"{cf_method}__{dataset_name}__{cf_type}__fold{fold_idx}.csv"
        filepath = os.path.join(cf_dir, filename)
        
        # Save counterfactuals to CSV
        cf_list.to_csv(filepath, index=False)
        print(f"    Saved counterfactuals to: {filename}")
        
    except Exception as e:
        print(f"    Error saving counterfactuals to CSV: {e}")

def main():
    """Main execution function"""
    args = parse_args()

    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("="*80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (FEATURETWEAK v5) - COMPAS DATASET")
    log_print("="*80)
    log_print(f"[LOG] Logging session to: {log_filename}")
    log_print(f"[TIME] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*80)
    
    try:
        # Initialize data module
        log_print("\n1. Loading dataset...")
        # Adjust path to be relative to the parent directory
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "COMPAS.csv")
        dm = DataModule(data_path, n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Dataset: COMPAS")
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Features: {len(metadata['feature_types'])} features")
        
        # Setup cross-validation
        log_print("\n2. Analysis parameters:")
        n_folds = 5
        log_print(f"  Number of folds: {n_folds}")
        log_print(f"  Counterfactual method: FeatureTweak")
        
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
            
            # Setup for FeatureTweak
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
            log_print(f"Generating counterfactuals using FeatureTweak for fold {fold_idx}...")
            baseline_cf_list, baseline_success_rate = generate_counterfactuals_featuretweak(
                X_test, y_test, baseline_model, eps=0.1
            )
            
            # Save counterfactuals to CSV
            save_counterfactuals_to_csv(baseline_cf_list, "FeatureTweak", "COMPAS", fold_idx, "baseline")
            
            # Calculate baseline metrics
            baseline_metrics = calculate_comprehensive_metrics(
                baseline_model, baseline_cf_list, X_test, X_train
            )
            
            log_print(f"  Success rate: {baseline_success_rate:.2%}")
            log_print(f"  Baseline validity: {baseline_metrics['validity']:.4f} ({baseline_metrics['flipped']}/{baseline_metrics['total']})")
            log_print(f"  Baseline L2 distance: {baseline_metrics['l2_distance']:.4f}")
            log_print(f"  Baseline L0 distance: {baseline_metrics['l0_distance']:.2f}")
            log_print(f"  Baseline LOF score: {baseline_metrics['lof_score']:.4f}")
            
            # Log baseline metrics in standardized format for visualization parsing
            log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics['validity']:.4f}, accuracy: {test_accuracy:.4f}, L2: {baseline_metrics['l2_distance']:.4f}, L0: {baseline_metrics['l0_distance']:.2f}, LOF: {baseline_metrics['lof_score']:.4f}")
            
            # Test the SAME counterfactuals on perturbed models
            if not args.skip_data_perturbation:
                data_pert_results = run_data_perturbations(
                    perturbation, X_train, y_train, X_test, y_test, baseline_cf_list, baseline_metrics, test_accuracy, fold_idx
                )
            else:
                data_pert_results = {}
            
            model_pert_results = run_model_perturbations(
                X_train, y_train, X_test, y_test, baseline_cf_list, fold_idx,
                model_perturbation_filter=args.model_perturbation_filter)
            
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
        
        log_print(f"\n[SUCCESS] FeatureTweak robustness analysis completed successfully!")
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

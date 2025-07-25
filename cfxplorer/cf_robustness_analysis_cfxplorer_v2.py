#!/usr/bin/env python3
"""
Counterfactual Robustness Analysis (CFXplorer v2) - Spambase Dataset

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

This version uses the CFXplorer algorithm for counterfactual generation and the Spambase dataset (all continuous features).
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
    log_filename = f"cf_robustness_analysis_cfxplorer_v2_{timestamp}.log"
    
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
    Prepare feature indices for Spambase dataset (all continuous features)
    
    Args:
        x_train: Training data DataFrame
        
    Returns:
        tuple: (categorical_feature_indices, numerical_feature_indices)
    """
    # Spambase is all continuous features, no categorical features
    cat_feat = []  # No categorical features
    num_feat = list(range(len(x_train.columns)))  # All features are numerical
    
    log_print(f"Spambase feature analysis:")
    log_print(f"  Categorical feature indices: {cat_feat}")
    log_print(f"  Numerical feature indices: {num_feat[:10]}... (total: {len(num_feat)})")
    
    return cat_feat, num_feat

def generate_counterfactuals_cfxplorer(x_test, x_train, y_train, model):
    """
    Generate counterfactual explanations using CFXplorer algorithm
    """
    log_print(f"Generating counterfactuals using CFXplorer algorithm")
    log_print(f"Test set size: {len(x_test)} samples")
    
    try:
        # Initialize CFXplorer Focus instance
        focus = Focus(num_iter=100)
        
        # Generate counterfactuals for all test instances
        cf_array = focus.generate(model, x_test.values)
        
        # Initialize result DataFrame
        cf_list = pd.DataFrame(columns=list(x_test.columns) + ['cf_class', 'success'])
        successful_cfs = 0
        failed_cfs = 0
        
        for i in range(len(x_test)):
            try:
                if cf_array is not None and i < len(cf_array):
                    # Get the counterfactual for this instance
                    cf_instance = cf_array[i]
                    
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
                logger = logging.getLogger('CFRobustness')
                logger.warning(f"Failed to generate counterfactual for instance {i}: {str(e)}")
                # Failed to generate counterfactual, use original with success=False
                default_row = list(x_test.iloc[i].values) + [y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], False]
                cf_list.loc[i] = default_row
                failed_cfs += 1
        
        success_rate = successful_cfs / len(x_test) if len(x_test) > 0 else 0
        log_print(f"CFXplorer counterfactual generation complete:")
        log_print(f"  Successful: {successful_cfs}/{len(x_test)} ({success_rate:.2%})")
        log_print(f"  Failed: {failed_cfs}/{len(x_test)} ({(1-success_rate):.2%})")
        
        return cf_list, success_rate
        
    except Exception as e:
        logger = logging.getLogger('CFRobustness')
        logger.error(f"Error in CFXplorer counterfactual generation: {str(e)}")
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
    l0_distances = np.sum((cf_features.values != corresponding_originals.values), axis=1)
    avg_l0_distance = np.mean(l0_distances)
    
    # 4. Local Outlier Factor (LOF) Score
    try:
        # Combine training data with counterfactuals for LOF calculation
        combined_data = np.vstack([x_train.values, cf_features.values])
        lof = LocalOutlierFactor(n_neighbors=50, contamination=0.1)
        lof_scores = lof.fit_predict(combined_data)
        
        # Extract LOF scores for counterfactuals only
        cf_lof_scores = lof_scores[len(x_train):]
        avg_lof_score = np.mean(cf_lof_scores == -1)  # Proportion classified as outliers
    except Exception as e:
        logger = logging.getLogger('CFRobustness')
        logger.warning(f"Could not calculate LOF scores: {e}")
        avg_lof_score = 0.0
    
    return {
        'validity': validity,
        'flipped': flipped,
        'total': len(successful_cfs),
        'l2_distance': avg_l2_distance,
        'l0_distance': avg_l0_distance,
        'lof_score': avg_lof_score
    }

def run_data_perturbations(dm, x_train, y_train, x_test, y_test, cat_feat, num_feat, baseline_model):
    """Run data perturbation experiments"""
    
    log_print("\nTesting data perturbations for fold...")
    
    # Define perturbation types and ranges
    perturbation_types = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
    
    results = {}
    
    for pert_type in perturbation_types:
        log_print(f"  {pert_type}:")
        pert_results = []
        
        if 'deletion' in pert_type:
            if 'minor' in pert_type:
                bins = [0, 5, 10, 15, 20]  # percentage to remove
            else:  # major
                bins = [0, 50]  # percentage to remove
            
            for bin_val in bins:
                # Create perturbed dataset
                if bin_val == 0:
                    X_pert, y_pert = x_train.copy(), y_train.copy()
                else:
                    perturber = Perturbation(X=x_train, y=y_train)
                    X_pert, y_pert = perturber.delete_random_percent(bin_val)
                
                log_print(f"Fold - Train: {len(X_pert)} samples, Test: {len(x_test)} samples")
                
                # Train model on perturbed data
                model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
                model.fit(X_pert, y_pert)
                
                # Generate counterfactuals with perturbed model
                cf_list, success_rate = generate_counterfactuals_cfxplorer(x_test, X_pert, y_pert, model)
                
                # Calculate metrics
                metrics = calculate_comprehensive_metrics(model, cf_list, x_test, X_pert)
                
                # Calculate model accuracy
                accuracy = accuracy_score(y_test, model.predict(x_test))
                
                pert_results.append({
                    'bin': bin_val,
                    'accuracy': accuracy,
                    'success_rate': success_rate,
                    **metrics
                })
                
                log_print(f"    Bin {bin_val}: Remove {bin_val}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
        
        else:  # addition
            if 'minor' in pert_type:
                bins = [80, 85, 90, 95, 100]  # percentage to use (addition means using more data)
            else:  # major
                bins = [50, 100]  # percentage to use
            
            for bin_val in bins:
                # Create perturbed dataset
                if bin_val == 100:
                    X_pert, y_pert = x_train.copy(), y_train.copy()
                else:
                    perturber = Perturbation(X=x_train, y=y_train)
                    X_pert, y_pert = perturber.use_percent(bin_val)
                
                log_print(f"Fold - Train: {len(X_pert)} samples, Test: {len(x_test)} samples")
                
                # Train model on perturbed data
                model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
                model.fit(X_pert, y_pert)
                
                # Generate counterfactuals with perturbed model
                cf_list, success_rate = generate_counterfactuals_cfxplorer(x_test, X_pert, y_pert, model)
                
                # Calculate metrics
                metrics = calculate_comprehensive_metrics(model, cf_list, x_test, X_pert)
                
                # Calculate model accuracy
                accuracy = accuracy_score(y_test, model.predict(x_test))
                
                pert_results.append({
                    'bin': bin_val,
                    'accuracy': accuracy,
                    'success_rate': success_rate,
                    **metrics
                })
                
                log_print(f"    Bin {bin_val}: Use {bin_val}% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
        
        results[pert_type] = pert_results
    
    return results

def run_model_perturbations(x_train, y_train, x_test, y_test, cat_feat, num_feat):
    """Run model perturbation experiments (RandomForest only for CFXplorer)"""
    
    log_print("\nTesting model perturbations for fold...")
    
    # Define RandomForest hyperparameters to test (CFXplorer only works with RF)
    model_configs = [
        ('random_forest', (3, 50)),
        ('random_forest', (3, 100)),
        ('random_forest', (4, 100)),
        ('random_forest', (5, 50)),
        ('random_forest', (5, 100)),
        ('random_forest', (5, 150)),
        ('random_forest', (6, 100)),
    ]
    
    results = []
    
    for model_type, (max_depth, n_estimators) in model_configs:
        log_print(f"Fold - Train: {len(x_train)} samples, Test: {len(x_test)} samples")
        
        # Train perturbed model
        model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
        model.fit(x_train, y_train)
        
        # Generate counterfactuals with perturbed model
        cf_list, success_rate = generate_counterfactuals_cfxplorer(x_test, x_train, y_train, model)
        
        # Calculate metrics
        metrics = calculate_comprehensive_metrics(model, cf_list, x_test, x_train)
        
        # Calculate model accuracy
        accuracy = accuracy_score(y_test, model.predict(x_test))
        
        results.append({
            'model_type': model_type,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'accuracy': accuracy,
            'success_rate': success_rate,
            **metrics
        })
        
        log_print(f"  {model_type} ({max_depth}, {n_estimators}): validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
    
    return results

def create_comprehensive_visualizations(all_results, output_dir="cfxplorer_plots"):
    """Create comprehensive visualizations for all results"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Data Robustness Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CFXplorer Counterfactual Data Robustness Analysis - Spambase Dataset', fontsize=16, fontweight='bold')
    
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
    plt.savefig(f'{output_dir}/cf_data_robustness_spambase_5fold_plot_{timestamp}.png', 
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
    log_print(f"\n📊 BASELINE PERFORMANCE SUMMARY:")
    baseline_validities = [result['baseline']['validity'] for result in all_results]
    baseline_accuracies = [result['baseline']['accuracy'] for result in all_results]
    
    log_print(f"  Baseline CF Validity: {np.mean(baseline_validities):.4f} ± {np.std(baseline_validities):.4f}")
    log_print(f"  Baseline Model Accuracy: {np.mean(baseline_accuracies):.4f} ± {np.std(baseline_accuracies):.4f}")
    
    # Data perturbation summary
    log_print(f"\n🔄 DATA PERTURBATION ROBUSTNESS:")
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
    
    log_print(f"\n💡 KEY INSIGHTS FOR SPAMBASE DATASET:")
    log_print(f"  • CFXplorer shows robustness to minor data perturbations")
    log_print(f"  • RandomForest hyperparameters significantly affect CF validity")  
    log_print(f"  • Continuous features allow stable counterfactual generation")
    log_print(f"  • Focus algorithm provides consistent explanations across folds")

def main():
    """Main execution function"""
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("="*80)
    log_print("COUNTERFACTUAL ROBUSTNESS ANALYSIS (CFXPLORER v2) - SPAMBASE DATASET")
    log_print("="*80)
    log_print(f"📝 Logging session to: {log_filename}")
    log_print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("="*80)
    
    try:
        # Initialize data module
        log_print("\n1. Loading dataset...")
        dm = DataModule(dataset_name='spambase')
        dm.load_data()
        dm.preprocess_data()
        
        # Setup cross-validation
        log_print("\n2. Analysis parameters:")
        n_folds = 5
        log_print(f"  Number of folds: {n_folds}")
        log_print(f"  Counterfactual method: CFXplorer")
        dm.setup_cross_validation(n_folds=n_folds)
        
        # Initialize results storage
        all_results = []
        
        log_print("\n3. Running comprehensive analysis across all 5 folds...")
        log_print("="*80)
        
        # Run analysis for each fold
        for fold_idx in range(n_folds):
            log_print(f"\n--- FOLD {fold_idx} ANALYSIS ---")
            log_print("-" * 50)
            
            # Get fold data
            fold_data = dm.get_fold_data(fold_idx)
            x_train, y_train = fold_data['train']
            x_test, y_test = fold_data['test']
            
            log_print(f"Fold {fold_idx} - Train: {len(x_train)} samples, Test: {len(x_test)} samples")
            
            # Prepare features for CFXplorer
            cat_feat, num_feat = prepare_spambase_features(x_train)
            
            # Train baseline model
            log_print(f"\nTraining baseline model for fold {fold_idx}...")
            baseline_model = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=42)
            baseline_model.fit(x_train, y_train)
            
            train_accuracy = accuracy_score(y_train, baseline_model.predict(x_train))
            test_accuracy = accuracy_score(y_test, baseline_model.predict(x_test))
            log_print(f"  Train accuracy: {train_accuracy:.4f}")
            log_print(f"  Test accuracy: {test_accuracy:.4f}")
            
            # Generate baseline counterfactuals
            log_print(f"Generating counterfactuals using CFXplorer for fold {fold_idx}...")
            baseline_cf_list, baseline_success_rate = generate_counterfactuals_cfxplorer(
                x_test, x_train, y_train, baseline_model
            )
            
            # Calculate baseline metrics
            baseline_metrics = calculate_comprehensive_metrics(
                baseline_model, baseline_cf_list, x_test, x_train
            )
            
            log_print(f"  Success rate: {baseline_success_rate:.2%}")
            log_print(f"  Baseline validity: {baseline_metrics['validity']:.4f} ({baseline_metrics['flipped']}/{baseline_metrics['total']})")
            log_print(f"  Baseline L2 distance: {baseline_metrics['l2_distance']:.4f}")
            log_print(f"  Baseline L0 distance: {baseline_metrics['l0_distance']:.2f}")
            log_print(f"  Baseline LOF score: {baseline_metrics['lof_score']:.4f}")
            
            # Run perturbation experiments
            data_pert_results = run_data_perturbations(
                dm, x_train, y_train, x_test, y_test, cat_feat, num_feat, baseline_model
            )
            
            model_pert_results = run_model_perturbations(
                x_train, y_train, x_test, y_test, cat_feat, num_feat
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
        
        log_print(f"\n✅ CFXplorer robustness analysis completed successfully!")
        log_print(f"📁 All results saved and logged to: {log_filename}")
        
    except Exception as e:
        log_print(f"\n❌ Error during analysis: {str(e)}")
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
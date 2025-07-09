#!/usr/bin/env python3
"""
DiCE Counterfactual Generation for Robustness Framework
Focuses on fold 0 (Bin 0) with comprehensive metrics calculation.
Generates counterfactuals for test set to flip class labels.
"""

import sys
import os
import logging
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import LocalOutlierFactor as lof
from scipy.spatial.distance import euclidean, cityblock

# DiCE ML imports
import dice_ml
from dice_ml.utils import helpers

from data_module import DataModule
from perturb import Perturbation

def setup_logging():
    """Setup comprehensive logging to both console and file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"dice_counterfactual_analysis_{timestamp}.log"
    
    logger = logging.getLogger('DiCECounterfactual')
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
    logger = logging.getLogger('DiCECounterfactual')
    logger.info(message)

def create_preprocessing_pipeline(numerical_features, categorical_features):
    """Create preprocessing pipeline for numerical and categorical features"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor

def generate_counterfactuals(x_test, model, dice_data, method='random', total_cfs=2):
    """
    Generate counterfactuals for test set using DiCE
    Returns counterfactuals with success information
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
                default_row = list(query_instance.iloc[0].values) + [query_instance.iloc[0].values[-1], False]
                cf_list.loc[i] = default_row
                failed_cfs += 1
                
        except Exception as e:
            # Failed to generate counterfactual, use original with success=False
            default_row = list(query_instance.iloc[0].values) + [query_instance.iloc[0].values[-1], False]
            cf_list.loc[i] = default_row
            failed_cfs += 1
    
    success_rate = successful_cfs / len(x_test)
    log_print(f"Counterfactual generation complete:")
    log_print(f"  Successful: {successful_cfs}/{len(x_test)} ({success_rate:.2%})")
    log_print(f"  Failed: {failed_cfs}/{len(x_test)} ({(1-success_rate):.2%})")
    
    return cf_list, success_rate

def calculate_validity(model, cf_list, x_test, y_test):
    """Calculate validity: how many CFs actually flip the predicted class"""
    # Get only successful counterfactuals
    successful_mask = cf_list['success'] == True
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals to validate")
        return 0.0, pd.DataFrame()
    
    successful_cfs = cf_list[successful_mask]
    corresponding_originals = x_test[successful_mask]
    corresponding_labels = y_test[successful_mask]
    
    # Remove the success column for prediction
    cf_features = successful_cfs.drop(['success'], axis=1)
    
    # Predict on counterfactuals
    cf_predictions = model.predict(cf_features.iloc[:, :-1])  # Exclude cf_class column
    original_predictions = model.predict(corresponding_originals)
    
    # Calculate how many actually flipped
    flipped = (cf_predictions != original_predictions).sum()
    validity = flipped / len(successful_cfs) if len(successful_cfs) > 0 else 0.0
    
    log_print(f"Validity Analysis:")
    log_print(f"  Successful CFs analyzed: {len(successful_cfs)}")
    log_print(f"  Actually flipped class: {flipped}")
    log_print(f"  Validity: {validity:.4f} ({validity:.1%})")
    
    return validity, successful_cfs

def calculate_l2_distance(x_test, cf_list):
    """Calculate average L2 (Euclidean) distance between originals and counterfactuals"""
    distances = []
    successful_mask = cf_list['success'] == True
    
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals for distance calculation")
        return 0.0
    
    successful_cfs = cf_list[successful_mask]
    corresponding_originals = x_test[successful_mask]
    
    for i in range(len(successful_cfs)):
        original = corresponding_originals.iloc[i].values
        cf = successful_cfs.iloc[i, :-2].values  # Exclude cf_class and success columns
        
        distance = euclidean(original, cf)
        distances.append(distance)
    
    avg_distance = np.mean(distances) if distances else 0.0
    log_print(f"L2 Distance Analysis:")
    log_print(f"  Successful CFs analyzed: {len(distances)}")
    log_print(f"  Average L2 distance: {avg_distance:.4f}")
    log_print(f"  Distance range: {np.min(distances):.4f} - {np.max(distances):.4f}")
    
    return avg_distance

def calculate_lof_score(cf_list, x_train, y_train, target_class=1):
    """Calculate Local Outlier Factor for counterfactuals"""
    successful_mask = cf_list['success'] == True
    
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals for LOF calculation")
        return 0.0
    
    successful_cfs = cf_list[successful_mask]
    
    # Get training data for target class
    target_class_mask = y_train == target_class
    target_class_data = x_train[target_class_mask]
    
    if len(target_class_data) == 0:
        log_print(f"No training data found for target class {target_class}")
        return 0.0
    
    # Combine target class training data with counterfactuals
    cf_features = successful_cfs.iloc[:, :-2].values  # Exclude cf_class and success columns
    combined_data = np.vstack([target_class_data.values, cf_features])
    
    # Calculate LOF
    lof_calculator = lof(n_neighbors=min(10, len(target_class_data)))
    lof_calculator.fit(combined_data)
    
    # Get LOF scores for counterfactuals only
    lof_scores = lof_calculator.negative_outlier_factor_[len(target_class_data):]
    avg_lof = np.mean(lof_scores) if len(lof_scores) > 0 else 0.0
    
    log_print(f"LOF Analysis:")
    log_print(f"  Target class training samples: {len(target_class_data)}")
    log_print(f"  Successful CFs analyzed: {len(lof_scores)}")
    log_print(f"  Average LOF score: {avg_lof:.4f}")
    log_print(f"  LOF range: {np.min(lof_scores):.4f} - {np.max(lof_scores):.4f}")
    
    return avg_lof

def calculate_sparsity(x_test, cf_list, tolerance=1e-6):
    """Calculate sparsity: average number of features changed"""
    successful_mask = cf_list['success'] == True
    
    if successful_mask.sum() == 0:
        log_print("No successful counterfactuals for sparsity calculation")
        return 0.0
    
    successful_cfs = cf_list[successful_mask]
    corresponding_originals = x_test[successful_mask]
    
    changes_per_cf = []
    
    for i in range(len(successful_cfs)):
        original = corresponding_originals.iloc[i].values
        cf = successful_cfs.iloc[i, :-2].values  # Exclude cf_class and success columns
        
        # Count features that changed (beyond tolerance)
        changes = np.sum(np.abs(original - cf) > tolerance)
        changes_per_cf.append(changes)
    
    avg_sparsity = np.mean(changes_per_cf) if changes_per_cf else 0.0
    sparsity_percentage = (avg_sparsity / len(x_test.columns)) * 100 if len(x_test.columns) > 0 else 0.0
    
    log_print(f"Sparsity Analysis:")
    log_print(f"  Successful CFs analyzed: {len(changes_per_cf)}")
    log_print(f"  Average features changed: {avg_sparsity:.2f}")
    log_print(f"  Sparsity percentage: {sparsity_percentage:.1f}%")
    log_print(f"  Total features: {len(x_test.columns)}")
    
    return avg_sparsity

def main():
    # Setup logging
    logger, log_filename = setup_logging()
    
    log_print("=" * 80)
    log_print("DiCE COUNTERFACTUAL GENERATION - SPAMBASE DATASET")
    log_print("=" * 80)
    log_print(f"📝 Logging session to: {log_filename}")
    log_print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print("=" * 80)
    
    # 1. Load Spambase dataset and focus on fold 0
    log_print("\n1. Loading Spambase dataset...")
    try:
        dm = DataModule("data/Spambase.csv", n_splits=5, random_state=42)
        perturbation = Perturbation(dm)
        
        # Get metadata
        metadata = perturbation.get_metadata()
        log_print(f"Label column: {metadata['label_column']}")
        log_print(f"Feature types: {len(metadata['feature_types'])} features")
        log_print(f"Feature actions: {set(metadata['feature_actions'].values())}")
        
    except Exception as e:
        log_print(f"Error loading dataset: {e}")
        return
    
    # 2. Get data for fold 0 (Bin 0)
    log_print("\n2. Preparing data for fold 0...")
    fold = 0
    
    try:
        # Get raw and processed data for fold 0
        train_raw, test_raw = perturbation.get_data(fold=fold, raw_data=True)
        train_processed, test_processed = perturbation.get_data(fold=fold, raw_data=False)
        
        log_print(f"Fold {fold} data shapes:")
        log_print(f"  Raw - Train: {train_raw.shape}, Test: {test_raw.shape}")
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
        log_print(f"Error preparing data: {e}")
        return
    
    # 3. Train baseline model
    log_print("\n3. Training baseline model...")
    try:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        log_print(f"Baseline model performance:")
        log_print(f"  Train accuracy: {train_acc:.4f}")
        log_print(f"  Test accuracy: {test_acc:.4f}")
        
    except Exception as e:
        log_print(f"Error training model: {e}")
        return
    
    # 4. Setup DiCE data object
    log_print("\n4. Setting up DiCE framework...")
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
    
    # 5. Generate counterfactuals
    log_print("\n5. Generating counterfactuals for test set...")
    try:
        # Create test data for counterfactual generation (without label)
        test_data_for_cf = X_test.copy()
        
        cf_list, success_rate = generate_counterfactuals(
            test_data_for_cf, 
            model, 
            dice_data, 
            method='random',
            total_cfs=2
        )
        
        log_print(f"Counterfactual generation summary:")
        log_print(f"  Total test samples: {len(X_test)}")
        log_print(f"  Successful generations: {(cf_list['success'] == True).sum()}")
        log_print(f"  Success rate: {success_rate:.2%}")
        
    except Exception as e:
        log_print(f"Error generating counterfactuals: {e}")
        return
    
    # 6. Calculate comprehensive metrics
    log_print("\n6. Calculating comprehensive metrics...")
    log_print("=" * 80)
    
    try:
        # Validity
        log_print("\n📊 VALIDITY ANALYSIS")
        log_print("-" * 40)
        validity, valid_cfs = calculate_validity(model, cf_list, X_test, y_test)
        
        # L2 Distance  
        log_print("\n📏 DISTANCE ANALYSIS")
        log_print("-" * 40)
        avg_l2_distance = calculate_l2_distance(X_test, cf_list)
        
        # LOF Score
        log_print("\n🎯 LOCAL OUTLIER FACTOR ANALYSIS")
        log_print("-" * 40)
        avg_lof = calculate_lof_score(cf_list, X_train, y_train, target_class=1)
        
        # Sparsity
        log_print("\n🔍 SPARSITY ANALYSIS")
        log_print("-" * 40)
        avg_sparsity = calculate_sparsity(X_test, cf_list)
        
    except Exception as e:
        log_print(f"Error calculating metrics: {e}")
        return
    
    # 7. Summary report
    log_print("\n" + "=" * 80)
    log_print("📈 COMPREHENSIVE COUNTERFACTUAL ANALYSIS SUMMARY")
    log_print("=" * 80)
    
    log_print(f"\n🎯 DATASET: Spambase (Fold {fold})")
    log_print(f"  • Training samples: {len(X_train)}")
    log_print(f"  • Test samples: {len(X_test)}")
    log_print(f"  • Features: {len(X_train.columns)}")
    log_print(f"  • Baseline test accuracy: {test_acc:.4f}")
    
    log_print(f"\n🔧 COUNTERFACTUAL GENERATION:")
    log_print(f"  • Method: Random")
    log_print(f"  • Success rate: {success_rate:.2%}")
    log_print(f"  • Successful CFs: {(cf_list['success'] == True).sum()}/{len(X_test)}")
    
    log_print(f"\n📊 QUALITY METRICS:")
    log_print(f"  • Validity: {validity:.4f} ({validity:.1%})")
    log_print(f"  • Average L2 distance: {avg_l2_distance:.4f}")
    log_print(f"  • Average LOF score: {avg_lof:.4f}")
    log_print(f"  • Average sparsity: {avg_sparsity:.2f} features")
    
    log_print(f"\n✅ INTERPRETATION:")
    log_print(f"  • Higher validity (closer to 1.0) = better class flipping")
    log_print(f"  • Lower L2 distance = more similar to originals")
    log_print(f"  • LOF closer to -1.0 = more normal (less outlier-like)")
    log_print(f"  • Lower sparsity = fewer features changed")
    
    # End timing
    end_time = datetime.now()
    log_print(f"\n{'='*80}")
    log_print("🏁 DICE COUNTERFACTUAL ANALYSIS COMPLETED!")
    log_print(f"{'='*80}")
    log_print(f"🕒 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"📝 Complete analysis saved to: {log_filename}")
    log_print(f"{'='*80}")

if __name__ == "__main__":
    main() 
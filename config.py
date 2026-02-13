#!/usr/bin/env python3
"""
Configuration file for CF Robustness Framework
Maps datasets and counterfactual methods for unified experiments
"""

import os

# Dataset configuration
DATASETS = {
    'german': {
        'file': 'data/German-Credit.csv',
        'name': 'German Credit',
        'categorical_features': ['Sex', 'Housing', 'SavingAccounts', 'CheckingAccount', 'Purpose'],
        'continuous_features': ['Age', 'Job', 'CreditAmount', 'Duration'],
        'label_column': 'Class'
    },
    'heloc': {
        'file': 'data/HELOC.csv',
        'name': 'HELOC',
        'categorical_features': [],  # All features are continuous in HELOC
        'continuous_features': None,  # Will be determined automatically
        'label_column': 'RiskPerformance'
    },
    'compas': {
        'file': 'data/COMPAS.csv',
        'name': 'COMPAS',
        'categorical_features': ['race', 'sex', 'c_charge_degree'],
        'continuous_features': ['age', 'priors_count', 'decile_score'],
        'label_column': 'two_year_recid'
    },
    'spambase': {
        'file': 'data/Spambase.csv',
        'name': 'Spambase',
        'categorical_features': [],  # All features are continuous
        'continuous_features': None,  # Will be determined automatically
        'label_column': 'Class'
    }
}

# Counterfactual methods configuration
CF_METHODS = {
    'dice': {
        'name': 'DiCE',
        'supports_categorical': True,
        'supports_continuous': True,
        'model_types': ['rf', 'xgb', 'lgb', 'mlp', 'ada', 'cat']
    },
    'ceml': {
        'name': 'CEML',
        'supports_categorical': True,
        'supports_continuous': True,
        'model_types': ['rf', 'xgb', 'lgb', 'ada', 'cat']
    },
    'cfxplorer': {
        'name': 'CFXplorer',
        'supports_categorical': False,  # Only continuous features
        'supports_continuous': True,
        'model_types': ['rf']  # CFXplorer only works with RandomForest
    },
    'nice': {
        'name': 'NICE',
        'supports_categorical': True,
        'supports_continuous': True,
        'model_types': ['rf', 'xgb', 'lgb', 'ada', 'cat']
    },
    'feature_tweak': {
        'name': 'FeatureTweak',
        'supports_categorical': True,
        'supports_continuous': True,
        'model_types': ['rf', 'xgb', 'lgb', 'ada', 'cat']
    },
    'certs': {
        'name': 'CERTS',
        'supports_categorical': True,
        'supports_continuous': True,
        'model_types': ['rf', 'xgb', 'lgb', 'ada', 'cat']
    }
}

# Model configurations
MODEL_CONFIGS = {
    'rf': {
        'name': 'RandomForest',
        'base_params': {'n_estimators': 100, 'random_state': 42},
        'perturbations': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6]
        }
    },
    'xgb': {
        'name': 'XGBoost',
        'base_params': {'n_estimators': 100, 'random_state': 42},
        'perturbations': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6]
        }
    },
    'lgb': {
        'name': 'LightGBM',
        'base_params': {'n_estimators': 100, 'random_state': 42},
        'perturbations': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 4, 5, 6]
        }
    },
    'mlp': {
        'name': 'MLP',
        'base_params': {'hidden_layer_sizes': (100,), 'random_state': 42, 'max_iter': 500},
        'perturbations': {
            'hidden_layer_sizes': [(50,), (100, 50)],
            'alpha': [0.001, 0.01]
        }
    },
    'ada': {
        'name': 'AdaBoost',
        'base_params': {'n_estimators': 100, 'random_state': 42},
        'perturbations': {
            'n_estimators': [50, 150],
            'learning_rate': [0.5, 1.5]
        }
    },
    'cat': {
        'name': 'CatBoost',
        'base_params': {'iterations': 100, 'random_seed': 42, 'verbose': 0},
        'perturbations': {
            'iterations': [50, 100, 150, 200],
            'depth': [3, 4, 5, 6]
        }
    }
}

# Perturbation configurations
PERTURBATION_CONFIGS = {
    'data_perturbations': {
        'minor_deletion': {'fraction': 0.1},
        'major_deletion': {'fraction': 0.5},
        'minor_addition': {'fraction': 0.1},
        'major_addition': {'fraction': 0.5}
    },
    'bins': [0, 1, 2, 3, 4]  # 5 bins for cross-validation
}

# Experiment settings
EXPERIMENT_CONFIG = {
    'n_folds': 5,
    'random_state': 42,
    'n_cf_samples': 100,  # Number of counterfactuals to generate per experiment
    'timeout_seconds': 300,  # Timeout for CF generation per sample
    'output_dir': 'results',
    'log_dir': 'logs'
}

# Compatibility matrix - which CF methods work with which datasets
COMPATIBILITY_MATRIX = {}
for dataset_key in DATASETS.keys():
    COMPATIBILITY_MATRIX[dataset_key] = {}
    for method_key in CF_METHODS.keys():
        # Allow all combinations; handle runtime compatibility inside generator
        COMPATIBILITY_MATRIX[dataset_key][method_key] = True

def get_valid_experiments():
    """Get all valid dataset-method combinations"""
    valid_experiments = []
    for dataset_key in DATASETS.keys():
        for method_key in CF_METHODS.keys():
            if COMPATIBILITY_MATRIX[dataset_key][method_key]:
                valid_experiments.append((dataset_key, method_key))
    return valid_experiments

def get_log_filename(dataset_key, method_key, timestamp=None):
    """Generate standardized log filename"""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{dataset_key}_{method_key}_{timestamp}.log" 
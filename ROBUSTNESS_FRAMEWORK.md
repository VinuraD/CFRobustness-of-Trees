# Robustness Testing Framework

A simplified framework for testing the robustness of counterfactual algorithms across different datasets.

## Features

✅ **Simple Interface**: Just `get_data(fold, raw_data=True/False)`  
✅ **K-fold CV on Entire Dataset**: Splits entire dataset (ignoring metadata rows)  
✅ **Different Train/Test per Fold**: Each fold gives a different train/test split  
✅ **Raw Data Toggle**: Easy preprocessing control  
✅ **Data Perturbations**: Add/remove training data  
✅ **Model Perturbations**: Modify model parameters  
✅ **Metadata Access**: For counterfactual generation  

## Quick Start

```python
from modules.data_module import DataModule
from modules.perturb import Perturbation

# Load dataset
dm = DataModule("data/German-Credit.csv", n_splits=5, random_state=42)
perturbation = Perturbation(dm)

# Get raw data for fold 0
train_raw, test_raw = perturbation.get_data(fold=0, raw_data=True)

# Get preprocessed data for fold 0  
train_processed, test_processed = perturbation.get_data(fold=0, raw_data=False)

# Apply data perturbations
train_reduced = perturbation.perturb_data(train_raw, 'remove_minor', fraction=0.1)
train_augmented = perturbation.perturb_data(train_raw, 'add_major', fraction=0.5)

# Apply model perturbations
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model_small = perturbation.perturb_model(model, 'n_estimators', n_estimators=50)
model_shallow = perturbation.perturb_model(model, 'max_depth', max_depth=3)

# Get metadata for counterfactuals
metadata = perturbation.get_metadata()
print(f"Label column: {metadata['label_column']}")
print(f"Feature types: {metadata['feature_types']}")
print(f"Feature actions: {metadata['feature_actions']}")
```

## How K-fold CV Works

The framework applies **k-fold cross validation on the entire dataset** (excluding metadata rows):

- **Fold 0**: Train on 80% of data, Test on remaining 20%
- **Fold 1**: Train on different 80% of data, Test on different 20%  
- **Fold 2**: Train on different 80% of data, Test on different 20%
- etc.

Each fold gives you a **completely different train/test split** from the total dataset.

## API Reference

### DataModule

```python
DataModule(csv_path, n_splits=5, random_state=42)
```

**Methods:**
- `get_data(fold, raw_data=True)` → Returns (train_df, test_df)
- `get_metadata()` → Returns metadata dict

### Perturbation

```python
Perturbation(data_module)
```

**Methods:**
- `get_data(fold, raw_data=True)` → Returns (train_df, test_df)
- `perturb_data(data, type, **kwargs)` → Returns perturbed DataFrame
- `perturb_model(model, type, **kwargs)` → Returns perturbed model
- `get_metadata()` → Returns metadata dict

### Data Perturbation Types

- `'add_minor'`: Add small amount of data (default: 10%)
- `'add_major'`: Add large amount of data (default: 50%)
- `'remove_minor'`: Remove small amount of data (default: 10%)
- `'remove_major'`: Remove large amount of data (default: 50%)

**Parameters:**
- `fraction=0.1`: Fraction of data to add/remove

### Model Perturbation Types

- `'n_estimators'`: Change number of estimators
- `'max_depth'`: Change maximum depth

**Parameters:**
- `n_estimators=50`: New number of estimators
- `max_depth=3`: New maximum depth

## Supported Datasets

The framework works with CSV files containing:
- **Row 1**: Feature types (`N`=numerical, `D`=discrete, `B`=binary, `C`=categorical)
- **Row 2**: Feature actions (`FREE`, `INC`, `FIXED`, `PREDICT`)
- **Row 3+**: Actual data

**Included datasets:**
- German Credit (1,000 samples, 10 features)
- HELOC (10,459 samples, 24 features)
- ProPublica (6,172 samples, 12 features)
- Spambase (4,601 samples, 58 features)

## Example Workflow

```python
# 1. Load and setup
dm = DataModule("data/German-Credit.csv")
perturbation = Perturbation(dm)

# 2. Test across folds
for fold in range(5):
    # Get data for this fold
    train, test = perturbation.get_data(fold, raw_data=False)
    
    # Train baseline model
    model = RandomForestClassifier()
    model.fit(train.drop('Class', axis=1), train['Class'])
    
    # Test data robustness
    train_reduced = perturbation.perturb_data(train, 'remove_minor', fraction=0.1)
    # Train and evaluate on perturbed data...
    
    # Test model robustness  
    model_small = perturbation.perturb_model(model, 'n_estimators', n_estimators=50)
    # Evaluate perturbed model...
```

## Requirements

- pandas
- numpy
- scikit-learn
- xgboost (optional)
- lightgbm (optional)

Run `python usage_example.py` to see a complete demonstration. 
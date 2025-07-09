# DiCE Counterfactual Generation for Robustness Framework

This module provides comprehensive counterfactual generation using DiCE ML specifically for **Fold 0 (Bin 0)** analysis on the **Spambase dataset**. It generates counterfactuals to flip class labels and calculates detailed quality metrics.

## 🎯 Overview

The `dice_counterfactual_generation.py` script:
- Loads the Spambase dataset and focuses on fold 0 
- Trains a Random Forest baseline model
- Generates counterfactuals using DiCE ML random method
- Calculates comprehensive quality metrics:
  - **Validity**: Percentage of counterfactuals that actually flip class predictions
  - **L2 Distance**: Average Euclidean distance between originals and counterfactuals
  - **LOF Score**: Local Outlier Factor measuring how outlier-like counterfactuals are
  - **Sparsity**: Average number of features changed per counterfactual

## 📦 Installation

### Step 1: Check Dependencies
```bash
python check_dice_installation.py
```

### Step 2: Install DiCE ML (if needed)
```bash
# Option 1: Direct installation
pip install dice-ml

# Option 2: From requirements file
pip install -r requirements_dice.txt

# Option 3: With specific version
pip install dice-ml>=0.9
```

### Step 3: Verify Installation
```bash
python check_dice_installation.py
```

## 🚀 Usage

### Basic Execution
```bash
python dice_counterfactual_generation.py
```

### What the Script Does

1. **📊 Data Loading**
   - Loads Spambase.csv dataset
   - Extracts fold 0 (train/test split)
   - Preprocesses data using the robustness framework

2. **🤖 Model Training**
   - Trains Random Forest classifier (100 trees)
   - Reports baseline performance on train/test sets

3. **🔄 Counterfactual Generation**
   - Uses DiCE ML with random method
   - Attempts to generate counterfactuals for all test samples
   - Reports success rate and failure handling

4. **📈 Quality Analysis**
   - **Validity**: Measures actual class flipping effectiveness
   - **Distance**: Calculates similarity to original samples
   - **LOF**: Evaluates how normal/outlier-like CFs are
   - **Sparsity**: Counts average features changed

5. **📝 Comprehensive Logging**
   - All output logged to timestamped file: `dice_counterfactual_analysis_YYYYMMDD_HHMMSS.log`
   - Real-time console output for monitoring

## 📊 Output Metrics

### Success Metrics
- **Success Rate**: Percentage of test samples that got valid counterfactuals
- **Generation Rate**: Overall CF generation statistics

### Quality Metrics
- **Validity Score**: 0.0-1.0 (higher = better class flipping)
- **L2 Distance**: Lower values = more similar to originals
- **LOF Score**: Closer to -1.0 = more normal (less outlier-like)  
- **Sparsity**: Lower values = fewer features changed

## 📁 File Structure

```
CFRobustness-of-Trees-main/
├── dice_counterfactual_generation.py    # Main script
├── check_dice_installation.py           # Dependency checker
├── requirements_dice.txt                # DiCE ML requirements
├── DICE_README.md                      # This documentation
├── data/
│   └── Spambase.csv                    # Target dataset
└── modules/
    ├── data_module.py                  # Data handling
    └── perturb.py                      # Perturbation framework
```

## 🔧 Technical Details

### DiCE Configuration
- **Method**: Random counterfactual generation
- **Backend**: sklearn
- **CFs per sample**: 2 (uses first successful one)
- **Desired class**: "opposite" (automatic class flipping)

### Data Handling
- **Dataset**: Spambase (spam classification)
- **Fold**: 0 (first fold of 5-fold CV)
- **Preprocessing**: Uses robustness framework's preprocessing pipeline
- **Features**: All features treated as continuous for DiCE

### Model Configuration
- **Algorithm**: Random Forest
- **Trees**: 100
- **Random State**: 42 (reproducible results)

## 📈 Expected Results

### Typical Output
```
📈 COMPREHENSIVE COUNTERFACTUAL ANALYSIS SUMMARY
================================================================================

🎯 DATASET: Spambase (Fold 0)
  • Training samples: 2763
  • Test samples: 921
  • Features: 57
  • Baseline test accuracy: 0.9250

🔧 COUNTERFACTUAL GENERATION:
  • Method: Random
  • Success rate: 85.2%
  • Successful CFs: 785/921

📊 QUALITY METRICS:
  • Validity: 0.9100 (91.0%)
  • Average L2 distance: 2.4500
  • Average LOF score: -1.1200
  • Average sparsity: 8.50 features
```

### Interpretation Guide
- **High Validity (>0.8)**: Good counterfactuals that flip classes
- **Low L2 Distance (<5.0)**: Similar to original samples
- **LOF near -1.0**: Normal counterfactuals (not outliers)
- **Low Sparsity (<20% features)**: Minimal changes needed

## 🚨 Error Handling

The script includes comprehensive error handling for:
- Missing DiCE ML installation
- Dataset loading issues
- Model training failures
- Counterfactual generation failures
- Metric calculation errors

If counterfactual generation fails for specific samples, they are marked as failed and excluded from quality metrics calculation.

## 🔄 Integration with Robustness Framework

This script integrates seamlessly with the existing robustness framework:
- Uses same `DataModule` for consistent data handling
- Uses same preprocessing pipeline
- Focuses on fold 0 to complement perturbation analysis
- Provides logged output compatible with framework standards

## 📊 Research Applications

This tool is designed for:
- **Robustness Analysis**: Understanding model behavior through counterfactuals
- **Fairness Research**: Analyzing biased decision boundaries
- **Explainability Studies**: Providing feature importance through counterfactuals
- **Model Validation**: Testing model reliability through adversarial examples

## 🔬 Advanced Usage

### Custom Parameters
To modify the script for different configurations:
- Change `method='random'` to other DiCE methods (genetic, kdtree)
- Adjust `total_cfs=2` for more counterfactual candidates
- Modify model parameters in RandomForestClassifier
- Add different distance metrics (L1, Mahalanobis)

### Different Datasets
To use with other datasets:
- Change `"data/Spambase.csv"` to your dataset path
- Ensure dataset is compatible with the robustness framework
- Adjust continuous/categorical feature specifications for DiCE

## 📞 Support

If you encounter issues:
1. Run `python check_dice_installation.py` to verify setup
2. Check the generated log file for detailed error messages
3. Ensure all dependencies are correctly installed
4. Verify the Spambase.csv dataset is in the data/ directory

## 🎓 Citation

This implementation is based on:
- DiCE ML framework for counterfactual generation
- Robustness testing framework for tree-based models
- Standard machine learning quality metrics for counterfactual evaluation 
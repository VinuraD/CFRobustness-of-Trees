#!/usr/bin/env python3
"""
Update CF analysis files to include standardized baseline logging format
"""

import os
import re

def update_baseline_logging(file_path):
    """Add standardized baseline logging format to a CF analysis file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Check if already has the standardized logging
    if 'Bin 0: Remove 0%' in content and 'validity:' in content and 'L2:' in content and 'L0:' in content and 'LOF:' in content:
        print(f"  Already has standardized logging: {file_path}")
        return False
    
    # Look for the pattern where baseline metrics are logged
    patterns_to_find = [
        # Pattern 1: Standard baseline logging
        (r'(\s+log_print\(f"  Baseline LOF score: \{baseline_metrics\[\'lof_score\'\]:.4f\}"\))',
         r'\1\n            \n            # Log baseline metrics in standardized format for visualization parsing\n            log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {test_acc:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")'),
        
        # Pattern 2: Alternative baseline logging
        (r'(\s+log_print\(f"  Baseline LOF score: \{baseline_metrics\[\'lof_score\'\]:.4f\}"\))',
         r'\1\n            \n            # Log baseline metrics in standardized format for visualization parsing\n            log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {test_accuracy:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")'),
        
        # Pattern 3: For files that use different variable names
        (r'(\s+log_print\(f"  Baseline LOF score: \{baseline_metrics\[\'lof_score\'\]:.4f\}"\))',
         r'\1\n            \n            # Log baseline metrics in standardized format for visualization parsing\n            log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {model_accuracy:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")')
    ]
    
    modified = False
    for pattern, replacement in patterns_to_find:
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            modified = True
            break
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Updated: {file_path}")
        return True
    else:
        print(f"  No baseline logging pattern found: {file_path}")
        return False

def main():
    # Define all the CF analysis files that need updating
    files_to_update = [
        'DiCE/cf_robustness_analysis_v3.py',
        'DiCE/cf_robustness_analysis_v4_heloc.py', 
        'DiCE/cf_robustness_analysis_v5_compas.py',
        'CEML/cf_robustness_analysis_ceml_v3.py',
        'CEML/cf_robustness_analysis_ceml_v4_heloc.py',
        'CEML/cf_robustness_analysis_ceml_v5_compas.py',
        'feature_tweak/cf_robustness_analysis_featuretweak_v3.py',
        'feature_tweak/cf_robustness_analysis_featuretweak_v4_heloc.py',
        'feature_tweak/cf_robustness_analysis_featuretweak_v5_compas.py',
        'cfxplorer/cf_robustness_analysis_cfxplorer_v2.py',
        'cfxplorer/cf_robustness_analysis_cfxplorer_v3.py',
        'cfxplorer/cf_robustness_analysis_cfxplorer_v4_heloc.py',
        'cfxplorer/cf_robustness_analysis_cfxplorer_v5_compas.py',
        'NICE/cf_robustness_analysis_nice_v2.py',
        'NICE/cf_robustness_analysis_nice_v3.py',
        'NICE/cf_robustness_analysis_nice_v4_heloc.py',
        'NICE/cf_robustness_analysis_nice_v5_compas.py'
    ]
    
    # Base directory
    base_dir = '../'
    
    print("Checking and updating CF analysis files for standardized baseline logging...")
    
    updated_count = 0
    for file_path in files_to_update:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path}")
            if update_baseline_logging(full_path):
                updated_count += 1
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print(f"\nCompleted! Updated {updated_count} files with standardized baseline logging.")

if __name__ == "__main__":
    main()

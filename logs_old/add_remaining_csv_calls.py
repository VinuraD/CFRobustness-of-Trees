#!/usr/bin/env python3
"""
Add CSV saving calls to all remaining CF analysis files systematically
"""

import os
import re

def add_csv_call_to_file(file_path, cf_method, dataset_name):
    """Add CSV saving call to a specific CF analysis file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Check if already has the CSV saving call
    if f'save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)' in content:
        print(f"  Already has CSV saving call: {file_path}")
        return False
    
    # Fix corrupted baseline logging format first
    corrupted_patterns = [
        r"log_print\(f\"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\\?'validity'\\?\]:.4f\}, accuracy: \{test_acc:.4f\}, L2: \{baseline_metrics\[\\?'l2_distance'\\?\]:.4f\}, L0: \{baseline_metrics\[\\?'l0_distance'\\?\]:.2f\}, LOF: \{baseline_metrics\[\\?'lof_score'\\?\]:.4f\}\"\)",
        r"log_print\(f\"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\\'validity\\'\\]:.4f\}, accuracy: \{test_acc:.4f\}, L2: \{baseline_metrics\[\\'l2_distance\\'\\]:.4f\}, L0: \{baseline_metrics\[\\'l0_distance\\'\\]:.2f\}, LOF: \{baseline_metrics\[\\'lof_score\\'\\]:.4f\}\"\)"
    ]
    
    correct_pattern = 'log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {test_acc:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")'
    
    for corrupted_pattern in corrupted_patterns:
        if re.search(corrupted_pattern, content):
            content = re.sub(corrupted_pattern, correct_pattern, content)
            print(f"  Fixed baseline logging format")
    
    # Look for the success_rate append pattern and add CSV call before it
    pattern = r'(\s+)(all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))'
    replacement = rf'\1# Save counterfactuals to CSV\n\1save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\1\n\1\2'
    
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print(f"  Added CSV saving call: {file_path}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print(f"  Pattern not found: {file_path}")
        return False

def main():
    # Define all CF analysis files that need CSV saving calls
    files_info = [
        # CEML
        ('CEML/cf_robustness_analysis_ceml_v2.py', 'CEML', 'Spambase'),
        ('CEML/cf_robustness_analysis_ceml_v3.py', 'CEML', 'German-Credit'),
        ('CEML/cf_robustness_analysis_ceml_v4_heloc.py', 'CEML', 'HELOC'),
        ('CEML/cf_robustness_analysis_ceml_v5_compas.py', 'CEML', 'COMPAS'),
        
        # Feature Tweak
        ('feature_tweak/cf_robustness_analysis_featuretweak_v2.py', 'FeatureTweak', 'Spambase'),
        ('feature_tweak/cf_robustness_analysis_featuretweak_v3.py', 'FeatureTweak', 'German-Credit'),
        ('feature_tweak/cf_robustness_analysis_featuretweak_v4_heloc.py', 'FeatureTweak', 'HELOC'),
        ('feature_tweak/cf_robustness_analysis_featuretweak_v5_compas.py', 'FeatureTweak', 'COMPAS'),
        
        # CFXplorer
        ('cfxplorer/cf_robustness_analysis_cfxplorer_v2.py', 'CFXplorer', 'Spambase'),
        ('cfxplorer/cf_robustness_analysis_cfxplorer_v3.py', 'CFXplorer', 'German-Credit'),
        ('cfxplorer/cf_robustness_analysis_cfxplorer_v4_heloc.py', 'CFXplorer', 'HELOC'),
        ('cfxplorer/cf_robustness_analysis_cfxplorer_v5_compas.py', 'CFXplorer', 'COMPAS'),
        
        # NICE
        ('NICE/cf_robustness_analysis_nice_v2.py', 'NICE', 'Spambase'),
        ('NICE/cf_robustness_analysis_nice_v3.py', 'NICE', 'German-Credit'),
        ('NICE/cf_robustness_analysis_nice_v4_heloc.py', 'NICE', 'HELOC'),
        ('NICE/cf_robustness_analysis_nice_v5_compas.py', 'NICE', 'COMPAS'),
    ]
    
    # Base directory
    base_dir = '../'
    
    print("Adding CSV saving calls to remaining CF analysis files...")
    
    added_count = 0
    
    for file_path, cf_method, dataset_name in files_info:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path} ({cf_method}, {dataset_name})")
            
            if add_csv_call_to_file(full_path, cf_method, dataset_name):
                added_count += 1
                
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print(f"\nCompleted!")
    print(f"  Added CSV saving calls to {added_count} files")

if __name__ == "__main__":
    main()

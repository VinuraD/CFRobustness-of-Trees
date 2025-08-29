#!/usr/bin/env python3
"""
Fix CSV saving calls and baseline logging format
"""

import os
import re

def fix_file(file_path, cf_method, dataset_name):
    """Fix CSV saving call and baseline logging format in a CF analysis file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    changed = False
    
    # 1. Fix baseline logging format if corrupted
    corrupted_pattern = r"log_print\(f\"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\\?'validity'\\?\]:.4f\}, accuracy: \{test_acc:.4f\}, L2: \{baseline_metrics\[\\?'l2_distance'\\?\]:.4f\}, L0: \{baseline_metrics\[\\?'l0_distance'\\?\]:.2f\}, LOF: \{baseline_metrics\[\\?'lof_score'\\?\]:.4f\}\"\)"
    correct_pattern = 'log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {test_acc:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")'
    
    if re.search(corrupted_pattern, content):
        content = re.sub(corrupted_pattern, correct_pattern, content)
        changed = True
        print(f"  Fixed baseline logging format in: {file_path}")
    
    # 2. Add CSV saving call if not present
    if 'save_counterfactuals_to_csv(cf_list,' not in content:
        # Find the pattern after cf_list generation
        pattern = r'(all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))'
        replacement = f'# Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n            \n            \\1'
        
        if re.search(pattern, content):
            content = re.sub(pattern, replacement, content)
            changed = True
            print(f"  Added CSV saving call to: {file_path}")
    
    if changed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    else:
        print(f"  No changes needed: {file_path}")
        return False

def main():
    # Define all CF analysis files with their method names and datasets
    files_info = [
        # DiCE
        ('DiCE/cf_robustness_analysis_v3.py', 'DiCE', 'German-Credit'),
        ('DiCE/cf_robustness_analysis_v4_heloc.py', 'DiCE', 'HELOC'),
        ('DiCE/cf_robustness_analysis_v5_compas.py', 'DiCE', 'COMPAS'),
        
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
    
    print("Fixing CSV saving calls and baseline logging format...")
    
    fixed_count = 0
    
    for file_path, cf_method, dataset_name in files_info:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path} ({cf_method}, {dataset_name})")
            
            if fix_file(full_path, cf_method, dataset_name):
                fixed_count += 1
                
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print(f"\nCompleted!")
    print(f"  Fixed {fixed_count} files")

if __name__ == "__main__":
    main()

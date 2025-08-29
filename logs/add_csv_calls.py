#!/usr/bin/env python3
"""
Add CSV saving calls to all CF analysis files
"""

import os
import re

def add_csv_saving_call_manual(file_path, cf_method, dataset_name):
    """Add the CSV saving call after counterfactual generation"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Check if already has the CSV saving call
    if 'save_counterfactuals_to_csv(' in content:
        print(f"  Already has CSV saving call: {file_path}")
        return False
    
    # Find different patterns for counterfactual generation based on method
    patterns_to_find = []
    
    if cf_method == "DiCE":
        patterns_to_find = [
            (r'(\s+cf_list, success_rate = generate_counterfactuals\(\s+test_data_for_cf,\s+baseline_model,\s+dice_data,\s+method=\'random\',\s+total_cfs=2\s+\))\s+(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\2'),
        ]
    elif cf_method == "CEML":
        patterns_to_find = [
            (r'(\s+cf_list, success_rate = generate_counterfactuals_ceml\([^)]+\))\s+(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\2'),
        ]
    elif cf_method == "FeatureTweak":
        patterns_to_find = [
            (r'(\s+cf_list, success_rate = generate_counterfactuals_featuretweak\([^)]+\))\s+(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\2'),
        ]
    elif cf_method == "CFXplorer":
        patterns_to_find = [
            (r'(\s+cf_list, success_rate = generate_counterfactuals_cfxplorer\([^)]+\))\s+(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\2'),
        ]
    elif cf_method == "NICE":
        patterns_to_find = [
            (r'(\s+cf_list, success_rate = generate_counterfactuals_nice\([^)]+\))\s+(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n\2'),
        ]
    
    # Try general patterns if specific ones don't work
    if not patterns_to_find:
        patterns_to_find = [
            (r'(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
             rf'            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)\n            \n\1'),
        ]
    
    modified = False
    for pattern, replacement in patterns_to_find:
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.DOTALL)
            modified = True
            break
    
    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Added CSV saving call to: {file_path}")
        return True
    else:
        print(f"  No suitable pattern found for CSV saving call: {file_path}")
        # Let's try the manual approach - insert after success_rate line
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if "all_fold_results['baseline_success_rate'].append(success_rate)" in line:
                # Insert before this line
                indent = len(line) - len(line.lstrip())
                csv_call = " " * indent + "# Save counterfactuals to CSV"
                csv_call2 = " " * indent + f'save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold)'
                csv_call3 = " " * indent
                lines.insert(i, csv_call3)
                lines.insert(i, csv_call2)
                lines.insert(i, csv_call)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"  Added CSV saving call (manual) to: {file_path}")
                return True
        
        return False

def main():
    # Define all CF analysis files with their method names and datasets
    files_info = [
        # DiCE - already done
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
    
    print("Adding CSV saving calls to CF analysis files...")
    
    call_added_count = 0
    
    for file_path, cf_method, dataset_name in files_info:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path} ({cf_method}, {dataset_name})")
            
            # Add the CSV saving call
            if add_csv_saving_call_manual(full_path, cf_method, dataset_name):
                call_added_count += 1
                
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print(f"\nCompleted!")
    print(f"  Added CSV saving call to {call_added_count} files")

if __name__ == "__main__":
    main()

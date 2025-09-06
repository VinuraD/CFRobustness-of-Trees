#!/usr/bin/env python3
"""
Add CSV saving functionality to all CF analysis files
"""

import os
import re

def add_csv_saving_function(file_path):
    """Add CSV saving function to a CF analysis file"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Check if already has the CSV saving function
    if 'save_counterfactuals_to_csv' in content:
        print(f"  Already has CSV saving function: {file_path}")
        return False
    
    # Define the CSV saving function to add
    csv_function = '''
def save_counterfactuals_to_csv(cf_list, cf_method, dataset_name, fold_idx):
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
        filename = f"{cf_method}__{dataset_name}__fold{fold_idx}.csv"
        filepath = os.path.join(cf_dir, filename)
        
        # Save counterfactuals to CSV
        cf_list.to_csv(filepath, index=False)
        print(f"    Saved counterfactuals to: {filename}")
        
    except Exception as e:
        print(f"    Error saving counterfactuals to CSV: {e}")

'''
    
    # Find where to insert the function (after other helper functions, before main)
    # Look for the pattern right before the main execution
    patterns_to_find = [
        # Pattern 1: Before main execution
        r'(def main\(\):)',
        # Pattern 2: Before if __name__ == "__main__"
        r'(if __name__ == "__main__":)',
        # Pattern 3: Before log_print function if it exists
        r'(# Set up logging\ndef setup_logging\(\):)',
    ]
    
    inserted = False
    for pattern in patterns_to_find:
        if re.search(pattern, content):
            content = re.sub(pattern, csv_function + r'\1', content)
            inserted = True
            break
    
    if not inserted:
        # Fallback: insert before the last function or class
        lines = content.split('\n')
        insert_pos = -1
        for i, line in enumerate(lines):
            if line.startswith('def ') or line.startswith('class '):
                insert_pos = i
        
        if insert_pos > 0:
            lines.insert(insert_pos, csv_function)
            content = '\n'.join(lines)
            inserted = True
    
    if inserted:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Added CSV saving function to: {file_path}")
        return True
    else:
        print(f"  Could not find insertion point: {file_path}")
        return False

def add_csv_saving_call(file_path, cf_method, dataset_name):
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
    
    # Find the pattern after counterfactual generation
    patterns_to_find = [
        # Pattern 1: After cf_list generation and success_rate calculation
        (r'(\s+cf_list, success_rate = generate_counterfactuals\([^)]+\))',
         rf'\1\n            \n            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold_idx)'),
        
        # Pattern 2: Alternative pattern for different CF methods
        (r'(\s+all_fold_results\[\'baseline_success_rate\'\]\.append\(success_rate\))',
         rf'            # Save counterfactuals to CSV\n            save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}", fold_idx)\n            \n\1'),
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
        print(f"  Added CSV saving call to: {file_path}")
        return True
    else:
        print(f"  No suitable pattern found for CSV saving call: {file_path}")
        return False

def main():
    # Define all CF analysis files with their method names and datasets
    files_info = [
        # DiCE
        ('DiCE/cf_robustness_analysis_v2.py', 'DiCE', 'Spambase'),
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
    
    print("Adding CSV saving functionality to CF analysis files...")
    
    function_added_count = 0
    call_added_count = 0
    
    for file_path, cf_method, dataset_name in files_info:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path} ({cf_method}, {dataset_name})")
            
            # Add the CSV saving function
            if add_csv_saving_function(full_path):
                function_added_count += 1
            
            # Add the CSV saving call
            if add_csv_saving_call(full_path, cf_method, dataset_name):
                call_added_count += 1
                
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print(f"\nCompleted!")
    print(f"  Added CSV saving function to {function_added_count} files")
    print(f"  Added CSV saving call to {call_added_count} files")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Verify that all CF analysis files have CSV saving functionality
"""

import os
import re

def check_file(file_path, cf_method, dataset_name):
    """Check if a CF analysis file has both CSV function and call"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Check for CSV saving function
    has_function = 'def save_counterfactuals_to_csv(' in content
    
    # Check for CSV saving call - multiple patterns
    call_patterns = [
        f'save_counterfactuals_to_csv(cf_list, "{cf_method}", "{dataset_name}"',
        f'save_counterfactuals_to_csv(baseline_cf_list, "{cf_method}", "{dataset_name}"'
    ]
    
    has_call = any(pattern in content for pattern in call_patterns)
    
    return has_function, has_call

def main():
    # Define all CF analysis files
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
    
    print("Verifying CSV saving functionality in all CF analysis files...")
    print("=" * 80)
    
    total_files = 0
    complete_files = 0
    
    for file_path, cf_method, dataset_name in files_info:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            total_files += 1
            has_function, has_call = check_file(full_path, cf_method, dataset_name)
            
            status = ""
            if has_function and has_call:
                status = "✅ COMPLETE"
                complete_files += 1
            elif has_function:
                status = "⚠️  FUNCTION ONLY"
            elif has_call:
                status = "⚠️  CALL ONLY"
            else:
                status = "❌ MISSING"
                
            print(f"{status:15} {file_path:60} ({cf_method}, {dataset_name})")
        else:
            print(f"❌ NOT FOUND   {file_path:60} ({cf_method}, {dataset_name})")
    
    print("=" * 80)
    print(f"Summary: {complete_files}/{total_files} files have complete CSV saving functionality")
    
    if complete_files == total_files:
        print("🎉 All CF analysis files are ready for CSV saving!")
    else:
        print("⚠️  Some files are missing CSV saving functionality")

if __name__ == "__main__":
    main()

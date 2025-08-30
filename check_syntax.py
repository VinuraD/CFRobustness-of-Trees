#!/usr/bin/env python3
"""
Check all CF analysis files for syntax errors
"""

import py_compile
import os
import sys

def check_syntax(file_path):
    """Check if a Python file has syntax errors"""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, None
    except py_compile.PyCompileError as e:
        return False, str(e)

def main():
    # Define all CF analysis files
    files_to_check = [
        'DiCE/cf_robustness_analysis_v2.py',
        'DiCE/cf_robustness_analysis_v3.py', 
        'DiCE/cf_robustness_analysis_v4_heloc.py',
        'DiCE/cf_robustness_analysis_v5_compas.py',
        'CEML/cf_robustness_analysis_ceml_v2.py',
        'CEML/cf_robustness_analysis_ceml_v3.py',
        'CEML/cf_robustness_analysis_ceml_v4_heloc.py',
        'CEML/cf_robustness_analysis_ceml_v5_compas.py',
        'feature_tweak/cf_robustness_analysis_featuretweak_v2.py',
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
        'NICE/cf_robustness_analysis_nice_v5_compas.py',
    ]
    
    print("Checking all CF analysis files for syntax errors...")
    print("=" * 80)
    
    error_files = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            is_valid, error = check_syntax(file_path)
            if is_valid:
                print(f"✅ OK      {file_path}")
            else:
                print(f"❌ ERROR   {file_path}")
                print(f"   Error: {error}")
                error_files.append(file_path)
        else:
            print(f"❓ MISSING {file_path}")
    
    print("=" * 80)
    if error_files:
        print(f"Found syntax errors in {len(error_files)} files:")
        for file_path in error_files:
            print(f"  - {file_path}")
    else:
        print("✅ All files have valid syntax!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Script to update all algorithm files to include L0, L2, and LOF in logging output
"""

import os
import re

def update_file_logging(file_path):
    """Update a single file to include L0, L2, LOF in logging"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: Data perturbation logging (Bin {bin_num}: Remove/Use X% -> validity: X, accuracy: X, L2: X, L0: X)
        # Add LOF if not already present
        pattern1 = r'(log_print\(f"    Bin \{[^}]+\}: (?:Remove|Use) [^"]*validity: \{[^}]+\:.4f\}, accuracy: \{[^}]+\:.4f\}, L2: \{[^}]+\:.4f\}, L0: \{[^}]+\:.2f\})"\)\)'
        replacement1 = r'\1, LOF: {cf_metrics[\'lof_score\']:.4f}"\)'
        
        # Also handle the case where it might be 'metrics' instead of 'cf_metrics'
        pattern1b = r'(log_print\(f"    Bin \{[^}]+\}: (?:Remove|Use) [^"]*validity: \{metrics\[\'validity\'\]\:.4f\}, accuracy: \{[^}]+\:.4f\})"\)\)'
        replacement1b = r'\1, L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}"\)'
        
        # Pattern 2: Model perturbation logging (model_type (params): validity: X, accuracy: X, L2: X, L0: X)
        # Add LOF if not already present
        pattern2 = r'(log_print\(f"  \{model_type\} \([^)]*\): validity: \{[^}]+\:.4f\}, accuracy: \{[^}]+\:.4f\}, L2: \{[^}]+\:.4f\}, L0: \{[^}]+\:.2f\})"\)\)'
        replacement2 = r'\1, LOF: {cf_metrics[\'lof_score\']:.4f}"\)'
        
        # Also handle the case where it might be 'metrics' instead of 'cf_metrics'
        pattern2b = r'(log_print\(f"[^"]*\{model_type\}[^"]*validity [^"]*\{metrics\[\'validity\'\]\:.4f\}, accuracy [^"]*\{[^}]+\:.4f\})"\)\)'
        replacement2b = r'\1, L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}"\)'
        
        # Apply replacements
        content = re.sub(pattern1, replacement1, content)
        content = re.sub(pattern1b, replacement1b, content)
        content = re.sub(pattern2, replacement2, content)
        content = re.sub(pattern2b, replacement2b, content)
        
        # Check if changes were made
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Updated: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Main function to update all algorithm files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Algorithm directories and their files
    algorithms = {
        'DiCE': ['cf_robustness_analysis_v4_heloc.py', 'cf_robustness_analysis_v5_compas.py'],
        'CEML': ['cf_robustness_analysis_ceml_v3.py', 'cf_robustness_analysis_ceml_v4_heloc.py', 'cf_robustness_analysis_ceml_v5_compas.py'],
        'feature_tweak': ['cf_robustness_analysis_featuretweak_v3.py', 'cf_robustness_analysis_featuretweak_v4_heloc.py', 'cf_robustness_analysis_featuretweak_v5_compas.py'],
        'NICE': ['cf_robustness_analysis_nice_v3.py', 'cf_robustness_analysis_nice_v4_heloc.py', 'cf_robustness_analysis_nice_v5_compas.py'],
        'cfxplorer': ['cf_robustness_analysis_cfxplorer_v3.py', 'cf_robustness_analysis_cfxplorer_v4_heloc.py', 'cf_robustness_analysis_cfxplorer_v5_compas.py']
    }
    
    updated_files = []
    failed_files = []
    
    for alg_name, files in algorithms.items():
        alg_dir = os.path.join(base_dir, alg_name)
        if not os.path.exists(alg_dir):
            print(f"Warning: Directory {alg_dir} does not exist")
            continue
            
        for file_name in files:
            file_path = os.path.join(alg_dir, file_name)
            if os.path.exists(file_path):
                if update_file_logging(file_path):
                    updated_files.append(file_path)
            else:
                print(f"Warning: File {file_path} does not exist")
                failed_files.append(file_path)
    
    print(f"\nSummary:")
    print(f"Updated files: {len(updated_files)}")
    print(f"Failed/missing files: {len(failed_files)}")
    
    if updated_files:
        print(f"\nUpdated files:")
        for f in updated_files:
            print(f"  - {f}")
    
    if failed_files:
        print(f"\nFailed/missing files:")
        for f in failed_files:
            print(f"  - {f}")

if __name__ == "__main__":
    main()

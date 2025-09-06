#!/usr/bin/env python3
"""
Fix f-string syntax errors in baseline logging across all CF analysis files
"""

import os
import re

def fix_fstring_syntax(file_path):
    """Fix f-string syntax error in baseline logging"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin1') as f:
            content = f.read()
    
    # Find the problematic f-string pattern
    old_pattern = r'log_print\(f"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\\?\'validity\'\\?\]:.4f\}, accuracy: \{([^}]+):.4f\}, L2: \{baseline_metrics\[\\?\'l2_distance\'\\?\]:.4f\}, L0: \{baseline_metrics\[\\?\'l0_distance\'\\?\]:.2f\}, LOF: \{baseline_metrics\[\\?\'lof_score\'\\?\]:.4f\}"\)'
    
    # Search for the pattern and extract the accuracy variable
    match = re.search(old_pattern, content)
    
    if match:
        accuracy_var = match.group(1)
        
        # Replace with proper f-string syntax
        new_line = f'log_print(f"    Bin 0: Remove 0% -> validity: {{baseline_metrics[\'validity\']:.4f}}, accuracy: {{{accuracy_var}:.4f}}, L2: {{baseline_metrics[\'l2_distance\']:.4f}}, L0: {{baseline_metrics[\'l0_distance\']:.2f}}, LOF: {{baseline_metrics[\'lof_score\']:.4f}}")'
        
        content = re.sub(old_pattern, new_line, content)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  Fixed f-string syntax in: {file_path}")
        return True
    else:
        # Try alternative patterns for files that might have different issues
        patterns_to_fix = [
            # Pattern with escaped quotes
            (r'log_print\(f"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\\\'validity\\\'.*?\}"\)',
             lambda m: fix_baseline_log_line(m, file_path)),
            
            # Pattern with different accuracy variables  
            (r'log_print\(f"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\'validity\'\]:.4f\}, accuracy: \{test_acc:.4f\}.*?"\)',
             lambda m: fix_baseline_log_line_test_acc(m, file_path)),
        ]
        
        for pattern, fix_func in patterns_to_fix:
            if re.search(pattern, content):
                content = fix_func(content)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"  Fixed f-string syntax (alt pattern) in: {file_path}")
                return True
        
        print(f"  No f-string issues found: {file_path}")
        return False

def fix_baseline_log_line(content, file_path):
    """Fix baseline logging line with proper syntax"""
    
    # Determine the correct accuracy variable based on file content
    if 'test_accuracy' in content:
        acc_var = 'test_accuracy'
    elif 'test_acc' in content:
        acc_var = 'test_acc'
    else:
        acc_var = 'test_acc'  # default
    
    # Replace with correct f-string syntax
    new_line = f'log_print(f"    Bin 0: Remove 0% -> validity: {{baseline_metrics[\'validity\']:.4f}}, accuracy: {{{acc_var}:.4f}}, L2: {{baseline_metrics[\'l2_distance\']:.4f}}, L0: {{baseline_metrics[\'l0_distance\']:.2f}}, LOF: {{baseline_metrics[\'lof_score\']:.4f}}")'
    
    # Replace the problematic line
    pattern = r'log_print\(f"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[.*?\}"\)'
    content = re.sub(pattern, new_line, content, flags=re.DOTALL)
    
    return content

def fix_baseline_log_line_test_acc(content, file_path):
    """Fix baseline logging line specifically for test_acc variable"""
    
    new_line = 'log_print(f"    Bin 0: Remove 0% -> validity: {baseline_metrics[\'validity\']:.4f}, accuracy: {test_acc:.4f}, L2: {baseline_metrics[\'l2_distance\']:.4f}, L0: {baseline_metrics[\'l0_distance\']:.2f}, LOF: {baseline_metrics[\'lof_score\']:.4f}")'
    
    # Replace the problematic line
    pattern = r'log_print\(f"    Bin 0: Remove 0% -> validity: \{baseline_metrics\[\'validity\'\]:.4f\}, accuracy: \{test_acc:.4f\}.*?"\)'
    content = re.sub(pattern, new_line, content, flags=re.DOTALL)
    
    return content

def main():
    # Define all CF analysis files
    files_to_fix = [
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
    
    # Base directory
    base_dir = '../'
    
    print("Fixing f-string syntax errors in baseline logging...")
    print("=" * 60)
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"\nProcessing: {file_path}")
            if fix_fstring_syntax(full_path):
                fixed_count += 1
        else:
            print(f"\nFile NOT found: {file_path}")
    
    print("\n" + "=" * 60)
    print(f"Fixed f-string syntax in {fixed_count} files")

if __name__ == "__main__":
    main()

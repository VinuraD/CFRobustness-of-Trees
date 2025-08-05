#!/usr/bin/env python3
"""
Batch update script to add LOF logging to all remaining algorithm files
"""

import os
import re

# Define files and their expected patterns
files_to_update = [
    # CEML files
    'CEML/cf_robustness_analysis_ceml_v3.py',
    'CEML/cf_robustness_analysis_ceml_v4_heloc.py', 
    'CEML/cf_robustness_analysis_ceml_v5_compas.py',
    
    # feature_tweak files
    'feature_tweak/cf_robustness_analysis_featuretweak_v3.py',
    'feature_tweak/cf_robustness_analysis_featuretweak_v4_heloc.py',
    'feature_tweak/cf_robustness_analysis_featuretweak_v5_compas.py',
    
    # NICE files
    'NICE/cf_robustness_analysis_nice_v3.py',
    'NICE/cf_robustness_analysis_nice_v4_heloc.py',
    'NICE/cf_robustness_analysis_nice_v5_compas.py',
    
    # cfxplorer files  
    'cfxplorer/cf_robustness_analysis_cfxplorer_v3.py',
    'cfxplorer/cf_robustness_analysis_cfxplorer_v4_heloc.py',
    'cfxplorer/cf_robustness_analysis_cfxplorer_v5_compas.py'
]

def update_logging_in_file(file_path):
    """Update logging patterns in a specific file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        
        # Pattern 1: Update data perturbation bin logging - CEML/DiCE/NICE pattern
        # Look for: log_print(f"    Bin {bin_num}: Remove/Use X% -> validity: {cf_metrics['validity']:.4f}, accuracy: {perturbed_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
        pattern1 = r'(log_print\(f"    Bin \{bin_num\}: (?:Remove|Use) [^"]*validity: \{cf_metrics\[\'validity\'\]:.4f\}, accuracy: \{perturbed_test_acc:.4f\}, L2: \{cf_metrics\[\'l2_distance\'\]:.4f\}, L0: \{cf_metrics\[\'l0_distance\'\]:.2f\}"\))'
        replacement1 = r'\1[:-2] + ", LOF: {cf_metrics[\'lof_score\']:.4f}\")'
        if re.search(pattern1, content):
            content = re.sub(pattern1, replacement1, content)
            changes_made = True
        
        # Pattern 2: Update data perturbation bin logging - feature_tweak/cfxplorer pattern  
        # Look for: log_print(f"    Bin {bin_val}: Remove/Use X% -> validity: {metrics['validity']:.4f}, accuracy: {accuracy:.4f}")
        pattern2 = r'(log_print\(f"    Bin \{bin_val\}: (?:Remove|Use) [^"]*validity: \{metrics\[\'validity\'\]:.4f\}, accuracy: \{accuracy:.4f\}"\))'
        replacement2 = r'\1[:-2] + ", L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}\")'
        if re.search(pattern2, content):
            content = re.sub(pattern2, replacement2, content)
            changes_made = True
        
        # Pattern 3: Update model perturbation logging - CEML/DiCE/NICE pattern
        # Look for: log_print(f"  {model_type} (X, Y): validity: {cf_metrics['validity']:.4f}, accuracy: {model_test_acc:.4f}, L2: {cf_metrics['l2_distance']:.4f}, L0: {cf_metrics['l0_distance']:.2f}")
        pattern3 = r'(log_print\(f"  \{model_type\} \([^)]*\): validity: \{cf_metrics\[\'validity\'\]:.4f\}, accuracy: \{model_test_acc:.4f\}, L2: \{cf_metrics\[\'l2_distance\'\]:.4f\}, L0: \{cf_metrics\[\'l0_distance\'\]:.2f\}"\))'
        replacement3 = r'\1[:-2] + ", LOF: {cf_metrics[\'lof_score\']:.4f}\")'
        if re.search(pattern3, content):
            content = re.sub(pattern3, replacement3, content)
            changes_made = True
            
        # Pattern 4: Update model perturbation logging - feature_tweak/cfxplorer pattern
        # Look for: log_print(f"      {model_type} {params}: validity {metrics['validity']:.4f}, accuracy {accuracy:.4f}")
        pattern4 = r'(log_print\(f"[^"]*\{model_type\}[^"]*validity \{metrics\[\'validity\'\]:.4f\}, accuracy \{accuracy:.4f\}"\))'
        replacement4 = r'\1[:-2] + ", L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}\")'
        if re.search(pattern4, content):
            content = re.sub(pattern4, replacement4, content)
            changes_made = True
        
        # Manual replacements for specific patterns that regex might miss
        
        # CEML/DiCE/NICE data perturbation pattern
        if 'cf_metrics[\'l0_distance\']:.2f}")' in content and 'LOF:' not in content:
            content = content.replace(
                'L0: {cf_metrics[\'l0_distance\']:.2f}")',
                'L0: {cf_metrics[\'l0_distance\']:.2f}, LOF: {cf_metrics[\'lof_score\']:.4f}")'
            )
            changes_made = True
        
        # feature_tweak/cfxplorer data perturbation pattern (needs full addition)
        if 'accuracy: {accuracy:.4f}")' in content and 'L2:' not in content and 'bin_val' in content:
            content = content.replace(
                'accuracy: {accuracy:.4f}")',
                'accuracy: {accuracy:.4f}, L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}")'
            )
            changes_made = True
        
        # CEML/DiCE/NICE model perturbation pattern  
        if 'cf_metrics[\'l0_distance\']:.2f}")' in content and 'model_type' in content and 'LOF:' not in content:
            content = content.replace(
                'L0: {cf_metrics[\'l0_distance\']:.2f}")',
                'L0: {cf_metrics[\'l0_distance\']:.2f}, LOF: {cf_metrics[\'lof_score\']:.4f}")'
            )
            changes_made = True
        
        # feature_tweak/cfxplorer model perturbation pattern (needs full addition)
        if 'accuracy {accuracy:.4f}")' in content and 'L2:' not in content and 'model_type' in content:
            content = content.replace(
                'accuracy {accuracy:.4f}")',
                'accuracy {accuracy:.4f}, L2: {metrics[\'l2_distance\']:.4f}, L0: {metrics[\'l0_distance\']:.2f}, LOF: {metrics[\'lof_score\']:.4f}")'
            )
            changes_made = True
        
        # Write back if changes were made
        if changes_made:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def main():
    """Update all remaining files"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    updated_count = 0
    failed_count = 0
    
    for file_path in files_to_update:
        full_path = os.path.join(base_dir, file_path)
        
        if os.path.exists(full_path):
            try:
                if update_logging_in_file(full_path):
                    print(f"✅ Updated: {file_path}")
                    updated_count += 1
                else:
                    print(f"⚠️  No changes needed: {file_path}")
            except Exception as e:
                print(f"❌ Failed: {file_path} - {e}")
                failed_count += 1
        else:
            print(f"❌ File not found: {file_path}")
            failed_count += 1
    
    print(f"\nSummary:")
    print(f"✅ Files updated: {updated_count}")
    print(f"❌ Files failed/missing: {failed_count}")
    print(f"📁 Total files processed: {len(files_to_update)}")

if __name__ == "__main__":
    main()

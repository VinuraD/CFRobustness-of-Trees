#!/usr/bin/env python3
"""
Unicode Character Fixer for CF Robustness Experiments

This script fixes Unicode encoding issues in all experiment files by replacing
Unicode emoji characters with plain text alternatives.
"""

import os
import re
from pathlib import Path

def fix_unicode_in_file(file_path):
    """Fix Unicode characters in a single file"""
    # Unicode character replacements
    replacements = {
        '📝': '[LOG]',
        '📊': '[SUMMARY]', 
        '🔄': '[INSIGHTS]',
        '✅': '[SUCCESS]',
        '❌': '[ERROR]',
        '🚀': '[LAUNCH]',
        '🔍': '[CHECK]',
        '👁️': '[MONITOR]',
        '🎉': '[FINISHED]',
        '⏹️': '[STOPPED]',
        '🛑': '[KILL]',
        '🔪': '[KILLED]',
        '💾': '[SAVED]',
        '📂': '[LOADED]',
        '📋': '[FOUND]',
        '⚠️': '[WARNING]',
        '🎯': '[SUMMARY]',
        '🕒': '[TIME]',  # Clock emoji
        '🕓': '[TIME]',
        '🕔': '[TIME]',
        '🕕': '[TIME]',
        '🕖': '[TIME]',
        '🕗': '[TIME]',
        '🕘': '[TIME]',
        '🕙': '[TIME]',
        '🕚': '[TIME]',
        '🕛': '[TIME]',
        '🕐': '[TIME]',
        '🕑': '[TIME]',
        '🕜': '[TIME]',
        '🕝': '[TIME]',
        '🕞': '[TIME]',
        '🕟': '[TIME]',
        '🕠': '[TIME]',
        '🕡': '[TIME]',
        '🕢': '[TIME]',
        '🕣': '[TIME]',
        '🕤': '[TIME]',
        '🕥': '[TIME]',
        '🕦': '[TIME]',
        '🕧': '[TIME]',
    }
    
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if any Unicode characters are present
        has_unicode = any(char in content for char in replacements.keys())
        
        if has_unicode:
            # Apply replacements
            for unicode_char, replacement in replacements.items():
                content = content.replace(unicode_char, replacement)
            
            # Write back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Fixed Unicode characters in: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix Unicode characters in all experiment files"""
    base_dir = Path(__file__).parent
    
    # Directories containing experiment files
    experiment_dirs = ['DiCE', 'CEML', 'cfxplorer', 'NICE', 'feature_tweak']
    
    total_fixed = 0
    
    print("Fixing Unicode characters in experiment files...")
    print("=" * 60)
    
    for dir_name in experiment_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"\nProcessing {dir_name} folder...")
            
            # Find all Python files
            py_files = list(dir_path.glob('*.py'))
            
            for py_file in py_files:
                if fix_unicode_in_file(py_file):
                    total_fixed += 1
                    
        else:
            print(f"Directory not found: {dir_path}")
    
    print("\n" + "=" * 60)
    print(f"Unicode fix complete! Fixed {total_fixed} files.")
    
    if total_fixed > 0:
        print("\nFixed Unicode characters:")
        print("  📝 → [LOG]")
        print("  📊 → [SUMMARY]")
        print("  🔄 → [INSIGHTS]")
        print("  ✅ → [SUCCESS]")
        print("  ❌ → [ERROR]")
        print("  And others...")

if __name__ == "__main__":
    main()

 #!/usr/bin/env python3

import os
import re

def test_ocean_lof_parsing():
    """Test OCEAN LOF parsing for all available datasets"""
    print("Testing OCEAN LOF parsing...")
    
    ocean_files = ['ocean_v2.log', 'ocean_v3.log', 'ocean_v5_compas.log']
    dataset_mapping = {
        'ocean_v2.log': 'spambase',
        'ocean_v3.log': 'german_credit',
        'ocean_v5_compas.log': 'compas'
    }
    
    for filename in ocean_files:
        if not os.path.exists(filename):
            print(f"ERROR: {filename} not found")
            continue
            
        print(f"\n--- Testing {filename} ({dataset_mapping[filename]}) ---")
        
        try:
            # Try multiple encodings
            content = None
            for encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(filename, 'r', encoding=encoding) as f:
                        content = f.read()
                    print(f"Successfully read with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"Failed to read {filename} with any encoding")
                continue
                
            # Test OCEAN LOF pattern
            baseline_lof_pattern = r'CF LOF Score: ([\d.]+)(?:\s*[±�]?\s*[\d.]+)?'
            lof_matches = re.findall(baseline_lof_pattern, content)
            
            print(f"Found {len(lof_matches)} traditional LOF scores")
            
            if not lof_matches:
                # Try OCEAN baseline bin pattern
                baseline_bin_pattern = r'Bin 0: Remove 0%.*?LOF: ([-+]?[\d.]+)'
                bin_lof_matches = re.findall(baseline_bin_pattern, content)
                print(f"Found {len(bin_lof_matches)} baseline bin LOF scores")
                
                if bin_lof_matches:
                    lof_values = [float(lof) for lof in bin_lof_matches]
                    avg_lof = sum(lof_values) / len(lof_values)
                    print(f"Average baseline LOF: {avg_lof:.4f}")
                    print(f"LOF values: {lof_values[:5]}...")  # Show first 5
                
            if not lof_matches:
                # Try alternative patterns
                print("Trying alternative patterns...")
                
                # Look for any line containing "LOF"
                lof_lines = [line.strip() for line in content.split('\n') if 'LOF' in line.upper()]
                print(f"Found {len(lof_lines)} lines containing 'LOF':")
                for line in lof_lines[:5]:  # Show first 5
                    print(f"  {line}")
                    
        except Exception as e:
            print(f"ERROR processing {filename}: {e}")

if __name__ == "__main__":
    test_ocean_lof_parsing()

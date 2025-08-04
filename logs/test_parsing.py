"""
Simple test script to verify log file parsing logic
and check if visualization dependencies are available.
"""

import os
import sys
import re

def check_dependencies():
    """Check if required packages are available"""
    required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is available")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def test_log_parsing():
    """Test the log file parsing logic"""
    logs_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all log files
    log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
    
    print(f"\nFound {len(log_files)} log files:")
    for log_file in sorted(log_files):
        print(f"  - {log_file}")
    
    # Test parsing on one log file
    if log_files:
        test_file = log_files[0]
        test_path = os.path.join(logs_dir, test_file)
        
        print(f"\nTesting parsing on: {test_file}")
        
        with open(test_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Test baseline extraction
        baseline_pattern = r'Baseline counterfactual validity: ([\d.]+) ± ([\d.]+)'
        baseline_match = re.search(baseline_pattern, content)
        if baseline_match:
            validity = float(baseline_match.group(1))
            print(f"  ✓ Found baseline validity: {validity}")
        else:
            print("  ✗ Could not find baseline validity")
        
        # Test data perturbation extraction
        perturbation_types = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
        found_perturbations = []
        
        for pert_type in perturbation_types:
            if pert_type in content:
                found_perturbations.append(pert_type)
        
        print(f"  ✓ Found perturbation types: {found_perturbations}")
        
        # Test model perturbation extraction
        summary_pattern = r'((?:random_forest|xgboost|lightgbm|adaboost)_[\d_]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+)'
        matches = re.findall(summary_pattern, content)
        print(f"  ✓ Found {len(matches)} model configuration results")
        
        if matches:
            print("    Sample model results:")
            for i, match in enumerate(matches[:3]):  # Show first 3
                parts = match.split()
                if len(parts) >= 4:
                    model_name = parts[0]
                    validity = float(parts[1])
                    accuracy = float(parts[3])
                    print(f"      {model_name}: validity={validity:.3f}, accuracy={accuracy:.3f}")
    
    return True

def main():
    """Main test function"""
    print("=" * 60)
    print("LOG PARSING TEST SCRIPT")
    print("=" * 60)
    
    print("1. Checking dependencies...")
    deps_ok = check_dependencies()
    
    print("\n2. Testing log file parsing...")
    parsing_ok = test_log_parsing()
    
    print("\n" + "=" * 60)
    if deps_ok and parsing_ok:
        print("✓ ALL TESTS PASSED!")
        print("\nYou can now run the main visualization script:")
        print("python visualization_analysis.py")
    else:
        print("✗ SOME TESTS FAILED!")
        if not deps_ok:
            print("\nTo install missing dependencies, run:")
            print("pip install pandas numpy matplotlib seaborn")
            print("OR")
            print("conda env create -f ../environment.yml")
            print("conda activate cf-robustness")
    print("=" * 60)

if __name__ == "__main__":
    main()

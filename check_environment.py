#!/usr/bin/env python3
"""
Environment Check Script for CF Robustness Framework

This script verifies that all required dependencies are properly installed
and can be imported successfully.
"""

import sys
import importlib
from packaging import version

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    py_version = sys.version_info
    if py_version >= (3, 8) and py_version < (3, 11):
        print(f"✓ Python {py_version.major}.{py_version.minor}.{py_version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {py_version.major}.{py_version.minor}.{py_version.micro} is not compatible")
        print("  Required: Python >= 3.8 and < 3.11")
        return False

def check_package(package_name, min_version=None, import_name=None):
    """Check if a package is installed and meets minimum version requirement."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        
        if min_version:
            try:
                if hasattr(module, '__version__'):
                    pkg_version = module.__version__
                else:
                    # For some packages, version might be in a different location
                    pkg_version = importlib.import_module(f"{import_name}.version").__version__
                
                if version.parse(pkg_version) >= version.parse(min_version):
                    print(f"✓ {package_name} {pkg_version} (>= {min_version})")
                    return True
                else:
                    print(f"✗ {package_name} {pkg_version} (requires >= {min_version})")
                    return False
            except:
                print(f"? {package_name} (version check failed, but package is available)")
                return True
        else:
            print(f"✓ {package_name} (available)")
            return True
            
    except ImportError:
        print(f"✗ {package_name} (not installed)")
        return False

def main():
    """Run all environment checks."""
    print("=" * 60)
    print("CF Robustness Framework - Environment Check")
    print("=" * 60)
    
    all_good = True
    
    # Check Python version
    all_good &= check_python_version()
    print()
    
    # Core dependencies
    print("Checking core dependencies...")
    core_packages = [
        ("numpy", "1.20.0"),
        ("pandas", "1.3.0"),
        ("scipy", "1.7.0"),
        ("sklearn", "1.0.0", "sklearn"),
        ("matplotlib", "3.4.0"),
        ("seaborn", "0.11.0"),
    ]
    
    for package_info in core_packages:
        all_good &= check_package(*package_info)
    print()
    
    # Machine learning libraries
    print("Checking machine learning libraries...")
    ml_packages = [
        ("xgboost", "1.5.0"),
        ("lightgbm", "3.2.0"),
        ("dice_ml", "0.9", "dice_ml"),
    ]
    
    for package_info in ml_packages:
        all_good &= check_package(*package_info)
    print()
    
    # Optional packages
    print("Checking optional packages...")
    optional_packages = [
        ("jupyter", None),
        ("ipykernel", None),
        ("numba", None),
        ("pytest", None),
    ]
    
    for package_info in optional_packages:
        check_package(*package_info)  # Don't affect overall status
    print()
    
    # CFXplorer (special case)
    print("Checking CFXplorer...")
    try:
        import cfxplorer
        print("✓ CFXplorer (available)")
    except ImportError:
        print("? CFXplorer (not installed - required only for CFXplorer-based analysis)")
    print()
    
    # Framework modules
    print("Checking framework modules...")
    try:
        sys.path.append("modules")
        import data_module
        import perturb
        print("✓ Framework modules (data_module, perturb)")
    except ImportError as e:
        print(f"✗ Framework modules (error: {e})")
        all_good = False
    print()
    
    # Final status
    print("=" * 60)
    if all_good:
        print("✓ Environment check PASSED")
        print("All required dependencies are properly installed.")
        print("You can now run the CF robustness analysis scripts.")
    else:
        print("✗ Environment check FAILED") 
        print("Some required dependencies are missing or outdated.")
        print("Please install missing packages using:")
        print("  conda env create -f environment.yml")
        print("  OR")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main()) 
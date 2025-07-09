#!/usr/bin/env python3
"""
Setup script for DiCE ML conda environment
Creates a new conda environment with all required dependencies
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}")
    print(f"Command: {command}")
    print("-" * 60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout.strip():
                print("Output:")
                print(result.stdout)
        else:
            print("❌ Error!")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False
    
    return True

def main():
    print("=" * 80)
    print("DiCE ML CONDA ENVIRONMENT SETUP")
    print("=" * 80)
    print("This script will create a new conda environment with DiCE ML")
    print("and all required dependencies for counterfactual generation.")
    print("=" * 80)
    
    env_name = "dice_cf_env"
    python_version = "3.9"
    
    # Step 1: Create conda environment
    if not run_command(
        f"conda create -n {env_name} python={python_version} -y",
        f"Creating conda environment '{env_name}' with Python {python_version}"
    ):
        print("\n❌ Failed to create conda environment!")
        return False
    
    # Step 2: Install basic scientific packages via conda
    if not run_command(
        f"conda install -n {env_name} numpy pandas scikit-learn scipy -y",
        "Installing basic scientific packages via conda"
    ):
        print("\n❌ Failed to install basic packages!")
        return False
    
    # Step 3: Install DiCE ML via pip in the conda environment
    if not run_command(
        f"conda run -n {env_name} pip install dice-ml",
        "Installing DiCE ML via pip in the conda environment"
    ):
        print("\n❌ Failed to install DiCE ML!")
        return False
    
    # Step 4: Verify installation
    print(f"\n🔍 Verifying installation in '{env_name}' environment...")
    if not run_command(
        f"conda run -n {env_name} python -c \"import dice_ml; print(f'DiCE ML version: {{dice_ml.__version__}}')\"",
        "Verifying DiCE ML installation"
    ):
        print("\n❌ Failed to verify DiCE ML installation!")
        return False
    
    print(f"\n{'='*80}")
    print("🎉 CONDA ENVIRONMENT SETUP COMPLETE!")
    print(f"{'='*80}")
    print(f"Environment name: {env_name}")
    print(f"Python version: {python_version}")
    print("\n📋 NEXT STEPS:")
    print(f"1. Activate the environment: conda activate {env_name}")
    print("2. Run dependency check: python check_dice_installation.py")
    print("3. Run counterfactual generation: python dice_counterfactual_generation.py")
    print("\n💡 OR use the automated script:")
    print("   python run_dice_in_conda.py")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed! Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!") 
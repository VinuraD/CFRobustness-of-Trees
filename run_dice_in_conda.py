#!/usr/bin/env python3
"""
Automated script to run DiCE counterfactual generation in conda environment
This script handles environment activation and runs all necessary steps
"""

import subprocess
import sys
import os
from datetime import datetime

def run_in_conda(command, env_name="dice_cf_env", description="Running command"):
    """Run a command in the specified conda environment"""
    print(f"\n🔧 {description}")
    print(f"Environment: {env_name}")
    print(f"Command: {command}")
    print("-" * 60)
    
    full_command = f"conda run -n {env_name} {command}"
    
    try:
        # Use Popen for real-time output
        process = subprocess.Popen(
            full_command, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ Command completed successfully!")
            return True
        else:
            print(f"❌ Command failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        return False

def check_conda_installed():
    """Check if conda is installed and available"""
    try:
        result = subprocess.run("conda --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Conda is available: {result.stdout.strip()}")
            return True
        else:
            print("❌ Conda is not available!")
            return False
    except Exception as e:
        print(f"❌ Error checking conda: {e}")
        return False

def check_environment_exists(env_name="dice_cf_env"):
    """Check if the conda environment exists"""
    try:
        result = subprocess.run(f"conda env list", shell=True, capture_output=True, text=True)
        if env_name in result.stdout:
            print(f"✅ Environment '{env_name}' exists")
            return True
        else:
            print(f"❌ Environment '{env_name}' does not exist")
            return False
    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False

def main():
    start_time = datetime.now()
    
    print("=" * 80)
    print("DiCE COUNTERFACTUAL GENERATION - AUTOMATED CONDA EXECUTION")
    print("=" * 80)
    print(f"🕒 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    env_name = "dice_cf_env"
    
    # Step 1: Check conda availability
    print("\n1️⃣ Checking conda installation...")
    if not check_conda_installed():
        print("\n❌ Conda is not installed or not available in PATH!")
        print("Please install Anaconda or Miniconda and try again.")
        return False
    
    # Step 2: Check if environment exists, create if not
    print(f"\n2️⃣ Checking for environment '{env_name}'...")
    if not check_environment_exists(env_name):
        print(f"\n🔧 Environment '{env_name}' not found. Creating it now...")
        print("This may take several minutes...")
        
        # Run the setup script
        if not subprocess.run("python setup_dice_environment.py", shell=True).returncode == 0:
            print(f"\n❌ Failed to create environment '{env_name}'!")
            print("Please run 'python setup_dice_environment.py' manually and check for errors.")
            return False
    
    # Step 3: Run dependency check
    print(f"\n3️⃣ Running dependency check in '{env_name}'...")
    if not run_in_conda(
        "python check_dice_installation.py",
        env_name,
        "Checking DiCE ML and dependencies"
    ):
        print(f"\n❌ Dependency check failed in environment '{env_name}'!")
        return False
    
    # Step 4: Run the main DiCE counterfactual generation
    print(f"\n4️⃣ Running DiCE counterfactual generation in '{env_name}'...")
    if not run_in_conda(
        "python dice_counterfactual_generation.py",
        env_name,
        "Executing DiCE counterfactual generation for Spambase fold 0"
    ):
        print(f"\n❌ DiCE counterfactual generation failed!")
        return False
    
    # Step 5: Success summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*80}")
    print("🎉 DiCE COUNTERFACTUAL ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"🕒 Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕒 Completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ Total duration: {duration}")
    print(f"🌍 Environment used: {env_name}")
    print(f"📊 Dataset analyzed: Spambase (Fold 0)")
    print(f"🤖 Method: DiCE ML Random Counterfactual Generation")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  • Log file: dice_counterfactual_analysis_YYYYMMDD_HHMMSS.log")
    print(f"  • Check current directory for timestamped log file")
    
    print(f"\n🔧 ENVIRONMENT INFO:")
    print(f"  • To manually activate: conda activate {env_name}")
    print(f"  • To deactivate: conda deactivate")
    print(f"  • To remove environment: conda env remove -n {env_name}")
    
    print(f"\n📈 NEXT STEPS:")
    print(f"  • Review the generated log file for detailed metrics")
    print(f"  • Analyze validity, L2 distance, LOF, and sparsity results")
    print(f"  • Compare with baseline robustness framework results")
    print(f"{'='*80}")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Execution failed! Please check the errors above.")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Ensure conda is installed and in PATH")
        print("2. Run 'python setup_dice_environment.py' manually if needed")
        print("3. Check that data/Spambase.csv exists")
        print("4. Verify modules/data_module.py and modules/perturb.py are available")
        sys.exit(1)
    else:
        print("\n✅ All operations completed successfully!") 
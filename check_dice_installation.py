#!/usr/bin/env python3
"""
Check DiCE ML installation and provide setup instructions
"""

def check_dice_installation():
    print("=" * 60)
    print("DiCE ML INSTALLATION CHECK")
    print("=" * 60)
    
    try:
        import dice_ml
        print("✅ DiCE ML is installed!")
        print(f"   Version: {dice_ml.__version__}")
        
        # Test basic functionality
        try:
            from dice_ml.utils import helpers
            print("✅ DiCE utils are accessible")
        except ImportError as e:
            print(f"⚠️  DiCE utils import warning: {e}")
            
        return True
        
    except ImportError:
        print("❌ DiCE ML is NOT installed!")
        print("\n📦 INSTALLATION INSTRUCTIONS:")
        print("   Option 1 - Install via pip:")
        print("   pip install dice-ml")
        print("\n   Option 2 - Install from requirements:")
        print("   pip install -r requirements_dice.txt")
        print("\n   Option 3 - Install with specific version:")
        print("   pip install dice-ml>=0.9")
        
        print("\n🔧 ADDITIONAL DEPENDENCIES:")
        print("   The following packages are also required:")
        print("   - scikit-learn>=1.0.0")
        print("   - pandas>=1.3.0") 
        print("   - numpy>=1.20.0")
        print("   - scipy>=1.7.0")
        
        print("\n💡 TIP:")
        print("   After installation, restart your Python environment")
        print("   and run this script again to verify the installation.")
        
        return False

def check_other_dependencies():
    print("\n" + "=" * 60)
    print("OTHER DEPENDENCIES CHECK")
    print("=" * 60)
    
    dependencies = [
        ('sklearn', 'scikit-learn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy')
    ]
    
    all_good = True
    
    for module_name, package_name in dependencies:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: NOT INSTALLED")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    dice_ok = check_dice_installation()
    deps_ok = check_other_dependencies()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if dice_ok and deps_ok:
        print("🎉 All dependencies are installed!")
        print("   You can now run: python dice_counterfactual_generation.py")
    else:
        print("⚠️  Some dependencies are missing.")
        print("   Please install the missing packages before proceeding.")
        
    print("=" * 60) 
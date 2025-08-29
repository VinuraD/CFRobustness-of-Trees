#!/usr/bin/env python3
"""
Run the visualization analysis and create a summary report
"""

import sys
import os
sys.path.append('.')

def main():
    print("🚀 Running CFRobustness visualization analysis...")
    
    # Import and run
    from visualization_analysis import CFRobustnessAnalyzer
    
    analyzer = CFRobustnessAnalyzer()
    
    print("📊 Starting analysis...")
    analyzer.run_analysis()
    
    print("✅ Analysis complete!")
    
    # Check what algorithms were processed
    algorithms = list(analyzer.data_perturbation_results.keys())
    print(f"🔍 Algorithms processed: {algorithms}")
    
    # Check OCEAN specifically
    if 'ocean' in analyzer.data_perturbation_results:
        datasets = list(analyzer.data_perturbation_results['ocean'].keys())
        print(f"🌊 OCEAN datasets: {datasets}")
        
        for dataset in datasets:
            data_keys = list(analyzer.data_perturbation_results['ocean'][dataset].keys())
            print(f"   {dataset}: {len(data_keys)} data keys")
    else:
        print("❌ No OCEAN data found")
    
    # List created plots
    if os.path.exists('img'):
        print(f"\n📈 Plots created in img/ folder")
        for subdir in os.listdir('img'):
            if os.path.isdir(f'img/{subdir}'):
                files = os.listdir(f'img/{subdir}')
                print(f"   {subdir}: {len(files)} plots")

if __name__ == "__main__":
    main()

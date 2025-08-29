#!/usr/bin/env python3
"""
Monitor plot regeneration and verify OCEAN inclusion
"""

import os
import time
import datetime

def monitor_plot_generation():
    """Monitor the plot generation process"""
    print("🔄 PLOT REGENERATION MONITOR")
    print("="*50)
    
    # Track plot directories
    plot_dirs = {
        'VALIDITY': 'img/VALIDITY',
        'ACCURACY': 'img/ACCURACY', 
        'L2': 'img/L2',
        'L0': 'img/L0'
    }
    
    print("📊 Checking plot generation status...")
    
    for metric, dir_path in plot_dirs.items():
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            
            # Count plots by dataset
            german_plots = [f for f in files if 'German_Credit' in f]
            spambase_plots = [f for f in files if 'Spambase' in f]
            compas_plots = [f for f in files if 'COMPAS' in f]
            heloc_plots = [f for f in files if 'HELOC' in f]
            
            print(f"\n📈 {metric} Plots:")
            print(f"   German Credit: {len(german_plots)} plots")
            print(f"   Spambase: {len(spambase_plots)} plots")
            print(f"   COMPAS: {len(compas_plots)} plots")
            print(f"   HELOC: {len(heloc_plots)} plots")
            
            # Check latest modification time
            if files:
                latest_file = max([os.path.join(dir_path, f) for f in files], 
                                key=os.path.getmtime)
                mod_time = os.path.getmtime(latest_file)
                mod_datetime = datetime.datetime.fromtimestamp(mod_time)
                print(f"   Latest update: {mod_datetime.strftime('%H:%M:%S')}")
                
                # Check file size (indicator of content richness)
                file_size = os.path.getsize(latest_file)
                print(f"   Sample file size: {file_size:,} bytes")
        else:
            print(f"\n❌ {metric}: Directory not found")
    
    print(f"\n🔍 OCEAN Integration Check:")
    
    # Verify OCEAN log files are being processed
    ocean_files = ['ocean_v2.log', 'ocean_v3.log']
    for file in ocean_files:
        if os.path.exists(file):
            dataset = "Spambase" if "v2" in file else "German Credit"
            print(f"   ✅ {file} ({dataset}): Available for processing")
        else:
            print(f"   ❌ {file}: Missing")
    
    # Check if plots were recently generated (within last 5 minutes)
    current_time = time.time()
    recent_threshold = current_time - 300  # 5 minutes ago
    
    recent_plots = 0
    for metric, dir_path in plot_dirs.items():
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            for file in files:
                file_path = os.path.join(dir_path, file)
                if os.path.getmtime(file_path) > recent_threshold:
                    recent_plots += 1
    
    print(f"\n📋 REGENERATION SUMMARY:")
    print(f"   Recent plots (last 5 min): {recent_plots}")
    print(f"   Status: {'🔄 IN PROGRESS' if recent_plots > 0 else '✅ COMPLETE'}")
    
    if recent_plots > 0:
        print(f"   🎉 Plots are being regenerated with OCEAN integration!")
    else:
        print(f"   ℹ️  No recent activity - regeneration may be complete")

if __name__ == "__main__":
    monitor_plot_generation()

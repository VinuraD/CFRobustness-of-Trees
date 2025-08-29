#!/usr/bin/env python3

import pandas as pd
import os

def check_lof_summary():
    """Check the LOF summary to verify OCEAN is included"""
    
    lof_summary_path = os.path.join("img", "LOF", "lof_baseline_summary.csv")
    
    if not os.path.exists(lof_summary_path):
        print(f"ERROR: {lof_summary_path} not found")
        return
    
    print("Reading LOF baseline summary...")
    df = pd.read_csv(lof_summary_path)
    
    print(f"LOF summary shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nFirst few rows:")
    print(df.head())
    
    # Check if OCEAN is included
    if 'Algorithm' in df.columns:
        algorithms = df['Algorithm'].unique() if 'Algorithm' in df.columns else []
        print(f"\nAlgorithms found: {algorithms}")
        
        if 'OCEAN' in algorithms:
            print("✓ OCEAN is successfully included in LOF analysis!")
            
            # Show OCEAN-specific data
            ocean_data = df[df['Algorithm'] == 'OCEAN']
            print(f"\nOCEAN LOF data:")
            print(ocean_data)
        else:
            print("✗ OCEAN not found in LOF analysis")
    else:
        # The CSV might have multi-level columns
        print("\nMulti-level columns detected. Full data:")
        print(df)

if __name__ == "__main__":
    check_lof_summary()

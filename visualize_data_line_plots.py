"""
Line Plot Visualizations for German Credit Data Perturbations

Creates line plots showing how mean validity changes across bins for each perturbation type,
with different colors for each CF method and error bars for standard deviation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from baseline_data_reader import ExperimentDataReader
import os

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Set consistent font styling
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

def load_dataset_data(dataset_name):
    """Load dataset data and return the reader object"""
    print(f"Loading {dataset_name} data...")
    reader = ExperimentDataReader(dataset_name.lower().replace(' ', '_'))
    if reader.load_and_analyze():
        return reader
    else:
        raise ValueError(f"Failed to load {dataset_name} data")

def create_line_plot_for_perturbation(data_dict, perturbation_type, dataset_name, save_dir="visualizations/data_perturb"):
    """Create a line plot for a specific perturbation type across all methods"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for methods
    colors = {
        'DICE': '#1f77b4',      # Blue
        'NICE': '#ff7f0e',      # Orange  
        'FOCUS': '#2ca02c',     # Green (previously cfxplorer)
        'OCEAN': '#d62728',     # Red
        'FEATURE TWEAK': '#9467bd',  # Purple
        'CEML': '#8c564b'       # Brown
    }
    
    # Plot each method (data perturbations include all 6 methods)
    for method in ['DICE', 'NICE', 'FOCUS', 'OCEAN', 'FEATURE TWEAK', 'CEML']:
        if method in data_dict and perturbation_type in data_dict[method]:
            bin_data = data_dict[method][perturbation_type]
            
            # Extract bins, validity means, and std
            bins = [d['bin'] for d in bin_data]
            validity_means = [d['mean_validity'] for d in bin_data]
            validity_stds = [d['std_validity'] if pd.notna(d['std_validity']) else 0 for d in bin_data]
            
            # Plot line with error bars
            plt.errorbar(bins, validity_means, yerr=validity_stds, 
                        label=method, color=colors[method], 
                        marker='o', linewidth=2.5, markersize=8, capsize=5)
    
    # Customize the plot
    plt.xlabel('Bin', fontweight='bold')
    plt.ylabel('Mean Validity', fontweight='bold')
    plt.title(f'{dataset_name.replace("_", " ").title()} - {perturbation_type.replace("_", " ").title()}: Mean Validity vs Bin', 
              fontweight='bold', pad=20)
    
    # Place legend at top right corner
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits and integer ticks for x-axis
    plt.ylim(0, 1.1)  # Validity should be between 0 and 1
    
    # Set x-axis to show integer values only
    # Set x-axis to show integer values only
    if bins:
        all_bins = sorted(set(bins))
        plt.xticks(all_bins)
        plt.xlim(min(all_bins) - 0.5, max(all_bins) + 0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_{perturbation_type}_validity_vs_bin.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved validity plot: {plot_path}")
    
    plt.close()  # Close the figure to free memory
    
    return plot_path

def create_accuracy_line_plot_for_perturbation(data_dict, perturbation_type, dataset_name, save_dir="visualizations/data_perturb"):
    """Create a line plot for accuracy for a specific perturbation type across all methods"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Define colors for methods
    colors = {
        'DICE': '#1f77b4',      # Blue
        'NICE': '#ff7f0e',      # Orange  
        'FOCUS': '#2ca02c',     # Green (previously cfxplorer)
        'OCEAN': '#d62728',     # Red
        'FEATURE TWEAK': '#9467bd',  # Purple
        'CEML': '#8c564b'       # Brown
    }
    
    # Plot each method (data perturbations include all 6 methods)
    for method in ['DICE', 'NICE', 'FOCUS', 'OCEAN', 'FEATURE TWEAK', 'CEML']:
        if method in data_dict and perturbation_type in data_dict[method]:
            bin_data = data_dict[method][perturbation_type]
            
            # Extract bins, accuracy means, and std
            bins = [d['bin'] for d in bin_data]
            accuracy_means = [d['mean_accuracy'] for d in bin_data]
            accuracy_stds = [d['std_accuracy'] if pd.notna(d['std_accuracy']) else 0 for d in bin_data]
            
            # Plot line with error bars
            plt.errorbar(bins, accuracy_means, yerr=accuracy_stds, 
                        label=method, color=colors[method], 
                        marker='s', linewidth=2.5, markersize=8, capsize=5)
    
    # Customize the plot
    plt.xlabel('Bin', fontweight='bold')
    plt.ylabel('Mean Accuracy', fontweight='bold')
    plt.title(f'{dataset_name.replace("_", " ").title()} - {perturbation_type.replace("_", " ").title()}: Mean Accuracy vs Bin', 
              fontweight='bold', pad=20)
    
    # Place legend at top right corner
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits and integer ticks for x-axis
    plt.ylim(0, 1.1)  # Accuracy should be between 0 and 1
    
    # Set x-axis to show integer values only
    if bins:
        all_bins = sorted(set(bins))
        plt.xticks(all_bins)
        plt.xlim(min(all_bins) - 0.5, max(all_bins) + 0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_{perturbation_type}_accuracy_vs_bin.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved accuracy plot: {plot_path}")
    
    plt.close()  # Close the figure to free memory
    
    return plot_path

def create_all_data_perturbation_plots(reader, dataset_name, save_dir="visualizations/data_perturb"):
    """Create line plots for all data perturbation types"""
    
    if not reader.data_data:
        print("ERROR: No data perturbations available")
        return
    
    # Get available perturbation types
    perturbation_types = set()
    for method_data in reader.data_data.values():
        perturbation_types.update(method_data.keys())
    
    perturbation_types = sorted(list(perturbation_types))
    print(f"\\nPLOT: Creating line plots for perturbation types: {perturbation_types}")
    
    created_plots = []
    
    for perturbation in perturbation_types:
        print(f"\\nTARGET: Creating plots for: {perturbation}")
        
        # Create validity plot
        validity_plot = create_line_plot_for_perturbation(reader.data_data, perturbation, dataset_name, save_dir)
        created_plots.append(validity_plot)
        
        # Create accuracy plot  
        accuracy_plot = create_accuracy_line_plot_for_perturbation(reader.data_data, perturbation, dataset_name, save_dir)
        created_plots.append(accuracy_plot)
    
    return created_plots

def display_data_summary(reader, dataset_name):
    """Display a summary of the data perturbations with bin information"""
    print("\\n" + "="*60)
    print(f"{dataset_name.upper().replace('_', ' ')} DATA PERTURBATIONS SUMMARY")
    print("="*60)
    
    if not reader.data_data:
        print("ERROR: No data perturbations available")
        return
    
    for method in ['DICE', 'NICE', 'FEATURE TWEAK', 'CEML']:
        if method in reader.data_data:
            print(f"\\nPLOT: {method}:")
            for perturbation, bin_data in reader.data_data[method].items():
                bins = [d['bin'] for d in bin_data]
                validity_range = [d['mean_validity'] for d in bin_data if pd.notna(d['mean_validity'])]
                
                if validity_range:
                    min_val, max_val = min(validity_range), max(validity_range)
                    print(f"  {perturbation}: {len(bins)} bins ({min(bins)}-{max(bins)}), " +
                          f"validity range: {min_val:.3f}-{max_val:.3f}")
                else:
                    print(f"  {perturbation}: {len(bins)} bins, no valid data")

def main():
    """Main function to create all data perturbations line plots for all datasets"""
    print("="*70)
    print("COUNTERFACTUAL ROBUSTNESS DATA PERTURBATIONS LINE PLOTS")
    print("="*70)
    
    datasets = ['german_credit', 'spambase', 'heloc', 'compas']
    all_created_plots = []
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"PROCESSING {dataset_name.upper().replace('_', ' ')} DATASET")
        print(f"{'='*50}")
        
        # Load data for this dataset
        reader = load_dataset_data(dataset_name)
        
        if reader is None:
            print(f"ERROR: Failed to load data for {dataset_name}")
            continue
        
        # Display summary
        display_data_summary(reader, dataset_name)
        
        # Create all line plots for this dataset
        print(f"\nPLOT: Creating line plot visualizations for {dataset_name}...")
        created_plots = create_all_data_perturbation_plots(reader, dataset_name)
        
        if created_plots:
            all_created_plots.extend(created_plots)
            print(f"SUCCESS: Created {len(created_plots)} plots for {dataset_name}!")
        else:
            print(f"ERROR: No plots created for {dataset_name}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: Created {len(all_created_plots)} total plots!")
    print(f"FOLDER: All plots saved in the 'visualizations' folder")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

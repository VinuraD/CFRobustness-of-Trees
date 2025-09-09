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
import argparse

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

# Set consistent font styling - Palatino font, size 14
import matplotlib as mpl
import matplotlib.font_manager as fm

# Try different Palatino font names that might be available on Windows
palatino_options = [
    'Palatino',
    'Palatino Linotype', 
    'Book Antiqua',  # Similar serif font often available on Windows
    'Times New Roman',
    'serif'
]

mpl.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": palatino_options,
    "font.size": 14,
})

def load_dataset_data(dataset_name):
    """Load dataset data and return the reader object"""
    print(f"Loading {dataset_name} data...")
    reader = ExperimentDataReader(dataset_name.lower().replace(' ', '_'))
    if reader.load_and_analyze():
        return reader
    else:
        raise ValueError(f"Failed to load {dataset_name} data")

def convert_bins_to_percentages(bins, perturbation_type):
    """Convert bin values to their corresponding percentages based on perturbation type"""
    if 'major' in perturbation_type:
        # For major addition/deletion: bin=0 is 0%, bin=1 is 50%
        return [bin_val * 50 for bin_val in bins]
    else:
        # For minor addition/deletion: bin=0 is 0%, bin=5 is 5%, bin=10 is 10%, etc.
        return bins  # bins already represent percentages directly

def create_line_plot_for_perturbation(data_dict, perturbation_type, dataset_name, save_dir="visualizations/data_perturb"):
    """Create a line plot for a specific perturbation type across all methods"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set up the plot
    plt.figure(figsize=(8, 4))
    
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
            
            # Convert bins to percentages
            percentage_bins = convert_bins_to_percentages(bins, perturbation_type)
            
            # Plot line with error bars
            plt.errorbar(percentage_bins, validity_means, yerr=validity_stds, 
                        label=method, color=colors[method], 
                        marker='o', linewidth=2.5, markersize=8, capsize=5)
    
    # Customize the plot
    plt.xlabel('Perturbation Percentage (%)', fontweight='bold', fontfamily='serif', fontsize=14)
    plt.ylabel('Mean Validity', fontweight='bold', fontfamily='serif', fontsize=14)
    plt.title(f'{dataset_name.replace("_", " ").title()} - {perturbation_type.replace("_", " ").title()}: Mean Validity vs Perturbation %', 
              fontweight='bold', pad=20, fontfamily='serif', fontsize=16)
    
    # Remove legend - will be created separately
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits and integer ticks for x-axis
    plt.ylim(0, 1.1)  # Validity should be between 0 and 1
    
    # Set x-axis to show percentage values
    if bins:
        all_percentage_bins = sorted(set(convert_bins_to_percentages(bins, perturbation_type)))
        plt.xticks(all_percentage_bins)
        plt.xlim(min(all_percentage_bins) - 0.5, max(all_percentage_bins) + 0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_{perturbation_type}_validity_vs_percentage.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved validity plot: {plot_path}")
    
    plt.close()  # Close the figure to free memory
    
    return plot_path

def create_accuracy_line_plot_for_perturbation(data_dict, perturbation_type, dataset_name, save_dir="visualizations/data_perturb"):
    """Create a line plot for accuracy for a specific perturbation type across all methods"""
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Set up the plot
    plt.figure(figsize=(8, 4))
    
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
            
            # Convert bins to percentages
            percentage_bins = convert_bins_to_percentages(bins, perturbation_type)
            
            # Plot line with error bars
            plt.errorbar(percentage_bins, accuracy_means, yerr=accuracy_stds, 
                        label=method, color=colors[method], 
                        marker='s', linewidth=2.5, markersize=8, capsize=5)
    
    # Customize the plot
    plt.xlabel('Perturbation Percentage (%)', fontweight='bold', fontfamily='serif', fontsize=14)
    plt.ylabel('Mean Accuracy', fontweight='bold', fontfamily='serif', fontsize=14)
    plt.title(f'{dataset_name.replace("_", " ").title()} - {perturbation_type.replace("_", " ").title()}: Mean Accuracy vs Perturbation %', 
              fontweight='bold', pad=20, fontfamily='serif', fontsize=16)
    
    # Remove legend - will be created separately
    plt.grid(True, alpha=0.3)
    
    # Set reasonable axis limits and integer ticks for x-axis
    plt.ylim(0, 1.1)  # Accuracy should be between 0 and 1
    
    # Set x-axis to show percentage values
    if bins:
        all_percentage_bins = sorted(set(convert_bins_to_percentages(bins, perturbation_type)))
        plt.xticks(all_percentage_bins)
        plt.xlim(min(all_percentage_bins) - 0.5, max(all_percentage_bins) + 0.5)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(save_dir, f'{dataset_name}_{perturbation_type}_accuracy_vs_percentage.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved accuracy plot: {plot_path}")
    
    plt.close()  # Close the figure to free memory
    
    return plot_path

def create_data_perturbation_grid_plot(datasets, save_dir="visualizations"):
    """Create 3x4 grid plot for data perturbations (3 datasets x 4 perturbation types)"""
    
    perturbation_types = ['major_deletion', 'major_addition', 'minor_deletion', 'minor_addition']
    
    # Create figure with 3x4 subplots
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))  # Reduced height from 10.5 to 10
    
    # Colors for methods
    colors = {
        'DICE': '#1f77b4',      # Blue
        'NICE': '#ff7f0e',      # Orange  
        'FOCUS': '#2ca02c',     # Green (previously cfxplorer)
        'OCEAN': '#d62728',     # Red
        'FEATURE TWEAK': '#9467bd',  # Purple
        'CEML': '#8c564b'       # Brown
    }
    
    for dataset_idx, dataset_name in enumerate(datasets):
        print(f"Processing {dataset_name} for grid plot...")
        
        # Load data for this dataset
        reader = load_dataset_data(dataset_name)
        
        if reader is None or not reader.data_data:
            print(f"ERROR: No data available for {dataset_name}")
            continue
            
        for perturb_idx, perturbation_type in enumerate(perturbation_types):
            ax = axes[dataset_idx, perturb_idx]
            
            # Plot each method
            for method in ['DICE', 'NICE', 'FOCUS', 'OCEAN', 'FEATURE TWEAK', 'CEML']:
                if method in reader.data_data and perturbation_type in reader.data_data[method]:
                    bin_data = reader.data_data[method][perturbation_type]
                    
                    # Extract data
                    bins = [d['bin'] for d in bin_data]
                    validity_means = [d['mean_validity'] for d in bin_data]
                    validity_stds = [d['std_validity'] if pd.notna(d['std_validity']) else 0 for d in bin_data]
                    
                    # Convert bins to percentages
                    percentage_bins = convert_bins_to_percentages(bins, perturbation_type)
                    
                    # Plot line with error bars
                    ax.errorbar(percentage_bins, validity_means, yerr=validity_stds, 
                               color=colors[method], marker='o', linewidth=2, markersize=6, capsize=3)
            
            # Customize subplot
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis based on perturbation type
            if bins:
                all_percentage_bins = sorted(set(convert_bins_to_percentages(bins, perturbation_type)))
                ax.set_xticks(all_percentage_bins)
                ax.set_xlim(min(all_percentage_bins) - 0.5, max(all_percentage_bins) + 0.5)
    
    # Remove individual subplot labels and titles for clean grid
    for ax in axes.flat:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
    
    # Add y-axis labels to all subplots
    for row in range(3):
        for col in range(4):
            ax = axes[row, col]
            ax.set_ylabel('Validity', fontsize=12)
    
    # Add x-axis labels to all subplots
    for row in range(3):
        for col in range(4):
            ax = axes[row, col]
            ax.set_xlabel('Perturbed data%', fontsize=12)
    
    # Add some space to the left and between rows, adjust layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15, hspace=0.4, wspace=0.3)
    
    # Add master roman numerals at the bottom for each column
    roman_numerals = ['(i)', '(ii)', '(iii)', '(iv)']
    # Position roman numerals below the x-axis labels of the bottom row (row 2)
    for col in range(4):
        # Get the bottom row subplot for this column
        bottom_ax = axes[2, col]  # Bottom row (index 2)
        # Position roman numeral centered below this subplot's x-axis label with more space
        bottom_ax.text(0.5, -0.35, roman_numerals[col], transform=bottom_ax.transAxes,
                      ha='center', va='top', fontsize=14, weight='bold')
    
    # Add row labels to the left of the y-axis labels using the added space
    row_labels = ['(a)', '(b)', '(c)']
    for row in range(3):
        fig.text(0.05, 0.85 - row * 0.27, row_labels[row], fontsize=16, weight='bold', 
                ha='center', va='center')
    
    # Save the grid plot
    os.makedirs(save_dir, exist_ok=True)
    grid_path = os.path.join(save_dir, 'data_perturbation_grid_plot.png')
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    print(f"Created data perturbation grid plot: {grid_path}")
    plt.close()
    
    return grid_path

def create_heloc_perturbation_grid_plot(save_dir="visualizations"):
    """Create 1x4 grid plot for HELOC data perturbations"""
    
    perturbation_types = ['major_deletion', 'major_addition', 'minor_deletion', 'minor_addition']
    dataset_name = 'heloc'
    
    # Create figure with 1x4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))  # Increased height by 3 points for better spacing
    
    # Colors for methods
    colors = {
        'DICE': '#1f77b4',      # Blue
        'NICE': '#ff7f0e',      # Orange  
        'FOCUS': '#2ca02c',     # Green (previously cfxplorer)
        'OCEAN': '#d62728',     # Red
        'FEATURE TWEAK': '#9467bd',  # Purple
        'CEML': '#8c564b'       # Brown
    }
    
    print(f"Processing {dataset_name} for 1x4 grid plot...")
    
    # Load data for HELOC
    reader = load_dataset_data(dataset_name)
    
    if reader is None or not reader.data_data:
        print(f"ERROR: No data available for {dataset_name}")
        return None
        
    for perturb_idx, perturbation_type in enumerate(perturbation_types):
        ax = axes[perturb_idx]
        
        # Plot each method
        for method in ['DICE', 'NICE', 'FOCUS', 'OCEAN', 'FEATURE TWEAK', 'CEML']:
            if method in reader.data_data and perturbation_type in reader.data_data[method]:
                bin_data = reader.data_data[method][perturbation_type]
                
                # Extract data
                bins = [d['bin'] for d in bin_data]
                validity_means = [d['mean_validity'] for d in bin_data]
                validity_stds = [d['std_validity'] if pd.notna(d['std_validity']) else 0 for d in bin_data]
                
                # Convert bins to percentages
                percentage_bins = convert_bins_to_percentages(bins, perturbation_type)
                
                # Plot line with error bars
                ax.errorbar(percentage_bins, validity_means, yerr=validity_stds, 
                           color=colors[method], marker='o', linewidth=2, markersize=6, capsize=3)
        
        # Customize subplot
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Set x-axis based on perturbation type
        if bins:
            all_percentage_bins = sorted(set(convert_bins_to_percentages(bins, perturbation_type)))
            ax.set_xticks(all_percentage_bins)
            ax.set_xlim(min(all_percentage_bins) - 0.5, max(all_percentage_bins) + 0.5)
    
    # Remove individual subplot labels and titles for clean grid
    for ax in axes.flat:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
    
    # Add y-axis labels to all subplots
    for col in range(4):
        ax = axes[col]
        ax.set_ylabel('Validity', fontsize=12)
    
    # Add x-axis labels to all subplots
    for col in range(4):
        ax = axes[col]
        ax.set_xlabel('Perturbed data%', fontsize=12)
    
    # Add some space to the left and adjust layout with adequate bottom margin
    plt.subplots_adjust(left=0.1, right=0.95, top=0.85, bottom=0.35, wspace=0.3)
    
    # Add master roman numerals at the bottom for each column
    roman_numerals = ['(i)', '(ii)', '(iii)', '(iv)']
    # Position roman numerals below the x-axis labels of each subplot with proper spacing
    for col in range(4):
        # Get the subplot for this column
        ax = axes[col]
        # Position roman numeral well below the x-axis label to ensure proper hierarchy
        ax.text(0.5, -0.25, roman_numerals[col], transform=ax.transAxes,
               ha='center', va='top', fontsize=14, weight='bold')
    
    # Save the grid plot
    os.makedirs(save_dir, exist_ok=True)
    grid_path = os.path.join(save_dir, 'heloc_data_perturbation_grid_plot.png')
    plt.savefig(grid_path, dpi=300, bbox_inches='tight')
    print(f"Created HELOC data perturbation grid plot: {grid_path}")
    plt.close()
    
    return grid_path

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
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate data perturbation visualizations')
    parser.add_argument('--gridplot', action='store_true', help='Generate 3x4 grid plot instead of individual plots')
    args = parser.parse_args()
    
    print("="*70)
    print("COUNTERFACTUAL ROBUSTNESS DATA PERTURBATIONS LINE PLOTS")
    print("="*70)
    
    datasets = ['german_credit', 'spambase', 'heloc', 'compas']
    
    if args.gridplot:
        print("Creating grid plots...")
        # Create 3x4 grid: compas, german_credit, spambase
        grid_datasets = ['compas', 'german_credit', 'spambase']
        grid_path_3x4 = create_data_perturbation_grid_plot(grid_datasets)
        print(f"3x4 Grid plot created: {grid_path_3x4}")
        
        # Create 1x4 grid for HELOC
        grid_path_1x4 = create_heloc_perturbation_grid_plot()
        print(f"1x4 HELOC Grid plot created: {grid_path_1x4}")
        return
    
    # Regular individual plots
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

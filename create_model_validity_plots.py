#!/usr/bin/env python3
"""
Step 2: Create line plots for model perturbations as requested
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set up plotting style with consistent font
plt.style.use('default')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)

from baseline_data_reader import ExperimentDataReader

def load_model_data(dataset_name):
    """Load and parse model perturbation data for a specific dataset"""
    
    print(f"Loading model perturbation data for {dataset_name}...")
    
    # Create reader and load model data
    reader = ExperimentDataReader(dataset_name)
    success = reader.load_and_analyze()
    
    if not success or not reader.model_data:
        print(f"ERROR: No model data found for {dataset_name}")
        return None
    
    print(f"SUCCESS: Successfully loaded model data for {dataset_name}")
    return reader

def parse_model_configurations(configs):
    """Parse model configurations to extract model types and parameters"""
    
    print(f"\nSEARCH: PARSING MODEL CONFIGURATIONS...")
    
    model_params = {}
    
    for config in configs:
        # Parse different model types
        if config.startswith('random_forest_'):
            model_type = 'Random Forest'
            params = config.replace('random_forest_', '').split('_')
        elif config.startswith('xgboost_'):
            model_type = 'XGBoost'
            params = config.replace('xgboost_', '').split('_')
        elif config.startswith('lightgbm_'):
            model_type = 'LightGBM'
            params = config.replace('lightgbm_', '').split('_')
        elif config.startswith('adaboost_'):
            model_type = 'AdaBoost'
            params = config.replace('adaboost_', '').split('_')
        else:
            print(f"  Unknown model type in config: {config}")
            continue
        
        if len(params) >= 2:
            try:
                max_depth = int(params[0])
                n_estimators = int(params[1])
                
                if model_type not in model_params:
                    model_params[model_type] = {'configs': [], 'max_depths': set(), 'n_estimators': set()}
                
                model_params[model_type]['configs'].append({
                    'config': config,
                    'max_depth': max_depth,
                    'n_estimators': n_estimators
                })
                model_params[model_type]['max_depths'].add(max_depth)
                model_params[model_type]['n_estimators'].add(n_estimators)
                
            except ValueError:
                print(f"  Could not parse parameters from: {config}")
    
    # Print summary
    print(f"\nParsed model types:")
    for model_type, data in model_params.items():
        max_depths = sorted(data['max_depths'])
        n_ests = sorted(data['n_estimators'])
        print(f"  {model_type}: {len(data['configs'])} configs")
        print(f"    max_depths: {max_depths}")
        print(f"    n_estimators: {n_ests}")
    
    return model_params

def create_validity_line_plot(model_data, model_params, cf_method, plot_type, dataset_name, save_dir="visualizations/model_perturb"):
    """
    Create line plot for validity with error bars
    
    Args:
        model_data: Parsed model data
        model_params: Parsed model parameters
        cf_method: CF method name ('NICE', 'DICE', 'FEATURE TWEAK', 'CEML')
        plot_type: '1' for max_depth=3 varying n_estimators, '2' for n_estimators=100 varying max_depth
        dataset_name: Name of the dataset for filename
        save_dir: Directory to save plots
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plot
    plt.figure(figsize=(12, 8))
    
    # Color palette for different model types
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    if plot_type == '1':
        # Type 1: max_depth=3, varying n_estimators
        fixed_param = 'max_depth'
        fixed_value = 3
        varying_param = 'n_estimators'
        x_label = 'Number of Estimators'
        title_suffix = f"(max_depth = {fixed_value})"
    else:
        # Type 2: n_estimators=100, varying max_depth
        fixed_param = 'n_estimators'
        fixed_value = 100
        varying_param = 'max_depth'
        x_label = 'Max Depth'
        title_suffix = f"(n_estimators = {fixed_value})"
    
    plot_data_exists = False
    
    for i, (model_type, data) in enumerate(sorted(model_params.items())):
        print(f"\n  Processing {model_type} for {cf_method}...")
        
        # Filter configurations for this plot type
        if plot_type == '1':
            # Get configs with max_depth=3
            filtered_configs = [c for c in data['configs'] if c['max_depth'] == fixed_value]
            x_values = []
            y_values = []
            y_errors = []
            
            for config_data in sorted(filtered_configs, key=lambda x: x['n_estimators']):
                config_name = config_data['config']
                n_est = config_data['n_estimators']
                
                if config_name in model_data[cf_method]:
                    entry = model_data[cf_method][config_name]
                    mean_val = entry['mean_validity']
                    std_val = entry['std_validity']
                    
                    if pd.notna(mean_val):
                        x_values.append(n_est)
                        y_values.append(mean_val * 100)  # Convert to percentage
                        y_errors.append(std_val * 100 if pd.notna(std_val) else 0)
                        print(f"    {config_name}: validity={mean_val:.3f}")
        else:
            # Get configs with n_estimators=100
            filtered_configs = [c for c in data['configs'] if c['n_estimators'] == fixed_value]
            x_values = []
            y_values = []
            y_errors = []
            
            for config_data in sorted(filtered_configs, key=lambda x: x['max_depth']):
                config_name = config_data['config']
                max_d = config_data['max_depth']
                
                if config_name in model_data[cf_method]:
                    entry = model_data[cf_method][config_name]
                    mean_val = entry['mean_validity']
                    std_val = entry['std_validity']
                    
                    if pd.notna(mean_val):
                        x_values.append(max_d)
                        y_values.append(mean_val * 100)  # Convert to percentage
                        y_errors.append(std_val * 100 if pd.notna(std_val) else 0)
                        print(f"    {config_name}: validity={mean_val:.3f}")
        
        # Plot the line for this model type
        if x_values:
            plt.errorbar(x_values, y_values, yerr=y_errors,
                        marker='o', linewidth=2.5, markersize=8,
                        color=colors[i % len(colors)],
                        label=model_type,
                        capsize=5, capthick=2, elinewidth=1.5)
            
            plot_data_exists = True
            print(f"    OK: Plotted {len(x_values)} points for {model_type}")
        else:
            print(f"    FAIL: No data to plot for {model_type}")
    
    if not plot_data_exists:
        print(f"  ERROR: NO DATA TO PLOT for {cf_method} plot type {plot_type}!")
        plt.close()
        return None
    
    # Customize the plot
    plt.title(f"{cf_method} - Validity vs {x_label}\n{title_suffix}", 
             fontweight='bold', pad=20)
    plt.xlabel(x_label, fontweight='bold')
    plt.ylabel('Validity (%)', fontweight='bold')
    
    # Set axis limits and clean integer ticks for x-axis
    plt.ylim(0, 105)
    
    # Set x-axis to show only integer values (no decimals)
    if plot_data_exists:
        all_x_values = []
        for i, (model_type, data) in enumerate(sorted(model_params.items())):
            if plot_type == '1':
                filtered_configs = [c for c in data['configs'] if c['max_depth'] == fixed_value]
                x_vals = [c['n_estimators'] for c in filtered_configs]
            else:
                filtered_configs = [c for c in data['configs'] if c['n_estimators'] == fixed_value]  
                x_vals = [c['max_depth'] for c in filtered_configs]
            all_x_values.extend(x_vals)
        
        if all_x_values:
            unique_x = sorted(set(all_x_values))
            plt.xticks(unique_x)
            plt.xlim(min(unique_x) - 0.5, max(unique_x) + 0.5)
    
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Place legend at top right corner
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    filename = f"{dataset_name}_{cf_method.lower().replace(' ', '_')}_validity_type{plot_type}.png"
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"OK: Saved plot: {plot_path}")
    
    plt.close()
    return plot_path

def main():
    """Main function to create all model perturbation plots for all datasets"""
    
    print("="*70)
    print("COUNTERFACTUAL ROBUSTNESS MODEL PERTURBATIONS LINE PLOTS")
    print("="*70)
    
    datasets = ['german_credit', 'spambase', 'heloc', 'compas']
    all_created_plots = []
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"PROCESSING {dataset_name.upper().replace('_', ' ')} DATASET")
        print(f"{'='*50}")
        
        # Load data for this dataset
        reader = load_model_data(dataset_name)
        
        if reader is None:
            print(f"ERROR: Failed to load data for {dataset_name}")
            continue
        
        # Create plots for each CF method using the new data structure
        created_plots = create_model_plots_for_dataset(reader, dataset_name)
        
        if created_plots:
            all_created_plots.extend(created_plots)
            print(f"SUCCESS: Created {len(created_plots)} plots for {dataset_name}!")
        else:
            print(f"ERROR: No plots created for {dataset_name}")
    
    print(f"\n{'='*70}")
    print(f"SUMMARY: Created {len(all_created_plots)} total plots!")
    print(f"FOLDER: All plots saved in the 'visualizations' folder")
    print(f"{'='*70}")
    
    return all_created_plots

def create_model_plots_for_dataset(reader, dataset_name):
    """Create proper model perturbation line plots for a specific dataset"""
    
    print(f"PLOT: Creating model perturbation plots for {dataset_name}...")
    
    if not reader.model_data:
        print(f"ERROR: No model data available for {dataset_name}")
        return []
    
    # Extract configurations from the data
    configs = []
    sample_method = list(reader.model_data.keys())[0]  # Get first available method
    configs = list(reader.model_data[sample_method].keys())
    
    print(f"Found {len(configs)} configurations")
    
    # Parse configurations to extract parameters
    model_params = parse_model_configurations(configs)
    
    if not model_params:
        print(f"ERROR: No valid model configurations found for {dataset_name}")
        return []
    
    # Updated method names to uppercase
    cf_methods = ['DICE', 'NICE', 'CEML', 'FEATURE TWEAK']
    created_plots = []
    
    for method in cf_methods:
        if method in reader.model_data:
            print(f"  Creating plots for {method}...")
            
            # Create two types of plots for each method
            # Type 1: max_depth=3, varying n_estimators  
            plot1 = create_validity_line_plot(reader.model_data, model_params, method, '1', dataset_name)
            if plot1:
                created_plots.append(plot1)
                print(f"    OK: Created Type 1 plot: {os.path.basename(plot1)}")
            
            # Type 2: n_estimators=100, varying max_depth
            plot2 = create_validity_line_plot(reader.model_data, model_params, method, '2', dataset_name)
            if plot2:
                created_plots.append(plot2)
                print(f"    OK: Created Type 2 plot: {os.path.basename(plot2)}")
        else:
            print(f"  ERROR: No data available for {method}")
    
    return created_plots

def create_method_summary_plot(method_data, method_name, dataset_name, save_dir="visualizations/model_perturb"):
    """Create a summary plot for a specific method showing validity across configurations"""
    
    # Extract validity values
    configs = list(method_data.keys())
    validities = []
    validity_stds = []
    
    for config in configs:
        data = method_data[config]
        if 'mean_validity' in data and pd.notna(data['mean_validity']):
            validities.append(data['mean_validity'])
            validity_stds.append(data.get('std_validity', 0) if pd.notna(data.get('std_validity', 0)) else 0)
        else:
            validities.append(0)
            validity_stds.append(0)
    
    if not validities or all(v == 0 for v in validities):
        print(f"    ERROR: No valid data for {method_name}")
        return None
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create bar plot
    x_pos = range(len(configs))
    bars = plt.bar(x_pos, validities, yerr=validity_stds, 
                   capsize=5, alpha=0.7, color='steelblue')
    
    # Customize the plot
    plt.title(f'{method_name} - Model Perturbation Validity\n{dataset_name.replace("_", " ").title()}', 
              fontweight='bold', pad=20)
    plt.xlabel('Model Configuration', fontweight='bold')
    plt.ylabel('Validity (%)', fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(x_pos, [f'Config {i+1}' for i in range(len(configs))], rotation=45)
    plt.ylim(0, 105)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, validities)):
        if val > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Ensure visualizations directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    filename = f"{dataset_name}_{method_name.lower().replace(' ', '_')}_model_validity.png"
    plot_path = os.path.join(save_dir, filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

if __name__ == "__main__":
    main()

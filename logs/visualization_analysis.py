"""
Counterfactual Robustness Analysis Visualization Script

This script analyzes the log files from different counterfactual algorithms
and creates comprehensive visualizations comparing their performance across
different types of data and model perturbations.

Datasets mapping:
- v2: Spambase
- v3: German Credit  
- v4: HELOC
- v5: COMPAS
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('default')  # Use clean default style
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_palette("husl")

class CFRobustnessAnalyzer:
    def __init__(self, logs_directory):
        self.logs_dir = logs_directory
        self.algorithms = ['CEML', 'DiCE', 'feature_tweak', 'NICE']
        self.datasets = {
            'v2': 'Spambase',
            'v3': 'German Credit', 
            'v4': 'HELOC',
            'v5': 'COMPAS'
        }
        self.metrics = ['validity', 'accuracy', 'L2', 'L0']
        self.data_perturbations = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
        
        # Storage for parsed data
        self.data_perturbation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.model_perturbation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.baseline_results = defaultdict(lambda: defaultdict(list))
        
    def parse_log_files(self):
        """Parse all log files and extract performance metrics"""
        print("Parsing log files...")
        
        for algorithm in self.algorithms:
            for version, dataset in self.datasets.items():
                # Priority order: .log files first, then .out files (for backward compatibility)
                log_file = None
                
                # Try .log files first
                if version == 'v4':
                    log_candidates = [f"{algorithm}_{version}_heloc.log"]
                elif version == 'v5':
                    log_candidates = [f"{algorithm}_{version}_compas.log"]
                else:
                    log_candidates = [f"{algorithm}_{version}.log"]
                
                # For NICE, also try .out files as fallback
                if algorithm == 'NICE':
                    if version == 'v4':
                        log_candidates.append(f"{algorithm}_{version}_heloc.out")
                    elif version == 'v5':
                        log_candidates.append(f"{algorithm}_{version}_compas.out")
                    else:
                        log_candidates.append(f"{algorithm}_{version}.out")
                
                # Find the first existing file
                for candidate in log_candidates:
                    candidate_path = os.path.join(self.logs_dir, candidate)
                    if os.path.exists(candidate_path):
                        log_file = candidate
                        log_path = candidate_path
                        break
                
                if log_file:
                    print(f"Processing {log_file}...")
                    self._parse_single_log_file(log_path, algorithm, dataset)
                else:
                    print(f"Warning: No log file found for {algorithm} {version}")
    
    def _parse_single_log_file(self, log_path, algorithm, dataset):
        """Parse a single log file and extract metrics"""
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extract baseline metrics
        baseline_pattern = r'Baseline counterfactual validity: ([\d.]+) ± ([\d.]+)'
        baseline_match = re.search(baseline_pattern, content)
        if baseline_match:
            validity = float(baseline_match.group(1))
            self.baseline_results[algorithm][dataset].append(validity)
        
        # Extract data perturbation results
        self._extract_data_perturbation_results(content, algorithm, dataset)
        
        # Extract model perturbation results  
        self._extract_model_perturbation_results(content, algorithm, dataset)
    
    def _extract_data_perturbation_results(self, content, algorithm, dataset):
        """Extract data perturbation results from log content"""
        # Look for data perturbation sections
        fold_sections = re.split(r'--- FOLD \d+ ANALYSIS ---', content)
        
        for fold_section in fold_sections[1:]:  # Skip first empty section
            # Extract perturbation results for each type
            for perturbation_type in self.data_perturbations:
                pattern = f'{perturbation_type}:(.*?)(?={"|".join(self.data_perturbations)}:|Testing model perturbations|--- FOLD)'
                match = re.search(pattern, fold_section, re.DOTALL)
                
                if match:
                    perturbation_section = match.group(1)
                    
                    # Extract bin-wise results with their values
                    bin_lines = re.findall(r'Bin (\d+): .*?validity: ([\d.]+), accuracy: ([\d.]+)', perturbation_section)
                    
                    if bin_lines:
                        for bin_num, validity, accuracy in bin_lines:
                            bin_key = f'{perturbation_type}_bin_{bin_num}'
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_validity'].append(float(validity))
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_accuracy'].append(float(accuracy))
                    
                    # Also extract the general metrics for compatibility
                    validity_matches = re.findall(r'validity: ([\d.]+)', perturbation_section)
                    accuracy_matches = re.findall(r'accuracy: ([\d.]+)', perturbation_section)
                    l2_matches = re.findall(r'L2: ([\d.]+)', perturbation_section)
                    l0_matches = re.findall(r'L0: ([\d.]+)', perturbation_section)
                    
                    if validity_matches and accuracy_matches:
                        # Take average across bins for this fold
                        avg_validity = np.mean([float(v) for v in validity_matches])
                        avg_accuracy = np.mean([float(a) for a in accuracy_matches])
                        avg_l2 = np.mean([float(l) for l in l2_matches]) if l2_matches else 0
                        avg_l0 = np.mean([float(l) for l in l0_matches]) if l0_matches else 0
                        
                        # Store all metrics
                        self.data_perturbation_results[algorithm][dataset][f'{perturbation_type}_validity'].append(avg_validity)
                        self.data_perturbation_results[algorithm][dataset][f'{perturbation_type}_accuracy'].append(avg_accuracy)
                        self.data_perturbation_results[algorithm][dataset][f'{perturbation_type}_l2'].append(avg_l2)
                        self.data_perturbation_results[algorithm][dataset][f'{perturbation_type}_l0'].append(avg_l0)
    
    def _extract_model_perturbation_results(self, content, algorithm, dataset):
        """Extract model perturbation summary results"""
        # Look for the summary table at the end
        summary_pattern = r'((?:random_forest|xgboost|lightgbm|adaboost)_[\d_]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+)'
        
        matches = re.findall(summary_pattern, content)
        
        if matches:
            # Group by model type and parameters for perturbation analysis
            model_groups = {
                'max_depth_3': {},
                'max_depth_4': {}, 
                'max_depth_5': {},
                'max_depth_6': {},
                'n_estimators_50': {},
                'n_estimators_100': {},
                'n_estimators_150': {},
                'n_estimators_200': {}
            }
            
            for match in matches:
                parts = match.split()
                if len(parts) >= 4:
                    model_name = parts[0]
                    validity = float(parts[1])
                    validity_std = float(parts[2])
                    accuracy = float(parts[3])
                    accuracy_std = float(parts[4])
                    
                    # Extract model type (random_forest, xgboost, lightgbm, adaboost)
                    model_type = model_name.split('_')[0] + '_' + model_name.split('_')[1] if model_name.startswith('random_forest') else model_name.split('_')[0]
                    
                    # Store for traditional model perturbation analysis
                    self.model_perturbation_results[algorithm][dataset]['validity'].append(validity)
                    self.model_perturbation_results[algorithm][dataset]['accuracy'].append(accuracy)
                    
                    # Group by perturbation type and model type for focused analysis
                    if '_3_' in model_name:
                        if model_type not in model_groups['max_depth_3']:
                            model_groups['max_depth_3'][model_type] = []
                        model_groups['max_depth_3'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_4_' in model_name:
                        if model_type not in model_groups['max_depth_4']:
                            model_groups['max_depth_4'][model_type] = []
                        model_groups['max_depth_4'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_5_' in model_name:
                        if model_type not in model_groups['max_depth_5']:
                            model_groups['max_depth_5'][model_type] = []
                        model_groups['max_depth_5'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_6_' in model_name:
                        if model_type not in model_groups['max_depth_6']:
                            model_groups['max_depth_6'][model_type] = []
                        model_groups['max_depth_6'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    
                    if '_50' in model_name:
                        if model_type not in model_groups['n_estimators_50']:
                            model_groups['n_estimators_50'][model_type] = []
                        model_groups['n_estimators_50'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_100' in model_name:
                        if model_type not in model_groups['n_estimators_100']:
                            model_groups['n_estimators_100'][model_type] = []
                        model_groups['n_estimators_100'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_150' in model_name:
                        if model_type not in model_groups['n_estimators_150']:
                            model_groups['n_estimators_150'][model_type] = []
                        model_groups['n_estimators_150'][model_type].append({'validity': validity, 'accuracy': accuracy})
                    elif '_200' in model_name:
                        if model_type not in model_groups['n_estimators_200']:
                            model_groups['n_estimators_200'][model_type] = []
                        model_groups['n_estimators_200'][model_type].append({'validity': validity, 'accuracy': accuracy})
            
            # Store grouped results for perturbation analysis with model type info
            for group_name, model_type_data in model_groups.items():
                for model_type, group_data in model_type_data.items():
                    if group_data:
                        avg_validity = np.mean([d['validity'] for d in group_data])
                        avg_accuracy = np.mean([d['accuracy'] for d in group_data])
                        
                        # Store with model type information
                        key_validity = f'{group_name}_{model_type}_validity'
                        key_accuracy = f'{group_name}_{model_type}_accuracy'
                        self.data_perturbation_results[algorithm][dataset][key_validity].append(avg_validity)
                        self.data_perturbation_results[algorithm][dataset][key_accuracy].append(avg_accuracy)
    
    def create_perturbation_value_plots(self):
        """Create line plots showing algorithm performance across perturbation values"""
        print("Creating perturbation value plots...")
        
        available_metrics = ['validity']  # Only validity for now as requested
        
        # Data perturbations with their value mappings
        data_perturbation_mappings = {
            'minor_deletion': {0: 100, 5: 95, 10: 90, 15: 85, 20: 80},  # Bin to percentage of data remaining
            'major_deletion': {0: 0, 1: 50},  # Bin 0: 0% removed, Bin 1: 50% removed
            'minor_addition': {0: 80, 5: 85, 10: 90, 15: 95, 20: 100},  # Bin to percentage mapping
            'major_addition': {0: 50, 1: 100}  # Bin 0: 50% used, Bin 1: 100% used
        }
        
        # Model perturbations with their value mappings
        model_perturbation_mappings = {
            'max_depth': [3, 4, 5, 6],
            'n_estimators': [50, 100, 150, 200]
        }
        
        # Create plots for data perturbations
        for perturbation_type in self.data_perturbations:
            for dataset in self.datasets.values():
                for metric in available_metrics:
                    self._create_data_perturbation_value_plot(perturbation_type, dataset, metric, data_perturbation_mappings[perturbation_type])
        
        # Create plots for model perturbations
        for perturbation_group, values in model_perturbation_mappings.items():
            for dataset in self.datasets.values():
                for metric in available_metrics:
                    self._create_model_perturbation_value_plot(perturbation_group, dataset, metric, values)
    
    def _create_data_perturbation_value_plot(self, perturbation_type, dataset, metric, bin_mapping):
        """Create a line plot for data perturbation showing perturbation values on x-axis"""
        
        # Collect data for all algorithms
        algorithm_data = {}
        has_data = False
        
        for algorithm in self.algorithms:
            if (algorithm in self.data_perturbation_results and 
                dataset in self.data_perturbation_results[algorithm]):
                
                x_values = []
                y_values = []
                y_errors = []
                
                for bin_num, perturbation_value in bin_mapping.items():
                    bin_key = f'{perturbation_type}_bin_{bin_num}_{metric}'
                    metric_values = self.data_perturbation_results[algorithm][dataset].get(bin_key, [])
                    
                    if metric_values:
                        x_values.append(perturbation_value)
                        y_values.append(np.mean(metric_values))
                        y_errors.append(np.std(metric_values) if len(metric_values) > 1 else 0)
                
                if x_values:
                    # Sort data points for proper line plotting
                    sorted_data = sorted(zip(x_values, y_values, y_errors))
                    if perturbation_type == 'minor_deletion':
                        # For minor deletion, sort in descending order (100 to 80)
                        sorted_data = sorted(zip(x_values, y_values, y_errors), reverse=True)
                    
                    x_values, y_values, y_errors = zip(*sorted_data)
                    
                    algorithm_data[algorithm] = {
                        'x_values': list(x_values),
                        'y_values': list(y_values),
                        'y_errors': list(y_errors)
                    }
                    has_data = True
        
        if not has_data:
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each algorithm as a different colored line
        for algorithm, data in algorithm_data.items():
            ax.errorbar(data['x_values'], data['y_values'], yerr=data['y_errors'],
                       marker='o', linewidth=2, markersize=6, capsize=3, 
                       capthick=1.5, label=algorithm, alpha=0.8)
        
        # Set labels and title with professional styling
        ax.set_xlabel(self._get_perturbation_xlabel(perturbation_type), fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
        ax.set_title(f'{perturbation_type.replace("_", " ").title()} - {dataset}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Style the legend and grid
        legend = ax.legend(frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.9, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(0.8)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Improve tick styling
        ax.tick_params(axis='both', which='major', labelsize=12, length=4, width=0.8)
        
        # Save the plot
        img_dir = os.path.join(self.logs_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        filename = f'{perturbation_type}_{dataset.replace(" ", "_")}_{metric}_vs_values.png'
        plt.savefig(os.path.join(img_dir, filename), dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated: {filename}")
    
    def _create_model_perturbation_value_plot(self, perturbation_group, dataset, metric, values):
        """Create a line plot for model perturbation showing parameter values on x-axis"""
        
        # Only use DiCE algorithm for model perturbations
        algorithm = 'DiCE'
        
        if not (algorithm in self.data_perturbation_results and 
                dataset in self.data_perturbation_results[algorithm]):
            return
        
        # Get available model types from the data
        model_types = set()
        for value in values:
            for key in self.data_perturbation_results[algorithm][dataset].keys():
                if f'{perturbation_group}_{value}_' in key and key.endswith(f'_{metric}'):
                    # Extract model type from key: max_depth_3_xgboost_validity -> xgboost
                    parts = key.split('_')
                    if len(parts) >= 4:
                        model_type = '_'.join(parts[3:-1])  # Everything between parameter and metric
                        model_types.add(model_type)
        
        if not model_types:
            return
        
        # Collect data for all model types
        model_data = {}
        has_data = False
        
        for model_type in model_types:
            x_values = []
            y_values = []
            y_errors = []
            
            for value in values:
                perturbation_key = f'{perturbation_group}_{value}_{model_type}_{metric}'
                metric_values = self.data_perturbation_results[algorithm][dataset].get(perturbation_key, [])
                
                if metric_values:
                    x_values.append(value)
                    y_values.append(np.mean(metric_values))
                    y_errors.append(np.std(metric_values) if len(metric_values) > 1 else 0)
            
            if x_values:
                model_data[model_type] = {
                    'x_values': x_values,
                    'y_values': y_values,
                    'y_errors': y_errors
                }
                has_data = True
        
        if not has_data:
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each model type as a different colored line
        for model_type, data in model_data.items():
            ax.errorbar(data['x_values'], data['y_values'], yerr=data['y_errors'],
                       marker='s', linewidth=2, markersize=6, capsize=3, 
                       capthick=1.5, label=model_type, alpha=0.8)
        
        # Set labels and title with professional styling
        ax.set_xlabel(perturbation_group.replace('_', ' ').title(), fontsize=14, fontweight='bold')
        ax.set_ylabel(f'{metric.upper()} Score', fontsize=14, fontweight='bold')
        ax.set_title(f'Model Perturbation: {perturbation_group.replace("_", " ").title()} - {dataset} (DiCE)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Style the legend and grid
        legend = ax.legend(frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.9, edgecolor='black', facecolor='white')
        legend.get_frame().set_linewidth(0.8)
        
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        # Improve tick styling
        ax.tick_params(axis='both', which='major', labelsize=12, length=4, width=0.8)
        
        # Save the plot
        img_dir = os.path.join(self.logs_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        filename = f'model_{perturbation_group}_{dataset.replace(" ", "_")}_{metric}_vs_values.png'
        plt.savefig(os.path.join(img_dir, filename), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated: {filename}")
    
    def _get_perturbation_xlabel(self, perturbation_type):
        """Get appropriate x-axis label for perturbation type"""
        if perturbation_type == 'minor_deletion':
            return 'Percentage of Data Remaining (%)'
        elif perturbation_type == 'major_deletion':
            return 'Percentage of Data Removed (%)'
        elif 'addition' in perturbation_type:
            return 'Percentage of Data Used (%)'
        else:
            return 'Perturbation Strength'

    def create_data_perturbation_visualizations(self):
        """Create visualizations for data perturbation analysis"""
        print("Creating data perturbation visualizations...")
        
        # Prepare data for visualization
        data_for_plot = []
        
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                for perturbation_type in self.data_perturbations:
                    validity_key = f'{perturbation_type}_validity'
                    if (algorithm in self.data_perturbation_results and 
                        dataset in self.data_perturbation_results[algorithm] and
                        validity_key in self.data_perturbation_results[algorithm][dataset]):
                        
                        values = self.data_perturbation_results[algorithm][dataset][validity_key]
                        if values:
                            avg_value = np.mean(values)
                            std_value = np.std(values) if len(values) > 1 else 0
                            
                            data_for_plot.append({
                                'Algorithm': algorithm,
                                'Dataset': dataset,
                                'Perturbation': perturbation_type,
                                'Validity': avg_value,
                                'Validity_Std': std_value
                            })
        
        if not data_for_plot:
            print("No data perturbation data found for visualization")
            return
            
        df = pd.DataFrame(data_for_plot)
        
        # Create line plots for different perturbation types
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, perturbation in enumerate(self.data_perturbations):
            if i < len(axes):
                perturbation_data = df[df['Perturbation'] == perturbation]
                
                if not perturbation_data.empty:
                    # Create line plot with error bars for each algorithm
                    for algorithm in self.algorithms:
                        alg_data = perturbation_data[perturbation_data['Algorithm'] == algorithm]
                        if not alg_data.empty:
                            datasets = alg_data['Dataset'].values
                            validities = alg_data['Validity'].values
                            errors = alg_data['Validity_Std'].values
                            
                            axes[i].errorbar(datasets, validities, yerr=errors, 
                                           marker='o', label=algorithm, linewidth=2, markersize=8)
                    
                    axes[i].set_title(f'Data Perturbation: {perturbation.replace("_", " ").title()}')
                    axes[i].set_xlabel('Dataset')
                    axes[i].set_ylabel('Validity')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        img_dir = os.path.join(self.logs_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(os.path.join(img_dir, 'data_perturbation_line_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create bar plots comparing algorithms across datasets
        self._create_data_perturbation_bar_plots(df)
    
    def _create_data_perturbation_bar_plots(self, df):
        """Create bar plots for data perturbation analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for i, dataset in enumerate(self.datasets.values()):
            if i < 4:
                ax = axes[i//2, i%2]
                dataset_data = df[df['Dataset'] == dataset]
                
                if not dataset_data.empty:
                    # Create grouped bar plot
                    perturbation_groups = dataset_data.groupby(['Algorithm', 'Perturbation'])['Validity'].mean().unstack()
                    perturbation_groups.plot(kind='bar', ax=ax, width=0.8)
                    
                    ax.set_title(f'Data Perturbation Robustness: {dataset}')
                    ax.set_xlabel('Algorithm')
                    ax.set_ylabel('Average Validity')
                    ax.legend(title='Perturbation Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        img_dir = os.path.join(self.logs_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(os.path.join(img_dir, 'data_perturbation_bars.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_model_perturbation_visualizations(self):
        """Create visualizations for model perturbation analysis"""
        print("Creating model perturbation visualizations...")
        
        # Prepare data for visualization
        data_for_plot = []
        
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                if (algorithm in self.model_perturbation_results and 
                    dataset in self.model_perturbation_results[algorithm]):
                    
                    validity_values = self.model_perturbation_results[algorithm][dataset].get('validity', [])
                    accuracy_values = self.model_perturbation_results[algorithm][dataset].get('accuracy', [])
                    
                    if validity_values:
                        avg_validity = np.mean(validity_values)
                        std_validity = np.std(validity_values) if len(validity_values) > 1 else 0
                        
                        avg_accuracy = np.mean(accuracy_values) if accuracy_values else 0
                        std_accuracy = np.std(accuracy_values) if len(accuracy_values) > 1 else 0
                        
                        data_for_plot.append({
                            'Algorithm': algorithm,
                            'Dataset': dataset,
                            'Avg_Validity': avg_validity,
                            'Std_Validity': std_validity,
                            'Avg_Accuracy': avg_accuracy,
                            'Std_Accuracy': std_accuracy,
                            'Robustness_Score': avg_validity  # Using validity as robustness measure
                        })
        
        if not data_for_plot:
            print("No model perturbation data found for visualization")
            return
            
        df = pd.DataFrame(data_for_plot)
        
        # Create comprehensive model perturbation visualizations
        self._create_model_perturbation_plots(df)
    
    def _create_model_perturbation_plots(self, df):
        """Create various plots for model perturbation analysis"""
        
        # 1. Line plots with error bars for validity and accuracy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Validity line plot
        for algorithm in self.algorithms:
            alg_data = df[df['Algorithm'] == algorithm]
            if not alg_data.empty:
                datasets = alg_data['Dataset'].values
                validities = alg_data['Avg_Validity'].values
                errors = alg_data['Std_Validity'].values
                
                axes[0,0].errorbar(datasets, validities, yerr=errors, 
                                 marker='o', label=algorithm, linewidth=2, markersize=8)
        
        axes[0,0].set_title('Model Perturbation: Average Validity')
        axes[0,0].set_xlabel('Dataset')
        axes[0,0].set_ylabel('Average Validity')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Accuracy line plot
        for algorithm in self.algorithms:
            alg_data = df[df['Algorithm'] == algorithm]
            if not alg_data.empty:
                datasets = alg_data['Dataset'].values
                accuracies = alg_data['Avg_Accuracy'].values
                errors = alg_data['Std_Accuracy'].values
                
                axes[0,1].errorbar(datasets, accuracies, yerr=errors, 
                                 marker='s', label=algorithm, linewidth=2, markersize=8)
        
        axes[0,1].set_title('Model Perturbation: Average Accuracy')
        axes[0,1].set_xlabel('Dataset')
        axes[0,1].set_ylabel('Average Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Robustness score line plot
        for algorithm in self.algorithms:
            alg_data = df[df['Algorithm'] == algorithm]
            if not alg_data.empty:
                datasets = alg_data['Dataset'].values
                robustness = alg_data['Robustness_Score'].values
                errors = alg_data['Std_Validity'].values  # Using validity std as proxy
                
                axes[1,0].errorbar(datasets, robustness, yerr=errors, 
                                 marker='^', label=algorithm, linewidth=2, markersize=8)
        
        axes[1,0].set_title('Model Perturbation: Robustness Score')
        axes[1,0].set_xlabel('Dataset')
        axes[1,0].set_ylabel('Robustness Score')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Standard deviation comparison
        for algorithm in self.algorithms:
            alg_data = df[df['Algorithm'] == algorithm]
            if not alg_data.empty:
                datasets = alg_data['Dataset'].values
                std_vals = alg_data['Std_Validity'].values
                
                axes[1,1].plot(datasets, std_vals, 
                             marker='D', label=algorithm, linewidth=2, markersize=8)
        
        axes[1,1].set_title('Model Perturbation: Validity Variability')
        axes[1,1].set_xlabel('Dataset')
        axes[1,1].set_ylabel('Validity Std Dev')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        img_dir = os.path.join(self.logs_dir, 'img')
        os.makedirs(img_dir, exist_ok=True)
        plt.savefig(os.path.join(img_dir, 'model_perturbation_line_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Bar plots for model perturbation comparison
        self._create_model_perturbation_bar_plots(df)
    
    def _create_model_perturbation_bar_plots(self, df):
        """Create bar plots for model perturbation analysis"""
        
        # Overall comparison across all datasets
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Algorithm comparison
        algorithm_summary = df.groupby('Algorithm').agg({
            'Avg_Validity': 'mean',
            'Avg_Accuracy': 'mean',
            'Std_Validity': 'mean'
        }).reset_index()
        
        # Validity comparison
        bars1 = axes[0,0].bar(algorithm_summary['Algorithm'], algorithm_summary['Avg_Validity'])
        axes[0,0].set_title('Average Validity Across All Datasets')
        axes[0,0].set_ylabel('Average Validity')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')
        
        # Accuracy comparison
        bars2 = axes[0,1].bar(algorithm_summary['Algorithm'], algorithm_summary['Avg_Accuracy'])
        axes[0,1].set_title('Average Accuracy Across All Datasets')
        axes[0,1].set_ylabel('Average Accuracy')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        for bar in bars2:
            height = bar.get_height()
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom')
        
        # Dataset-wise comparison
        dataset_summary = df.groupby('Dataset').agg({
            'Avg_Validity': 'mean',
            'Avg_Accuracy': 'mean'
        }).reset_index()
        
        bars3 = axes[1,0].bar(dataset_summary['Dataset'], dataset_summary['Avg_Validity'])
        axes[1,0].set_title('Average Validity by Dataset')
        axes[1,0].set_ylabel('Average Validity')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        for bar in bars3:
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom')
        
        # Variability comparison
        bars4 = axes[1,1].bar(algorithm_summary['Algorithm'], algorithm_summary['Std_Validity'])
        axes[1,1].set_title('Validity Variability Across Algorithms')
        axes[1,1].set_ylabel('Standard Deviation')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        for bar in bars4:
            height = bar.get_height()
            axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                          f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.logs_dir, 'model_perturbation_bars.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_summary(self):
        """Create a comprehensive summary visualization"""
        print("Creating comprehensive summary...")
        
        # Create a summary report
        summary_data = []
        
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                # Data perturbation summary
                data_pert_scores = []
                for pert_type in self.data_perturbations:
                    if (algorithm in self.data_perturbation_results and 
                        dataset in self.data_perturbation_results[algorithm] and
                        pert_type in self.data_perturbation_results[algorithm][dataset]):
                        values = self.data_perturbation_results[algorithm][dataset][pert_type]
                        if values:
                            data_pert_scores.append(np.mean(values))
                
                # Model perturbation summary
                model_pert_score = 0
                if (algorithm in self.model_perturbation_results and 
                    dataset in self.model_perturbation_results[algorithm] and
                    'validity' in self.model_perturbation_results[algorithm][dataset]):
                    validity_values = self.model_perturbation_results[algorithm][dataset]['validity']
                    if validity_values:
                        model_pert_score = np.mean(validity_values)
                
                summary_data.append({
                    'Algorithm': algorithm,
                    'Dataset': dataset,
                    'Data_Perturbation_Score': np.mean(data_pert_scores) if data_pert_scores else 0,
                    'Model_Perturbation_Score': model_pert_score,
                    'Overall_Robustness': (np.mean(data_pert_scores) + model_pert_score) / 2 if data_pert_scores else model_pert_score
                })
        
        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            
            # Create comprehensive comparison
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # Overall robustness line plot
            for algorithm in self.algorithms:
                alg_data = df_summary[df_summary['Algorithm'] == algorithm]
                if not alg_data.empty:
                    datasets = alg_data['Dataset'].values
                    robustness = alg_data['Overall_Robustness'].values
                    
                    axes[0,0].plot(datasets, robustness, 
                                 marker='o', label=algorithm, linewidth=2, markersize=8)
            
            axes[0,0].set_title('Overall Robustness Score')
            axes[0,0].set_xlabel('Dataset')
            axes[0,0].set_ylabel('Overall Robustness')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Data vs Model perturbation scatter
            for algorithm in self.algorithms:
                alg_data = df_summary[df_summary['Algorithm'] == algorithm]
                if not alg_data.empty:
                    axes[0,1].scatter(alg_data['Data_Perturbation_Score'], 
                                     alg_data['Model_Perturbation_Score'],
                                     label=algorithm, s=100, alpha=0.7)
            
            axes[0,1].set_xlabel('Data Perturbation Score')
            axes[0,1].set_ylabel('Model Perturbation Score')
            axes[0,1].set_title('Data vs Model Perturbation Robustness')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
            
            # Algorithm ranking
            algo_ranking = df_summary.groupby('Algorithm')['Overall_Robustness'].mean().sort_values(ascending=False)
            bars = axes[1,0].bar(algo_ranking.index, algo_ranking.values)
            axes[1,0].set_title('Algorithm Ranking by Overall Robustness')
            axes[1,0].set_ylabel('Average Robustness Score')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom')
            
            # Dataset difficulty ranking
            dataset_ranking = df_summary.groupby('Dataset')['Overall_Robustness'].mean().sort_values(ascending=True)
            bars = axes[1,1].bar(dataset_ranking.index, dataset_ranking.values)
            axes[1,1].set_title('Dataset Difficulty (Lower = More Challenging)')
            axes[1,1].set_ylabel('Average Robustness Score')
            axes[1,1].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.logs_dir, 'comprehensive_summary.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save summary table
            df_summary.to_csv(os.path.join(self.logs_dir, 'robustness_summary.csv'), index=False)
            print(f"Summary data saved to {os.path.join(self.logs_dir, 'robustness_summary.csv')}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 80)
        print("COUNTERFACTUAL ROBUSTNESS VISUALIZATION ANALYSIS")
        print("=" * 80)
        
        # Parse all log files
        self.parse_log_files()
        
        # Create new perturbation value plots (main focus)
        self.create_perturbation_value_plots()
        
        # Skip traditional visualizations as requested
        # self.create_data_perturbation_visualizations()
        # self.create_model_perturbation_visualizations()
        # self.create_comprehensive_summary()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("Generated visualizations:")
        print("- Perturbation value plots: *_vs_values.png")
        print("  (Shows algorithm performance across perturbation strength)")
        print("=" * 80)


def main():
    """Main function to run the analysis"""
    # Get the current directory (logs folder)
    logs_directory = os.path.dirname(os.path.abspath(__file__))
    
    # Create analyzer instance
    analyzer = CFRobustnessAnalyzer(logs_directory)
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()

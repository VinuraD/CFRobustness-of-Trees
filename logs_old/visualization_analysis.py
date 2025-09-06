"""
Counterfactual Robustness Analysis Visualization Script

This script analyzes the log files from different counterfactual algorithms
and creates comprehensive visualizations comparing their performance across
different types of data and model perturbations.

Supported Algorithms:
- CEML: Counterfactual Explanations for Machine Learning
- DiCE: Diverse Counterfactual Explanations
- feature_tweak: Feature Tweak counterfactual method
- NICE: Nearest Instance Counterfactual Explanations
- cfxplorer: Counterfactual Explorer
- OCEAN: Optimized Counterfactual Explanations Analysis (NEW)

Datasets mapping:
- v2: Spambase
- v3: German Credit  
- v4: HELOC
- v5: COMPAS

Note: OCEAN currently has logs for German Credit (v3) dataset.
The system is designed to automatically detect and parse OCEAN logs
for other datasets when they become available.
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
        self.algorithms = ['CEML', 'DiCE', 'feature_tweak', 'NICE', 'cfxplorer', 'OCEAN']
        self.datasets = {
            'v2': 'Spambase',
            'v3': 'German Credit', 
            'v4': 'HELOC',
            'v5': 'COMPAS'
        }
        self.metrics = ['validity', 'accuracy', 'L2', 'L0']
        self.data_perturbations = ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']
        # Note: OCEAN format uses 'data_deletion' instead of 'minor_deletion' in some cases
        
        # Storage for parsed data
        self.data_perturbation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.model_perturbation_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.baseline_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
    def parse_log_files(self):
        """Parse all log files and extract performance metrics"""
        print("Parsing log files...")
        
        for algorithm in self.algorithms:
            for version, dataset in self.datasets.items():
                # Priority order: .log files first, then .out files (for backward compatibility)
                log_file = None
                
                # Special handling for OCEAN - different file naming pattern
                if algorithm == 'OCEAN':
                    if version == 'v4':
                        log_candidates = [f"ocean_{version}_heloc.log"]
                    elif version == 'v5':
                        log_candidates = [f"ocean_{version}_compas.log"]
                    else:
                        log_candidates = [f"ocean_{version}.log"]
                else:
                    # Original logic for other algorithms
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
                    print(f"Completed processing {log_file}")
                else:
                    print(f"Warning: No log file found for {algorithm} {version}")
    
    def _parse_single_log_file(self, log_path, algorithm, dataset):
        """Parse a single log file and extract metrics"""
        # Try multiple encodings to handle ± symbols properly
        content = None
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                with open(log_path, 'r', encoding=encoding) as f:
                    content = f.read()
                print(f"Successfully read {log_path} with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            print(f"Failed to read {log_path} with any encoding")
            return
        
        # Special handling for OCEAN format
        if algorithm == 'OCEAN':
            print(f"Using OCEAN parser for {algorithm} {dataset}")
            self._parse_ocean_log_file(content, algorithm, dataset)
            return
        
        # Extract baseline metrics - handle both formats
        baseline_pattern1 = r'Baseline counterfactual validity: ([\d.]+) ± ([\d.]+)'
        baseline_pattern2 = r'Baseline CF Validity: ([\d.]+) ± ([\d.]+)'
        
        baseline_match = re.search(baseline_pattern1, content)
        if not baseline_match:
            baseline_match = re.search(baseline_pattern2, content)
            
        if baseline_match:
            validity = float(baseline_match.group(1))
            self.baseline_results[algorithm][dataset]['validity'].append(validity)
        
        # Extract baseline LOF score
        lof_pattern = r'Baseline LOF score: ([\d.]+)'
        lof_match = re.search(lof_pattern, content)
        if lof_match:
            lof_score = float(lof_match.group(1))
            self.baseline_results[algorithm][dataset]['LOF'].append(lof_score)
        
        # Extract data perturbation results
        self._extract_data_perturbation_results(content, algorithm, dataset)
        
        # Extract model perturbation results  
        self._extract_model_perturbation_results(content, algorithm, dataset)
    
    def _parse_ocean_log_file(self, content, algorithm, dataset):
        """Parse OCEAN log file format specifically"""
        print(f"Starting OCEAN parsing for {algorithm} - {dataset}")
        
        # Extract baseline metrics from OCEAN format - handle corrupted ± symbols
        # CF Validity: 0.3780 ± 0.0431 (handle both ± and � encoding, or corrupted format)
        baseline_validity_pattern = r'CF Validity: ([\d.]+)(?:\s*[±�]?\s*[\d.]+)?'
        baseline_match = re.search(baseline_validity_pattern, content)
        if baseline_match:
            validity = float(baseline_match.group(1))
            self.baseline_results[algorithm][dataset]['validity'].append(validity)
            print(f"Found OCEAN baseline validity: {validity}")
        else:
            print("No OCEAN baseline validity found")
        
        # CF LOF Score: 0.7180 ± 0.0736 (handle both ± and � encoding, or corrupted format)
        baseline_lof_pattern = r'CF LOF Score: ([\d.]+)(?:\s*[±�]?\s*[\d.]+)?'
        lof_match = re.search(baseline_lof_pattern, content)
        if lof_match:
            lof_score = float(lof_match.group(1))
            self.baseline_results[algorithm][dataset]['LOF'].append(lof_score)
            print(f"Found OCEAN baseline LOF: {lof_score}")
        else:
            # Try alternative OCEAN pattern - extract from baseline bin (Bin 0: Remove 0%)
            baseline_bin_pattern = r'Bin 0: Remove 0%.*?LOF: ([-+]?[\d.]+)'
            lof_matches = re.findall(baseline_bin_pattern, content)
            if lof_matches:
                # Take the mean of all baseline LOF values across folds
                lof_values = [float(lof) for lof in lof_matches]
                avg_lof = sum(lof_values) / len(lof_values)
                self.baseline_results[algorithm][dataset]['LOF'].append(avg_lof)
                print(f"Found OCEAN baseline LOF from bins: {avg_lof:.4f} (average of {len(lof_values)} values)")
            else:
                print("No OCEAN baseline LOF found")
        
        # Also try to extract additional baseline metrics for completeness
        # Model Accuracy: 0.7370 ± 0.0218
        baseline_accuracy_pattern = r'Model Accuracy: ([\d.]+) [±�] ([\d.]+)'
        accuracy_match = re.search(baseline_accuracy_pattern, content)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
            accuracy_std = float(accuracy_match.group(2))
            self.baseline_results[algorithm][dataset]['accuracy'].append(accuracy)
            print(f"Found OCEAN baseline accuracy: {accuracy} ± {accuracy_std}")
        
        # CF L2 Distance: 1.8840 ± 0.0291
        baseline_l2_pattern = r'CF L2 Distance: ([\d.]+) [±�] ([\d.]+)'
        l2_match = re.search(baseline_l2_pattern, content)
        if l2_match:
            l2_score = float(l2_match.group(1))
            l2_std = float(l2_match.group(2))
            self.baseline_results[algorithm][dataset]['L2'].append(l2_score)
            print(f"Found OCEAN baseline L2: {l2_score} ± {l2_std}")
        
        # CF L0 Distance: 8.0270 ± 0.1119
        baseline_l0_pattern = r'CF L0 Distance: ([\d.]+) [±�] ([\d.]+)'
        l0_match = re.search(baseline_l0_pattern, content)
        if l0_match:
            l0_score = float(l0_match.group(1))
            l0_std = float(l0_match.group(2))
            self.baseline_results[algorithm][dataset]['L0'].append(l0_score)
            print(f"Found OCEAN baseline L0: {l0_score} ± {l0_std}")
        
        # Extract data perturbation results from OCEAN summary
        print("Extracting OCEAN data perturbations...")
        self._extract_ocean_data_perturbation_results(content, algorithm, dataset)
        
        # Extract model perturbation results from OCEAN
        print("Extracting OCEAN model perturbations...")
        self._extract_ocean_model_perturbation_results(content, algorithm, dataset)
        
        print(f"Completed OCEAN parsing for {algorithm} - {dataset}")
    
    def _extract_ocean_data_perturbation_results(self, content, algorithm, dataset):
        """Extract OCEAN data perturbation results from summary section"""
        # Parse updated OCEAN summary format
        # DATA_DELETION:
        #   Amount 0 (100.0% data): Accuracy=0.7370±0.0218, Validity=0.3780±0.0431
        #   Amount 5 (95.0% data): Accuracy=0.7220±0.0199, Validity=0.3760±0.0350
        
        print(f"Extracting OCEAN data perturbations for {dataset}")
        
        # Map OCEAN perturbation names to our standard names
        ocean_to_standard = {
            'DATA_DELETION': 'minor_deletion',
            'MINOR_ADDITION': 'minor_addition', 
            'MAJOR_ADDITION': 'major_addition'
        }
        
        # Also try parsing from fold-by-fold data if summary not available
        fold_pattern = r'Fold \d+ - Train: \d+ samples, Test: \d+ samples\n.*?Bin (\d+): .*? -> validity: ([\d.]+), accuracy: ([\d.]+), L2: ([\d.]+), L0: ([\d.]+)'
        
        # Parse from summary section first
        for ocean_type, standard_type in ocean_to_standard.items():
            # Find the section for this perturbation type in summary
            pattern = f'{ocean_type}:(.*?)(?=\\n\\n[A-Z_]+:|\\nMODEL PERTURBATION|$)'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                section = match.group(1)
                print(f"Found OCEAN {ocean_type} section for {dataset}")
                
                # Updated pattern to handle multiple encoding formats:
                # 1. Proper: Amount 0 (100.0% data): Accuracy=0.7370±0.0218, Validity=0.3780±0.0431
                # 2. With �: Amount 0 (100.0% data): Accuracy=0.7370�0.0218, Validity=0.3780�0.0431  
                # 3. Concatenated: Amount 0 (100.0% data): Accuracy=0.90750.0131, Validity=0.41530.1042
                
                # First try the � pattern
                amount_pattern_1 = r"Amount (\d+) \([^)]+\): Accuracy=([\d.]+)�[\d.]+, Validity=([\d.]+)�[\d.]+"
                amounts = re.findall(amount_pattern_1, section)
                
                # If that fails, try the ± pattern
                if not amounts:
                    amount_pattern_2 = r"Amount (\d+) \([^)]+\): Accuracy=([\d.]+)±[\d.]+, Validity=([\d.]+)±[\d.]+"
                    amounts = re.findall(amount_pattern_2, section)
                
                # If that also fails, try the concatenated pattern (extract first decimal number)
                if not amounts:
                    # Pattern to capture the full potential concatenated string
                    amount_pattern_3 = r"Amount (\d+) \([^)]+\): Accuracy=([0-9.]+), Validity=([0-9.]+)"
                    potential_amounts = re.findall(amount_pattern_3, section)
                    
                    # For concatenated format, extract the first valid decimal number
                    amounts = []
                    for amount, acc_str, val_str in potential_amounts:
                        # Check if we have a concatenated format (contains multiple decimal points)
                        if acc_str.count('.') > 1 or val_str.count('.') > 1:
                            # Extract first 4-digit decimal number: 0.90750.0131 -> 0.9075
                            acc_match = re.match(r'(\d+\.\d{4})', acc_str)
                            val_match = re.match(r'(\d+\.\d{4})', val_str)
                            
                            if acc_match and val_match:
                                accuracy = acc_match.group(1)
                                validity = val_match.group(1)
                                amounts.append((amount, accuracy, validity))
                        else:
                            # Normal case - use as is
                            amounts.append((amount, acc_str, val_str))
                
                if amounts:
                    print(f"Found {len(amounts)} amount entries for {ocean_type}")
                    
                    # Store individual results with validation
                    for amount, accuracy, validity in amounts:
                        # Ensure values are properly converted to float
                        try:
                            accuracy_float = float(accuracy)
                            validity_float = float(validity)
                            
                            bin_key = f'{standard_type}_bin_{amount}'
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_validity'].append(validity_float)
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_accuracy'].append(accuracy_float)
                            
                            # OCEAN doesn't provide L0/L2 in summary, set to 0 for consistency
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_l2'].append(0)
                            self.data_perturbation_results[algorithm][dataset][f'{bin_key}_l0'].append(0)
                            print(f"OCEAN {ocean_type} Amount {amount}: Validity={validity_float}, Accuracy={accuracy_float}")
                        except ValueError as e:
                            print(f"ERROR: Could not convert OCEAN values to float - Amount {amount}, Accuracy='{accuracy}', Validity='{validity}': {e}")
                            continue
                    
                    # Also store general metrics (average across amounts)
                    # Calculate from successfully processed values only
                    valid_amounts = []
                    for amount, accuracy, validity in amounts:
                        try:
                            accuracy_float = float(accuracy)
                            validity_float = float(validity)
                            valid_amounts.append((amount, accuracy_float, validity_float))
                        except ValueError:
                            continue
                    
                    if valid_amounts:
                        avg_validity = np.mean([v[2] for v in valid_amounts])
                        avg_accuracy = np.mean([v[1] for v in valid_amounts])
                    
                    self.data_perturbation_results[algorithm][dataset][f'{standard_type}_validity'].append(avg_validity)
                    self.data_perturbation_results[algorithm][dataset][f'{standard_type}_accuracy'].append(avg_accuracy)
                    # OCEAN doesn't provide L0/L2 in perturbation results, set to 0
                    self.data_perturbation_results[algorithm][dataset][f'{standard_type}_l2'].append(0)
                    self.data_perturbation_results[algorithm][dataset][f'{standard_type}_l0'].append(0)
                    print(f"OCEAN {ocean_type} averages: Validity={avg_validity:.4f}, Accuracy={avg_accuracy:.4f}")
                else:
                    print(f"No amount entries found for OCEAN {ocean_type} in {dataset}")
            else:
                print(f"No OCEAN {ocean_type} section found for {dataset}")
        
        # Parse fold-by-fold data to extract L2 and L0 values
        self._extract_ocean_fold_metrics(content, algorithm, dataset)
        
        # Also parse CF distance metrics from the summary
        self._extract_ocean_cf_distances(content, algorithm, dataset)
    
    def _extract_ocean_model_perturbation_results(self, content, algorithm, dataset):
        """Extract OCEAN model perturbation results"""
        # Parse updated OCEAN model perturbation format (handle both ± and � encoding)
        # rf_50_3 {'n_estimators': 50, 'max_depth': 3}: Accuracy=0.7120±0.0081, Validity=0.0530±0.0129
        
        # Updated pattern for new naming convention - handle corrupted ± symbols
        model_pattern = r'rf_(\d+)_(\d+) \{[^}]+\}: Accuracy=([\d.]+)(?:[±�]?[\d.]+)?, Validity=([\d.]+)(?:[±�]?[\d.]+)?'
        matches = re.findall(model_pattern, content)
        
        if matches:
            print(f"Found {len(matches)} OCEAN model perturbation results for {dataset}")
            
            # Group by model parameters for analysis (similar to existing algorithms)
            model_groups = {
                'max_depth_3': {},
                'max_depth_4': {},
                'max_depth_5': {}, 
                'max_depth_6': {},
                'max_depth_7': {},
                'n_estimators_50': {},
                'n_estimators_100': {},
                'n_estimators_150': {},
                'n_estimators_200': {}
            }
            
            for n_estimators, max_depth, accuracy, validity in matches:
                n_estimators = int(n_estimators)
                max_depth = int(max_depth)
                accuracy = float(accuracy)
                validity = float(validity)
                
                # Store for traditional model perturbation analysis
                self.model_perturbation_results[algorithm][dataset]['validity'].append(validity)
                self.model_perturbation_results[algorithm][dataset]['accuracy'].append(accuracy)
                
                # Group by actual parameters for better analysis
                depth_key = f'max_depth_{max_depth}'
                n_est_key = f'n_estimators_{n_estimators}'
                model_type = 'random_forest'
                
                # Store in appropriate groups
                if depth_key in model_groups:
                    if model_type not in model_groups[depth_key]:
                        model_groups[depth_key][model_type] = []
                    model_groups[depth_key][model_type].append({'validity': validity, 'accuracy': accuracy})
                
                if n_est_key in model_groups:
                    if model_type not in model_groups[n_est_key]:
                        model_groups[n_est_key][model_type] = []
                    model_groups[n_est_key][model_type].append({'validity': validity, 'accuracy': accuracy})
                
                # Also store generic model results for compatibility
                model_id = f"{n_estimators}_{max_depth}"
                key_validity = f'model_{model_id}_{model_type}_validity'
                key_accuracy = f'model_{model_id}_{model_type}_accuracy'
                self.data_perturbation_results[algorithm][dataset][key_validity].append(validity)
                self.data_perturbation_results[algorithm][dataset][key_accuracy].append(accuracy)
                
                print(f"OCEAN model rf_{n_estimators}_{max_depth}: Validity={validity:.4f}, Accuracy={accuracy:.4f}")
            
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
                        print(f"OCEAN: Grouped {group_name} {model_type} - Validity: {avg_validity:.4f}, Accuracy: {avg_accuracy:.4f}")
        else:
            print(f"No OCEAN model perturbation results found for {dataset}")
    
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
                    
                    # Also look for L0 and L2 metrics in bin-wise results if available
                    bin_lines_with_l0_l2 = re.findall(r'Bin (\d+): .*?validity: ([\d.]+), accuracy: ([\d.]+)(?:, L2: ([\d.]+), L0: ([\d.]+))?', perturbation_section)
                    
                    if bin_lines_with_l0_l2:
                        for match in bin_lines_with_l0_l2:
                            bin_num, validity, accuracy = match[0], match[1], match[2]
                            l2_val, l0_val = match[3], match[4]
                            
                            bin_key = f'{perturbation_type}_bin_{bin_num}'
                            
                            # Only add L0 and L2 if they are present in the log
                            if l2_val:
                                self.data_perturbation_results[algorithm][dataset][f'{bin_key}_l2'].append(float(l2_val))
                            if l0_val:
                                self.data_perturbation_results[algorithm][dataset][f'{bin_key}_l0'].append(float(l0_val))
                    
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
        
        if algorithm == 'cfxplorer':
            # Handle cfxplorer format: random_forest (depth, n_estimators): validity X.XXXX, accuracy Y.YYYY
            pattern = r'random_forest \((\d+), (\d+)\): validity ([\d.]+), accuracy ([\d.]+)'
            matches = re.findall(pattern, content)
            
            if matches:
                for depth, n_est, validity, accuracy in matches:
                    depth = int(depth)
                    n_est = int(n_est)
                    validity = float(validity)
                    accuracy = float(accuracy)
                    
                    # Store for traditional model perturbation analysis
                    self.model_perturbation_results[algorithm][dataset]['validity'].append(validity)
                    self.model_perturbation_results[algorithm][dataset]['accuracy'].append(accuracy)
        else:
            # Handle other algorithms with tabular format
            # Look for the summary table at the end
            summary_pattern = r'((?:random_forest|xgboost|lightgbm|adaboost)_[\d_]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+\s+[\d.-]+)'
            
            matches = re.findall(summary_pattern, content)
            
            if matches:
                for match in matches:
                    parts = match.split()
                    if len(parts) >= 4:
                        model_name = parts[0]
                        validity = float(parts[1])
                        accuracy = float(parts[3])
                        
                        # Store for traditional model perturbation analysis
                        self.model_perturbation_results[algorithm][dataset]['validity'].append(validity)
                        self.model_perturbation_results[algorithm][dataset]['accuracy'].append(accuracy)

    def _extract_ocean_fold_metrics(self, content, algorithm, dataset):
        """Extract L2 and L0 metrics from OCEAN fold-by-fold data"""
        # Parse fold-by-fold data to extract L2 and L0 values
        # Bin 0: Remove 0% -> validity: 0.4400, accuracy: 0.7500, L2: 1.8768, L0: 8.02
        
        fold_data = {}
        
        # Map perturbation types based on the section names in fold data
        perturbation_mappings = {
            'data_deletion': 'minor_deletion',
            'minor_addition': 'minor_addition',
            'major_addition': 'major_addition'
        }
        
        for fold_type, standard_type in perturbation_mappings.items():
            # Find all bin entries for this perturbation type
            pattern = f'{fold_type}:(.*?)(?=\\n\\s*[a-z_]+:|\\nTesting model perturbations|=== Processing Fold)'
            
            sections = re.findall(pattern, content, re.DOTALL)
            
            if sections:
                print(f"Found {len(sections)} {fold_type} sections in fold data")
                
                for section in sections:
                    # Extract bin data: Bin X: ... -> validity: Y, accuracy: Z, L2: A, L0: B
                    bin_pattern = r'Bin (\d+): .*? -> validity: ([\d.]+), accuracy: ([\d.]+), L2: ([\d.]+), L0: ([\d.]+)'
                    bin_matches = re.findall(bin_pattern, section)
                    
                    for bin_num, validity, accuracy, l2, l0 in bin_matches:
                        bin_key = f'{standard_type}_bin_{bin_num}'
                        
                        # Store L2 and L0 values that were missing from summary
                        if f'{bin_key}_l2' not in fold_data:
                            fold_data[f'{bin_key}_l2'] = []
                        if f'{bin_key}_l0' not in fold_data:
                            fold_data[f'{bin_key}_l0'] = []
                            
                        fold_data[f'{bin_key}_l2'].append(float(l2))
                        fold_data[f'{bin_key}_l0'].append(float(l0))
        
        # Update the stored data with L2 and L0 values
        for key, values in fold_data.items():
            if values:  # Only update if we have data
                # Replace the placeholder 0 values with actual averages
                avg_value = np.mean(values)
                # Clear existing 0 values and add the real average
                if key in self.data_perturbation_results[algorithm][dataset]:
                    self.data_perturbation_results[algorithm][dataset][key] = [avg_value]
                print(f"Updated OCEAN {key}: {avg_value:.4f}")

    def _extract_ocean_cf_distances(self, content, algorithm, dataset):
        """Extract CF L2 and L0 distance metrics from OCEAN summary"""
        # Parse CF distance metrics: CF L2 Distance: 1.8840 ± 0.0291
        print(f"Extracting OCEAN CF distances for {dataset}")
        
        # L2 distance pattern - handle multiple formats
        # Try � symbol first
        l2_pattern_1 = r'CF L2 Distance: ([\d.]+) � [\d.]+'
        l2_matches = re.findall(l2_pattern_1, content)
        
        # If that fails, try ± symbol
        if not l2_matches:
            l2_pattern_2 = r'CF L2 Distance: ([\d.]+) ± [\d.]+'
            l2_matches = re.findall(l2_pattern_2, content)
        
        # If that also fails, try concatenated format
        if not l2_matches:
            l2_pattern_3 = r'CF L2 Distance: ([\d.]+)[\d.]+'
            l2_matches = re.findall(l2_pattern_3, content)
        
        # L0 distance pattern - handle multiple formats  
        # Try � symbol first
        l0_pattern_1 = r'CF L0 Distance: ([\d.]+) � [\d.]+'
        l0_matches = re.findall(l0_pattern_1, content)
        
        # If that fails, try ± symbol
        if not l0_matches:
            l0_pattern_2 = r'CF L0 Distance: ([\d.]+) ± [\d.]+'
            l0_matches = re.findall(l0_pattern_2, content)
        
        # If that also fails, try concatenated format
        if not l0_matches:
            l0_pattern_3 = r'CF L0 Distance: ([\d.]+)[\d.]+'
            l0_matches = re.findall(l0_pattern_3, content)
        
        if l2_matches:
            print(f"Found {len(l2_matches)} CF L2 distance values for {dataset}")
            # Store in general CF metrics (not perturbation-specific)
            if 'cf_l2' not in self.data_perturbation_results[algorithm][dataset]:
                self.data_perturbation_results[algorithm][dataset]['cf_l2'] = []
            for l2_val in l2_matches:
                self.data_perturbation_results[algorithm][dataset]['cf_l2'].append(float(l2_val))
                print(f"OCEAN CF L2 distance: {l2_val}")
                
        if l0_matches:
            print(f"Found {len(l0_matches)} CF L0 distance values for {dataset}")
            # Store in general CF metrics (not perturbation-specific)
            if 'cf_l0' not in self.data_perturbation_results[algorithm][dataset]:
                self.data_perturbation_results[algorithm][dataset]['cf_l0'] = []
            for l0_val in l0_matches:
                self.data_perturbation_results[algorithm][dataset]['cf_l0'].append(float(l0_val))
                print(f"OCEAN CF L0 distance: {l0_val}")
                        
        if not l2_matches and not l0_matches:
            print(f"No CF distance metrics found for OCEAN {dataset}")

    def create_l2_baseline_visualizations(self):
        """Create L2 (L2 Distance) baseline visualizations with log scale"""
        print("Creating L2 baseline visualizations...")
        
        # Check if L2 data exists
        l2_data = []
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                if (algorithm in self.baseline_results and 
                    dataset in self.baseline_results[algorithm] and
                    'L2' in self.baseline_results[algorithm][dataset]):
                    l2_values = self.baseline_results[algorithm][dataset]['L2']
                    if l2_values:
                        for l2_val in l2_values:
                            # Convert to log scale to handle extreme values
                            log_l2 = np.log10(abs(l2_val)) if l2_val != 0 else 0
                            l2_data.append({
                                'Algorithm': algorithm,
                                'Dataset': dataset,
                                'L2_Score': l2_val,
                                'Log_L2_Score': log_l2
                            })
        
        if not l2_data:
            print("No L2 baseline data found. Skipping L2 visualizations.")
            return
        
        # Create output directory
        l2_dir = os.path.join(self.logs_dir, 'img', 'L2_BASELINE')
        os.makedirs(l2_dir, exist_ok=True)
        
        df_l2 = pd.DataFrame(l2_data)
        print(f"L2 data summary:")
        print(f"Original L2 range: {df_l2['L2_Score'].min():.2f} to {df_l2['L2_Score'].max():.2f}")
        print(f"Log L2 range: {df_l2['Log_L2_Score'].min():.2f} to {df_l2['Log_L2_Score'].max():.2f}")
        
        # 1. L2 Scores by Algorithm - Original Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        algorithm_means = df_l2.groupby('Algorithm')['L2_Score'].mean().sort_values(ascending=True)
        algorithm_stds = df_l2.groupby('Algorithm')['L2_Score'].std()
        
        x_positions = range(len(algorithm_means))
        line = ax.errorbar(x_positions, algorithm_means.values, 
                          yerr=algorithm_stds[algorithm_means.index].values,
                          marker='o', linewidth=2, markersize=8, capsize=5, 
                          capthick=2, alpha=0.8, color='green')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions, algorithm_means.values, algorithm_stds[algorithm_means.index].values)):
            ax.annotate(f'{y:.2e}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('L2 Baseline Scores by Algorithm (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('L2 Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([alg.upper() for alg in algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_by_algorithm_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. L2 Scores by Algorithm - Log Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        log_algorithm_means = df_l2.groupby('Algorithm')['Log_L2_Score'].mean().sort_values(ascending=True)
        log_algorithm_stds = df_l2.groupby('Algorithm')['Log_L2_Score'].std()
        
        x_positions_log = range(len(log_algorithm_means))
        line_log = ax.errorbar(x_positions_log, log_algorithm_means.values, 
                              yerr=log_algorithm_stds[log_algorithm_means.index].values,
                              marker='s', linewidth=2, markersize=8, capsize=5, 
                              capthick=2, alpha=0.8, color='orange')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions_log, log_algorithm_means.values, log_algorithm_stds[log_algorithm_means.index].values)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('L2 Baseline Scores by Algorithm (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|L2 Score|)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions_log)
        ax.set_xticklabels([alg.upper() for alg in log_algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_by_algorithm_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. L2 Scores by Dataset - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        dataset_order = df_l2.groupby('Dataset')['L2_Score'].mean().sort_values(ascending=True).index
        
        bp = ax.boxplot([df_l2[df_l2['Dataset'] == dataset]['L2_Score'].values 
                        for dataset in dataset_order], 
                       labels=[dataset.upper() for dataset in dataset_order], patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('L2 Baseline Score Distribution by Dataset (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('L2 Score', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_by_dataset_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. L2 Scores by Dataset - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        log_dataset_order = df_l2.groupby('Dataset')['Log_L2_Score'].mean().sort_values(ascending=True).index
        
        bp_log = ax.boxplot([df_l2[df_l2['Dataset'] == dataset]['Log_L2_Score'].values 
                            for dataset in log_dataset_order], 
                           labels=[dataset.upper() for dataset in log_dataset_order], patch_artist=True)
        
        # Color the boxes
        colors_log = plt.cm.Set3(np.linspace(0, 1, len(log_dataset_order)))
        for patch, color in zip(bp_log['boxes'], colors_log):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('L2 Baseline Score Distribution by Dataset (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|L2 Score|)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_by_dataset_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. L2 Heatmap - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_orig = df_l2.groupby(['Algorithm', 'Dataset'])['L2_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_orig, annot=True, fmt='.2e', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'L2 Score'})
        ax.set_title('L2 Baseline Scores: Algorithm vs Dataset (Original Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_heatmap_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. L2 Heatmap - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_log = df_l2.groupby(['Algorithm', 'Dataset'])['Log_L2_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_log, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'Log₁₀(|L2 Score|)'})
        ax.set_title('L2 Baseline Scores: Algorithm vs Dataset (Log₁₀ Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(l2_dir, 'l2_baseline_heatmap_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save L2 summary data with both scales
        summary_df = df_l2.groupby(['Algorithm', 'Dataset']).agg({
            'L2_Score': ['mean', 'std', 'count'],
            'Log_L2_Score': ['mean', 'std']
        }).round(4)
        summary_df.to_csv(os.path.join(l2_dir, 'l2_baseline_summary.csv'))
        
        print(f"L2 baseline visualizations saved to: {l2_dir}")
        print("Generated files:")
        print("- l2_baseline_by_algorithm_original.png")
        print("- l2_baseline_by_algorithm_log.png")
        print("- l2_baseline_by_dataset_original.png")
        print("- l2_baseline_by_dataset_log.png")
        print("- l2_baseline_heatmap_original.png")
        print("- l2_baseline_heatmap_log.png")
        print("- l2_baseline_summary.csv")

    def create_l0_baseline_visualizations(self):
        """Create L0 (L0 Distance) baseline visualizations with log scale"""
        print("Creating L0 baseline visualizations...")
        
        # Check if L0 data exists
        l0_data = []
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                if (algorithm in self.baseline_results and 
                    dataset in self.baseline_results[algorithm] and
                    'L0' in self.baseline_results[algorithm][dataset]):
                    l0_values = self.baseline_results[algorithm][dataset]['L0']
                    if l0_values:
                        for l0_val in l0_values:
                            # Convert to log scale to handle extreme values
                            log_l0 = np.log10(abs(l0_val)) if l0_val != 0 else 0
                            l0_data.append({
                                'Algorithm': algorithm,
                                'Dataset': dataset,
                                'L0_Score': l0_val,
                                'Log_L0_Score': log_l0
                            })
        
        if not l0_data:
            print("No L0 baseline data found. Skipping L0 visualizations.")
            return
        
        # Create output directory
        l0_dir = os.path.join(self.logs_dir, 'img', 'L0_BASELINE')
        os.makedirs(l0_dir, exist_ok=True)
        
        df_l0 = pd.DataFrame(l0_data)
        print(f"L0 data summary:")
        print(f"Original L0 range: {df_l0['L0_Score'].min():.2f} to {df_l0['L0_Score'].max():.2f}")
        print(f"Log L0 range: {df_l0['Log_L0_Score'].min():.2f} to {df_l0['Log_L0_Score'].max():.2f}")
        
        # 1. L0 Scores by Algorithm - Original Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        algorithm_means = df_l0.groupby('Algorithm')['L0_Score'].mean().sort_values(ascending=True)
        algorithm_stds = df_l0.groupby('Algorithm')['L0_Score'].std()
        
        x_positions = range(len(algorithm_means))
        line = ax.errorbar(x_positions, algorithm_means.values, 
                          yerr=algorithm_stds[algorithm_means.index].values,
                          marker='o', linewidth=2, markersize=8, capsize=5, 
                          capthick=2, alpha=0.8, color='purple')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions, algorithm_means.values, algorithm_stds[algorithm_means.index].values)):
            ax.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('L0 Baseline Scores by Algorithm (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('L0 Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([alg.upper() for alg in algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_by_algorithm_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. L0 Scores by Algorithm - Log Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        log_algorithm_means = df_l0.groupby('Algorithm')['Log_L0_Score'].mean().sort_values(ascending=True)
        log_algorithm_stds = df_l0.groupby('Algorithm')['Log_L0_Score'].std()
        
        x_positions_log = range(len(log_algorithm_means))
        line_log = ax.errorbar(x_positions_log, log_algorithm_means.values, 
                              yerr=log_algorithm_stds[log_algorithm_means.index].values,
                              marker='s', linewidth=2, markersize=8, capsize=5, 
                              capthick=2, alpha=0.8, color='green')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions_log, log_algorithm_means.values, log_algorithm_stds[log_algorithm_means.index].values)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('L0 Baseline Scores by Algorithm (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|L0 Score|)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions_log)
        ax.set_xticklabels([alg.upper() for alg in log_algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_by_algorithm_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. L0 Scores by Dataset - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        dataset_order = df_l0.groupby('Dataset')['L0_Score'].mean().sort_values(ascending=True).index
        
        bp = ax.boxplot([df_l0[df_l0['Dataset'] == dataset]['L0_Score'].values 
                        for dataset in dataset_order], 
                       labels=[dataset.upper() for dataset in dataset_order], patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('L0 Baseline Score Distribution by Dataset (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('L0 Score', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_by_dataset_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. L0 Scores by Dataset - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        log_dataset_order = df_l0.groupby('Dataset')['Log_L0_Score'].mean().sort_values(ascending=True).index
        
        bp_log = ax.boxplot([df_l0[df_l0['Dataset'] == dataset]['Log_L0_Score'].values 
                            for dataset in log_dataset_order], 
                           labels=[dataset.upper() for dataset in log_dataset_order], patch_artist=True)
        
        # Color the boxes
        colors_log = plt.cm.Set3(np.linspace(0, 1, len(log_dataset_order)))
        for patch, color in zip(bp_log['boxes'], colors_log):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('L0 Baseline Score Distribution by Dataset (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|L0 Score|)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_by_dataset_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. L0 Heatmap - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_orig = df_l0.groupby(['Algorithm', 'Dataset'])['L0_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_orig, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'L0 Score'})
        ax.set_title('L0 Baseline Scores: Algorithm vs Dataset (Original Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_heatmap_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. L0 Heatmap - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_log = df_l0.groupby(['Algorithm', 'Dataset'])['Log_L0_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_log, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'Log₁₀(|L0 Score|)'})
        ax.set_title('L0 Baseline Scores: Algorithm vs Dataset (Log₁₀ Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(l0_dir, 'l0_baseline_heatmap_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save L0 summary data with both scales
        summary_df = df_l0.groupby(['Algorithm', 'Dataset']).agg({
            'L0_Score': ['mean', 'std', 'count'],
            'Log_L0_Score': ['mean', 'std']
        }).round(4)
        summary_df.to_csv(os.path.join(l0_dir, 'l0_baseline_summary.csv'))
        
        print(f"L0 baseline visualizations saved to: {l0_dir}")
        print("Generated files:")
        print("- l0_baseline_by_algorithm_original.png")
        print("- l0_baseline_by_algorithm_log.png")
        print("- l0_baseline_by_dataset_original.png")
        print("- l0_baseline_by_dataset_log.png")
        print("- l0_baseline_heatmap_original.png")
        print("- l0_baseline_heatmap_log.png")
        print("- l0_baseline_summary.csv")
        """Create line plots showing algorithm performance across perturbation values"""
        print("Creating perturbation value plots...")
        
        available_metrics = ['validity', 'accuracy', 'l0', 'l2']  # All available metrics
        
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
        
        # Create plots for model perturbations (only for metrics that are available in model perturbations)
        model_metrics = ['validity', 'accuracy']  # Model perturbations typically only have these
        for perturbation_group, values in model_perturbation_mappings.items():
            for dataset in self.datasets.values():
                for metric in model_metrics:
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
            print(f"No data found for {perturbation_type} - {dataset} - {metric}")
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
        ax.set_ylabel(self._get_metric_ylabel(metric), fontsize=14, fontweight='bold')
        ax.set_title(f'{perturbation_type.replace("_", " ").title()} - {dataset} ({metric.upper()})', 
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
        metric_dir = os.path.join(img_dir, metric.upper())
        os.makedirs(metric_dir, exist_ok=True)
        filename = f'{perturbation_type}_{dataset.replace(" ", "_")}_{metric}_vs_values.png'
        plt.savefig(os.path.join(metric_dir, filename), dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated: {metric.upper()}/{filename}")
    
    def _create_model_perturbation_value_plot(self, perturbation_group, dataset, metric, values):
        """Create a line plot for model perturbation showing parameter values on x-axis"""
        
        # Only use DiCE algorithm for model perturburbations
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
        ax.set_ylabel(self._get_metric_ylabel(metric), fontsize=14, fontweight='bold')
        ax.set_title(f'Model Perturbation: {perturbation_group.replace("_", " ").title()} - {dataset} (DiCE) ({metric.upper()})', 
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
        metric_dir = os.path.join(img_dir, metric.upper())
        os.makedirs(metric_dir, exist_ok=True)
        filename = f'model_{perturbation_group}_{dataset.replace(" ", "_")}_{metric}_vs_values.png'
        plt.savefig(os.path.join(metric_dir, filename), dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"Generated: {metric.upper()}/{filename}")
    
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
    
    def _get_metric_ylabel(self, metric):
        """Get appropriate y-axis label for metric type"""
        if metric == 'validity':
            return 'Validity Score'
        elif metric == 'accuracy':
            return 'Accuracy Score'
        elif metric == 'l0':
            return 'L0 Distance'
        elif metric == 'l2':
            return 'L2 Distance'
        else:
            return f'{metric.upper()} Score'

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
        general_dir = os.path.join(img_dir, 'GENERAL')
        os.makedirs(general_dir, exist_ok=True)
        plt.savefig(os.path.join(general_dir, 'data_perturbation_line_plots.png'), 
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
        general_dir = os.path.join(img_dir, 'GENERAL')
        os.makedirs(general_dir, exist_ok=True)
        plt.savefig(os.path.join(general_dir, 'data_perturbation_bars.png'), 
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
        general_dir = os.path.join(img_dir, 'GENERAL')
        os.makedirs(general_dir, exist_ok=True)
        plt.savefig(os.path.join(general_dir, 'model_perturbation_line_plots.png'), 
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
        plt.savefig(os.path.join(self.logs_dir, 'img', 'GENERAL', 'model_perturbation_bars.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_lof_baseline_visualizations(self):
        """Create LOF (Local Outlier Factor) baseline visualizations with log scale"""
        print("Creating LOF baseline visualizations...")
        
        # Check if LOF data exists
        lof_data = []
        for algorithm in self.algorithms:
            for dataset in self.datasets.values():
                if (algorithm in self.baseline_results and 
                    dataset in self.baseline_results[algorithm] and
                    'LOF' in self.baseline_results[algorithm][dataset]):
                    lof_values = self.baseline_results[algorithm][dataset]['LOF']
                    if lof_values:
                        for lof_val in lof_values:
                            # Convert to log scale to handle extreme values
                            # Use absolute value for log transformation since LOF can be negative
                            log_lof = np.log10(abs(lof_val)) if lof_val != 0 else 0
                            lof_data.append({
                                'Algorithm': algorithm,
                                'Dataset': dataset,
                                'LOF_Score': lof_val,
                                'Log_LOF_Score': log_lof
                            })
        
        if not lof_data:
            print("No LOF baseline data found. Skipping LOF visualizations.")
            return
        
        # Create output directory
        lof_dir = os.path.join(self.logs_dir, 'img', 'LOF_BASELINE')
        os.makedirs(lof_dir, exist_ok=True)
        
        df_lof = pd.DataFrame(lof_data)
        print(f"LOF data summary:")
        print(f"Original LOF range: {df_lof['LOF_Score'].min():.2f} to {df_lof['LOF_Score'].max():.2f}")
        print(f"Log LOF range: {df_lof['Log_LOF_Score'].min():.2f} to {df_lof['Log_LOF_Score'].max():.2f}")
        
        # 1. LOF Scores by Algorithm - Original Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        algorithm_means = df_lof.groupby('Algorithm')['LOF_Score'].mean().sort_values(ascending=True)
        algorithm_stds = df_lof.groupby('Algorithm')['LOF_Score'].std()
        
        x_positions = range(len(algorithm_means))
        line = ax.errorbar(x_positions, algorithm_means.values, 
                          yerr=algorithm_stds[algorithm_means.index].values,
                          marker='o', linewidth=2, markersize=8, capsize=5, 
                          capthick=2, alpha=0.8, color='blue')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions, algorithm_means.values, algorithm_stds[algorithm_means.index].values)):
            ax.annotate(f'{y:.2e}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('LOF Baseline Scores by Algorithm (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('LOF Score', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels([alg.upper() for alg in algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_by_algorithm_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. LOF Scores by Algorithm - Log Scale (Line Plot)
        fig, ax = plt.subplots(figsize=(12, 8))
        log_algorithm_means = df_lof.groupby('Algorithm')['Log_LOF_Score'].mean().sort_values(ascending=True)
        log_algorithm_stds = df_lof.groupby('Algorithm')['Log_LOF_Score'].std()
        
        x_positions_log = range(len(log_algorithm_means))
        line_log = ax.errorbar(x_positions_log, log_algorithm_means.values, 
                              yerr=log_algorithm_stds[log_algorithm_means.index].values,
                              marker='s', linewidth=2, markersize=8, capsize=5, 
                              capthick=2, alpha=0.8, color='red')
        
        # Add value labels near each marker
        for i, (x, y, std) in enumerate(zip(x_positions_log, log_algorithm_means.values, log_algorithm_stds[log_algorithm_means.index].values)):
            ax.annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                       xytext=(0,15), ha='center', fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title('LOF Baseline Scores by Algorithm (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|LOF Score|)', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions_log)
        ax.set_xticklabels([alg.upper() for alg in log_algorithm_means.index], rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_by_algorithm_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. LOF Scores by Dataset - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        dataset_order = df_lof.groupby('Dataset')['LOF_Score'].mean().sort_values(ascending=True).index
        
        bp = ax.boxplot([df_lof[df_lof['Dataset'] == dataset]['LOF_Score'].values 
                        for dataset in dataset_order], 
                       labels=[dataset.upper() for dataset in dataset_order], patch_artist=True)
        
        # Color the boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(dataset_order)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('LOF Baseline Score Distribution by Dataset (Original Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('LOF Score', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_by_dataset_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. LOF Scores by Dataset - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        log_dataset_order = df_lof.groupby('Dataset')['Log_LOF_Score'].mean().sort_values(ascending=True).index
        
        bp_log = ax.boxplot([df_lof[df_lof['Dataset'] == dataset]['Log_LOF_Score'].values 
                            for dataset in log_dataset_order], 
                           labels=[dataset.upper() for dataset in log_dataset_order], patch_artist=True)
        
        # Color the boxes
        colors_log = plt.cm.Set3(np.linspace(0, 1, len(log_dataset_order)))
        for patch, color in zip(bp_log['boxes'], colors_log):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('LOF Baseline Score Distribution by Dataset (Log₁₀ Scale)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Log₁₀(|LOF Score|)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_by_dataset_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. LOF Heatmap - Original Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_orig = df_lof.groupby(['Algorithm', 'Dataset'])['LOF_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_orig, annot=True, fmt='.2e', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'LOF Score'})
        ax.set_title('LOF Baseline Scores: Algorithm vs Dataset (Original Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_heatmap_original.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # 6. LOF Heatmap - Log Scale
        fig, ax = plt.subplots(figsize=(12, 8))
        pivot_df_log = df_lof.groupby(['Algorithm', 'Dataset'])['Log_LOF_Score'].mean().unstack()
        
        sns.heatmap(pivot_df_log, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax, cbar_kws={'label': 'Log₁₀(|LOF Score|)'})
        ax.set_title('LOF Baseline Scores: Algorithm vs Dataset (Log₁₀ Scale)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
        ax.set_ylabel('Algorithm', fontsize=14, fontweight='bold')
        
        # Update labels to uppercase
        ax.set_xticklabels([label.get_text().upper() for label in ax.get_xticklabels()])
        ax.set_yticklabels([label.get_text().upper() for label in ax.get_yticklabels()])
        
        plt.tight_layout()
        plt.savefig(os.path.join(lof_dir, 'lof_baseline_heatmap_log.png'), 
                    dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save LOF summary data with both scales
        summary_df = df_lof.groupby(['Algorithm', 'Dataset']).agg({
            'LOF_Score': ['mean', 'std', 'count'],
            'Log_LOF_Score': ['mean', 'std']
        }).round(4)
        summary_df.to_csv(os.path.join(lof_dir, 'lof_baseline_summary.csv'))
        
        print(f"LOF baseline visualizations saved to: {lof_dir}")
        print("Generated files:")
        print("- lof_baseline_by_algorithm_original.png")
        print("- lof_baseline_by_algorithm_log.png")
        print("- lof_baseline_by_dataset_original.png")
        print("- lof_baseline_by_dataset_log.png")
        print("- lof_baseline_heatmap_original.png")
        print("- lof_baseline_heatmap_log.png")
        print("- lof_baseline_summary.csv")
        df_lof.to_csv(os.path.join(lof_dir, 'lof_baseline_summary.csv'), index=False)
        print(f"LOF summary saved to {os.path.join(lof_dir, 'lof_baseline_summary.csv')}")
    
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
                    robustness = alg_data['Overall_Robustness'].values;
                    
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
            plt.savefig(os.path.join(self.logs_dir, 'img', 'GENERAL', 'comprehensive_summary.png'), 
                        dpi=300, bbox_inches='tight')
            plt.show()
            
            # Save summary table
            df_summary.to_csv(os.path.join(self.logs_dir, 'robustness_summary.csv'), index=False)
            print(f"Summary data saved to {os.path.join(self.logs_dir, 'robustness_summary.csv')}")
    
    def print_algorithm_coverage(self):
        """Print a summary of which algorithms and datasets have available data"""
        print("\n" + "=" * 60)
        print("ALGORITHM AND DATASET COVERAGE SUMMARY")
        print("=" * 60)
        
        # Check for log files
        print("\nLog File Availability:")
        print("-" * 30)
        
        coverage_matrix = {}
        for algorithm in self.algorithms:
            coverage_matrix[algorithm] = {}
            for version, dataset in self.datasets.items():
                # Check the same logic as in parse_log_files
                log_file = None
                
                if algorithm == 'OCEAN':
                    if version == 'v4':
                        log_candidates = [f"ocean_{version}_heloc.log"]
                    elif version == 'v5':
                        log_candidates = [f"ocean_{version}_compas.log"]
                    else:
                        log_candidates = [f"ocean_{version}.log"]
                else:
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
                        break
                
                coverage_matrix[algorithm][dataset] = "✓" if log_file else "✗"
        
        # Print coverage matrix
        header = f"{'Algorithm':<15}"
        for dataset in self.datasets.values():
            header += f"{dataset:<15}"
        print(header)
        print("-" * len(header))
        
        for algorithm in self.algorithms:
            row = f"{algorithm:<15}"
            for dataset in self.datasets.values():
                row += f"{coverage_matrix[algorithm][dataset]:<15}"
            print(row)
        
        # Special note for OCEAN
        print(f"\nSpecial Notes:")
        print(f"- OCEAN currently has log data for German Credit dataset")
        print(f"- OCEAN integration is ready for future datasets (HELOC, COMPAS, Spambase)")
        print(f"- OCEAN uses a different log format which is automatically detected and parsed")
        
        # Check parsed data availability
        print(f"\nParsed Data Summary:")
        print("-" * 30)
        
        for algorithm in self.algorithms:
            if algorithm in self.baseline_results:
                datasets_with_data = list(self.baseline_results[algorithm].keys())
                if datasets_with_data:
                    print(f"{algorithm}: {', '.join(datasets_with_data)}")
                else:
                    print(f"{algorithm}: No parsed data")
            else:
                print(f"{algorithm}: No parsed data")
        
        print("=" * 60)

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("=" * 80)
        print("COUNTERFACTUAL ROBUSTNESS VISUALIZATION ANALYSIS")
        print("=" * 80)
        
        # Parse all log files
        self.parse_log_files()
        
        # Print coverage summary (helpful for understanding OCEAN availability)
        self.print_algorithm_coverage()
        
        # Create baseline visualizations for all metrics
        self.create_lof_baseline_visualizations()
        self.create_l2_baseline_visualizations()  
        self.create_l0_baseline_visualizations()
        
        # Skip traditional visualizations as requested
        # self.create_data_perturbation_visualizations()
        # self.create_model_perturbation_visualizations()
        # self.create_comprehensive_summary()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("Generated baseline visualizations organized by metric:")
        print("- img/LOF_BASELINE/: LOF baseline comparison plots")
        print("- img/L2_BASELINE/: L2 distance baseline comparison plots") 
        print("- img/L0_BASELINE/: L0 distance baseline comparison plots")
        print("\nNOTE: OCEAN data is included in all applicable visualizations")
        print("All plots generated as separate figures with both original and log scales")
        print("=" * 80)
def main():
    """Main function to run the analysis"""
    print("Starting CFRobustness analysis...")
    
    try:
        # Get the current directory (logs folder)
        logs_directory = os.path.dirname(os.path.abspath(__file__))
        print(f"Logs directory: {logs_directory}")
        
        # Create analyzer instance
        print("Creating analyzer instance...")
        analyzer = CFRobustnessAnalyzer(logs_directory)
        
        # Run complete analysis
        print("Running complete analysis...")
        analyzer.run_complete_analysis()
        print("Analysis completed!")
    except Exception as e:
        print(f"ERROR in main: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Script started - calling main()")
    main()
    print("Script completed")

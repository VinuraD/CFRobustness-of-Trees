"""
Consolidated Data Reader for CF Robustness Analysis

This file contains unified classes to read and process:
1. All_Baseline_values.xlsx file (baseline metrics)
2. All dataset experiment files (German Credit, Spambase, COMPAS, HELOC)

Handles both separate files (German Credit) and combined files (other datasets)
"""

import pandas as pd
import numpy as np
import os
import re

class BaselineDataReader:
    """Class to read and process the All_Baseline_values.xlsx file"""
    
    def __init__(self, file_path="summary_tables/All_Baseline_values.xlsx"):
        self.file_path = file_path
        self.datasets = ['German Credit', 'Spambase', 'COMPAS', 'HELOC']
        self.methods = ['NICE', 'DiCE', 'OCEAN', 'cfxplorer', 'Feature Tweak', 'CEML']
        self.metrics = ['Model Accuracy', 'Success Rate', 'Validity', 'L2 Distance', 'L0 Distance', 'LOF Score']
        self.raw_data = None
        self.structured_data = None
        self.summary_df = None

    def load_data(self):
        """Load and parse the baseline values Excel file"""
        try:
            # Read the Excel file
            self.raw_data = pd.read_excel(self.file_path)
            print(f"✓ Loaded baseline data file: {self.file_path}")
            print(f"  Shape: {self.raw_data.shape}")
            
            # Parse the structure
            self._parse_baseline_structure()
            
            # Create summary DataFrame
            self._create_summary_dataframe()
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading baseline data: {e}")
            return False
    
    def _parse_baseline_structure(self):
        """Parse the structure of the baseline values file"""
        print(f"\n📊 Parsing baseline file structure...")
        
        # Initialize structured data storage
        self.structured_data = {}
        
        # The file has datasets in columns and metrics in rows
        # Each dataset has 6 columns (one per method)
        col_idx = 0
        
        for dataset in self.datasets:
            self.structured_data[dataset] = {}
            
            # Each dataset should have 6 methods
            for method in self.methods:
                if col_idx < len(self.raw_data.columns):
                    self.structured_data[dataset][method] = {}
                    
                    # Each method has 6 metrics (rows)
                    for metric_idx, metric in enumerate(self.metrics):
                        if metric_idx < len(self.raw_data):
                            value = self.raw_data.iloc[metric_idx, col_idx]
                            self.structured_data[dataset][method][metric] = value
                    
                    col_idx += 1
        
        print(f"✓ Parsed data for {len(self.structured_data)} datasets")
    
    def _create_summary_dataframe(self):
        """Create a summary DataFrame from structured data"""
        rows = []
        
        for dataset, dataset_data in self.structured_data.items():
            for method, method_data in dataset_data.items():
                for metric, value in method_data.items():
                    rows.append({
                        'Dataset': dataset,
                        'Method': method,
                        'Metric': metric,
                        'Value': value
                    })
        
        self.summary_df = pd.DataFrame(rows)
        print(f"✓ Created summary DataFrame with {len(rows)} entries")
    
    def get_method_data(self, dataset, method):
        """Get all metrics for a specific method and dataset"""
        if dataset in self.structured_data and method in self.structured_data[dataset]:
            return self.structured_data[dataset][method]
        else:
            return None
    
    def display_summary(self):
        """Display a formatted summary of the baseline data"""
        if self.summary_df is None:
            print("No data loaded. Call load_data() first.")
            return
            
        print("\n" + "="*100)
        print("BASELINE VALUES SUMMARY")
        print("="*100)
        
        print(f"Datasets: {self.datasets}")
        print(f"Methods: {self.methods}")
        print(f"Metrics: {self.metrics}")
        
        print(f"\nData completeness:")
        for dataset in self.datasets:
            if dataset in self.structured_data:
                available_methods = list(self.structured_data[dataset].keys())
                print(f"  {dataset}: {len(available_methods)} methods available")
        
        # Show some example values
        print(f"\nExample values (German Credit dataset):")
        if 'German Credit' in self.structured_data:
            for method in list(self.structured_data['German Credit'].keys())[:3]:
                print(f"  {method}:")
                for metric in self.metrics[:3]:
                    value = self.structured_data['German Credit'][method].get(metric, 'N/A')
                    print(f"    {metric}: {value}")
    
    def save_summary_csv(self, output_path="baseline_values_summary.csv", save_file=False):
        """Save the summary data to CSV format (optional)"""
        if self.summary_df is not None:
            if save_file:
                self.summary_df.to_csv(output_path, index=False)
                print(f"✓ Summary saved to {output_path}")
            else:
                print(f"📊 Summary ready (not saved to file). Call with save_file=True to export.")
        else:
            print("✗ No summary data available. Load data first.")


class ExperimentDataReader:
    """Class to read and process dataset experiment files for all datasets
    
    Handles both single files (Spambase, COMPAS, HELOC) and separate files (German Credit)
    """
    
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        # For model perturbations: DiCE, NICE, Feature Tweak, CEML (4 methods)
        self.model_target_methods = ['DICE', 'NICE', 'FEATURE TWEAK', 'CEML']
        
        # For data perturbations: DiCE, NICE, FOCUS (cfxplorer), OCEAN, Feature Tweak, CEML (6 methods) 
        self.data_target_methods = ['DICE', 'NICE', 'FOCUS', 'OCEAN', 'FEATURE TWEAK', 'CEML']
        self.model_data = None
        self.data_data = None
        self.structured_data = None
        
        # Determine file paths based on dataset
        if dataset_name.lower() in ['german_credit', 'spambase', 'heloc', 'compas']:
            # Use separate files for these datasets
            dataset_prefix = {
                'german_credit': 'German_Credit',
                'spambase': 'Spambase', 
                'heloc': 'Heloc',
                'compas': 'Compas'
            }[dataset_name.lower()]
            
            self.model_file = f"summary_tables/{dataset_prefix}_model_perturb.xlsx"
            self.data_file = f"summary_tables/{dataset_prefix}_data_perturb.xlsx"
            self.is_separate_files = True
        else:
            # Single file for other datasets
            file_mapping = {}
            self.single_file = file_mapping.get(dataset_name.lower())
            self.is_separate_files = False
    
    def load_and_analyze(self):
        """Load and parse the dataset files based on type"""
        if self.is_separate_files:
            return self._load_separate_files()
        else:
            return self._load_single_file()
    
    def _load_separate_files(self):
        """Load and parse separate model and data perturbation files (German Credit)"""
        try:
            print(f"📂 Loading separate files for {self.dataset_name}...")
            
            # Parse model perturbations
            print(f"\n🔍 PARSING MODEL PERTURBATIONS")
            print("-" * 40)
            self.model_data = self._parse_perturbation_file(self.model_file, "Model")
            
            # Parse data perturbations
            print(f"\n🔍 PARSING DATA PERTURBATIONS")
            print("-" * 40)
            self.data_data = self._parse_perturbation_file(self.data_file, "Data")
            
            return True
            
        except Exception as e:
            print(f"✗ Error loading separate files: {e}")
            return False
    
    def _load_single_file(self):
        """Load and parse single file with two tables (Spambase, COMPAS, HELOC)"""
        try:
            print(f"📂 Loading single file for {self.dataset_name}: {self.single_file}")
            
            df = pd.read_excel(self.single_file)
            print(f"✓ Loaded file with shape: {df.shape}")
            
            # Split into model and data perturbation tables
            model_df, data_df = self._split_combined_file(df)
            
            if model_df is not None:
                print(f"\n🔍 PARSING MODEL PERTURBATIONS FROM COMBINED FILE")
                print("-" * 50)
                self.model_data = self._parse_table_data(model_df, "Model")
            
            if data_df is not None:
                print(f"\n🔍 PARSING DATA PERTURBATIONS FROM COMBINED FILE")
                print("-" * 50)
                self.data_data = self._parse_table_data(data_df, "Data")
                
            return True
            
        except Exception as e:
            print(f"✗ Error loading single file: {e}")
            return False
    
    def _parse_perturbation_file(self, file_path, perturbation_type):
        """Parse a single perturbation file"""
        try:
            df = pd.read_excel(file_path)
            print(f"✓ Loaded {perturbation_type} file: {os.path.basename(file_path)}")
            print(f"  Shape: {df.shape}")
            
            # Check if this is data perturbations (has bin structure) or model perturbations
            if perturbation_type == "Data":
                return self._parse_data_perturbations_with_bins(df)
            else:
                return self._parse_model_perturbations_simple(df)
                
        except Exception as e:
            print(f"✗ Error parsing {file_path}: {e}")
            return None
    
    def _parse_data_perturbations_with_bins(self, df):
        """Parse data perturbations file with bin structure"""
        print("  Parsing data perturbations with bin structure...")
        
        results = {}
        
        # Based on the examination: 
        # Column 0: Perturbation type, Column 1: Bin, Column 2: Description
        # Columns 3-6: NICE (Mean Validity, Std Validity, Mean Accuracy, Std Accuracy)  
        # Columns 7-10: DiCE
        # Columns 11-14: cfxplorer  
        # Columns 15-18: OCEAN
        # Columns 19-22: Feature Tweak
        # Columns 23-26: CEML
        method_col_mapping = {
            'NICE': 3,      # Columns 3-6
            'DICE': 7,      # Columns 7-10
            'FOCUS': 11,    # Columns 11-14 (previously cfxplorer)
            'OCEAN': 15,    # Columns 15-18
            'FEATURE TWEAK': 19,  # Columns 19-22
            'CEML': 23      # Columns 23-26
        }
        
        for method in self.data_target_methods:
            if method in method_col_mapping:
                col_start = method_col_mapping[method]
                method_data = {}
                
                print(f"\n    {method} (columns {col_start}-{col_start+3}):")
                
                # Group rows by perturbation type
                perturbation_groups = {}
                current_perturbation = None
                
                for row_idx in range(1, len(df)):  # Skip header row
                    config_val = df.iloc[row_idx, 0]  # Configuration column
                    
                    if pd.notna(config_val) and isinstance(config_val, str):
                        if config_val in ['minor_deletion', 'major_deletion', 'minor_addition', 'major_addition']:
                            current_perturbation = config_val
                            if current_perturbation not in perturbation_groups:
                                perturbation_groups[current_perturbation] = []
                    
                    if current_perturbation:
                        bin_val = df.iloc[row_idx, 1] if row_idx < len(df) and pd.notna(df.iloc[row_idx, 1]) else None
                        
                        if bin_val is not None:
                            mean_validity = self._safe_float(df.iloc[row_idx, col_start])
                            std_validity = self._safe_float(df.iloc[row_idx, col_start + 1])
                            mean_accuracy = self._safe_float(df.iloc[row_idx, col_start + 2])
                            std_accuracy = self._safe_float(df.iloc[row_idx, col_start + 3])
                            
                            perturbation_groups[current_perturbation].append({
                                'bin': bin_val,
                                'mean_validity': mean_validity,
                                'std_validity': std_validity,
                                'mean_accuracy': mean_accuracy,
                                'std_accuracy': std_accuracy
                            })
                
                # Store the grouped data
                for perturbation, bin_data in perturbation_groups.items():
                    method_data[perturbation] = sorted(bin_data, key=lambda x: x['bin'])
                    print(f"      {perturbation}: {len(bin_data)} bins")
                
                results[method] = method_data
        
        return results
    
    def _parse_model_perturbations_simple(self, df):
        """Parse model perturbations file (original simple structure)"""
        print("  Parsing model perturbations with simple structure...")
        
        # Extract configurations (skip header row)
        configs = df.iloc[1:, 0].dropna().tolist()
        print(f"  Configurations ({len(configs)}): {configs[:3]}..." if len(configs) > 3 else f"  Configurations: {configs}")
        
        # Explicit column mapping based on file structure
        # NICE: columns 1-4, DiCE: columns 5-8, Feature Tweak: columns 9-12, CEML: columns 13-16
        method_col_mapping = {
            'NICE': 1,      # Columns 1-4
            'DICE': 5,      # Columns 5-8
            'FEATURE TWEAK': 9,   # Columns 9-12
            'CEML': 13      # Columns 13-16
        }
        
        # Parse method data
        results = {}
        
        for method in self.model_target_methods:
            if method in method_col_mapping:
                col_idx = method_col_mapping[method]
                
                if col_idx + 3 < len(df.columns):  # Need 4 columns for each method
                    method_data = {}
                    
                    print(f"\n    {method} (columns {col_idx}-{col_idx+3}):")
                    
                    # Parse data for each configuration
                    valid_count = 0
                    missing_count = 0
                    
                    for i, config in enumerate(configs):
                        row_idx = i + 1  # +1 because we skip header row
                        if row_idx < len(df):
                            mean_validity = self._safe_float(df.iloc[row_idx, col_idx])
                            std_validity = self._safe_float(df.iloc[row_idx, col_idx + 1])
                            mean_accuracy = self._safe_float(df.iloc[row_idx, col_idx + 2])
                            std_accuracy = self._safe_float(df.iloc[row_idx, col_idx + 3])
                            
                            method_data[config] = {
                                'mean_validity': mean_validity,
                                'std_validity': std_validity,
                                'mean_accuracy': mean_accuracy,
                                'std_accuracy': std_accuracy
                            }
                            
                            if pd.notna(mean_validity):
                                valid_count += 1
                            else:
                                missing_count += 1
                
                    results[method] = method_data
                    print(f"      ✓ {valid_count} valid entries, {missing_count} missing")
                    
                    # Validation: Check validity range
                    valid_validities = [v['mean_validity'] for v in method_data.values() if pd.notna(v['mean_validity'])]
                    if valid_validities:
                        max_validity = max(valid_validities)
                        min_validity = min(valid_validities)
                        if max_validity > 1.0:
                            print(f"      ⚠️  WARNING: Max validity {max_validity:.3f} > 1.0")
                        else:
                            print(f"      ✓ Validity range: {min_validity:.3f} - {max_validity:.3f}")
                else:
                    print(f"      ⚠️  {method}: Not enough columns remaining")
            else:
                print(f"      ⚠️  {method}: Column mapping not found")
        
        return results
    
    def _split_combined_file(self, df):
        """Split a combined file into model and data perturbation tables"""
        print(f"🔍 Analyzing combined file structure...")
        print(f"  Shape: {df.shape}")
        
        # Look for section indicators
        model_start = None
        data_start = None
        
        for i in range(len(df)):
            row_str = ' '.join([str(val) for val in df.iloc[i].tolist() if pd.notna(val)]).lower()
            if 'model' in row_str and 'perturbation' in row_str:
                model_start = i
                print(f"  Found Model Perturbations at row {i}")
            elif 'data' in row_str and 'perturbation' in row_str:
                data_start = i
                print(f"  Found Data Perturbations at row {i}")
        
        model_df = None
        data_df = None
        
        if model_start is not None:
            if data_start is not None:
                # Both sections found - split appropriately
                model_df = df.iloc[model_start:data_start].copy()
                data_df = df.iloc[data_start:].copy()
            else:
                # Only model section found
                model_df = df.iloc[model_start:].copy()
                
        elif data_start is not None:
            # Only data section found
            data_df = df.iloc[data_start:].copy()
        else:
            # Try to split by empty rows or other heuristics
            print("  No clear section indicators found, using heuristic splitting...")
            # Implementation for heuristic splitting if needed
            
        return model_df, data_df
    
    def _parse_table_data(self, df, perturbation_type):
        """Parse a table from a combined file"""
        if df is None or df.empty:
            print(f"  No data available for {perturbation_type} perturbations")
            return None
            
        # Reset index and clean up
        df = df.reset_index(drop=True)
        
        # Find the actual header row (look for configuration-like patterns)
        header_row = None
        for i in range(min(5, len(df))):  # Check first 5 rows
            row_vals = [str(val) for val in df.iloc[i].tolist() if pd.notna(val)]
            if any('config' in str(val).lower() or 'setting' in str(val).lower() for val in row_vals):
                header_row = i
                break
        
        if header_row is None:
            header_row = 0  # Default to first row
            
        print(f"  Using row {header_row} as header")
        
        # Extract configurations
        config_col = 0  # Assume first column contains configurations
        configs = []
        
        for i in range(header_row + 1, len(df)):
            config_val = df.iloc[i, config_col]
            if pd.notna(config_val) and str(config_val).strip():
                configs.append(str(config_val).strip())
            elif len(configs) > 0:  # Stop at first empty after we've started collecting
                break
                
        print(f"  Found {len(configs)} configurations: {configs[:3]}..." if len(configs) > 3 else f"  Configurations: {configs}")
        
        # Parse method data (similar to separate file logic)
        results = {}
        col_idx = 1  # Start after configuration column
        
        for method in self.target_methods:
            if col_idx + 3 < len(df.columns):
                method_data = {}
                
                print(f"\n    {method} (columns {col_idx}-{col_idx+3}):")
                
                valid_count = 0
                missing_count = 0
                
                for i, config in enumerate(configs):
                    row_idx = header_row + 1 + i
                    if row_idx < len(df):
                        mean_validity = self._safe_float(df.iloc[row_idx, col_idx])
                        std_validity = self._safe_float(df.iloc[row_idx, col_idx + 1])
                        mean_accuracy = self._safe_float(df.iloc[row_idx, col_idx + 2])
                        std_accuracy = self._safe_float(df.iloc[row_idx, col_idx + 3])
                        
                        method_data[config] = {
                            'mean_validity': mean_validity,
                            'std_validity': std_validity,
                            'mean_accuracy': mean_accuracy,
                            'std_accuracy': std_accuracy
                        }
                        
                        if pd.notna(mean_validity):
                            valid_count += 1
                        else:
                            missing_count += 1
                
                results[method] = method_data
                print(f"      ✓ {valid_count} valid entries, {missing_count} missing")
                
                # Validation
                valid_validities = [v['mean_validity'] for v in method_data.values() if pd.notna(v['mean_validity'])]
                if valid_validities:
                    max_validity = max(valid_validities)
                    min_validity = min(valid_validities)
                    if max_validity > 1.0:
                        print(f"      ⚠️  WARNING: Max validity {max_validity:.3f} > 1.0")
                    else:
                        print(f"      ✓ Validity range: {min_validity:.3f} - {max_validity:.3f}")
                
                col_idx += 4
            else:
                print(f"      ⚠️  {method}: Not enough columns remaining")
                break
                
        return results
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        if pd.isna(value):
            return np.nan
        try:
            return float(value)
        except (ValueError, TypeError):
            return np.nan
    
    def export_to_csv(self, output_dir="summary_tables", save_files=False):
        """Export parsed data to CSV files (optional)"""
        saved_files = []
        
        # Create DataFrames (always done for validation)
        if self.model_data:
            model_df = self._convert_to_dataframe(self.model_data, "Model")
            model_file = os.path.join(output_dir, f"{self.dataset_name}_model_perturbations.csv")
            
            if save_files:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_df.to_csv(model_file, index=False)
                print(f"✓ Saved model perturbations: {model_file}")
            
            saved_files.append(("Model", model_df, model_file if save_files else None))
        
        # Export data perturbations
        if self.data_data:
            data_df = self._convert_to_dataframe(self.data_data, "Data")
            data_file = os.path.join(output_dir, f"{self.dataset_name}_data_perturbations.csv")
            
            if save_files:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                data_df.to_csv(data_file, index=False)
                print(f"✓ Saved data perturbations: {data_file}")
            
            saved_files.append(("Data", data_df, data_file if save_files else None))
        
        return saved_files
    
    def _convert_to_dataframe(self, data_dict, perturbation_type):
        """Convert parsed data dictionary to DataFrame"""
        rows = []
        
        for method, method_data in data_dict.items():
            for config, metrics in method_data.items():
                row = {
                    'Method': method,
                    'Configuration': config,
                    'Perturbation_Type': perturbation_type,
                    'mean_validity': metrics['mean_validity'],
                    'std_validity': metrics['std_validity'],
                    'mean_accuracy': metrics['mean_accuracy'],
                    'std_accuracy': metrics['std_accuracy']
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def validate_data(self):
        """Validate the parsed data for sanity checks"""
        print(f"\n🔍 VALIDATION REPORT FOR {self.dataset_name.upper()}")
        print("=" * 60)
        
        for data_type, data in [("Model", self.model_data), ("Data", self.data_data)]:
            if data is None:
                print(f"\n{data_type} Perturbations: NO DATA")
                continue
                
            print(f"\n{data_type} Perturbations:")
            print("-" * 30)
            
            total_configs = 0
            
            for method in self.target_methods:
                if method in data:
                    method_data = data[method]
                    configs = list(method_data.keys())
                    
                    if total_configs == 0:
                        total_configs = len(configs)
                    
                    # Count valid vs missing
                    valid_count = sum(1 for v in method_data.values() if pd.notna(v['mean_validity']))
                    missing_count = len(configs) - valid_count
                    
                    # Check validity range
                    valid_validities = [v['mean_validity'] for v in method_data.values() if pd.notna(v['mean_validity'])]
                    validity_check = "✅" if all(v <= 1.0 for v in valid_validities) else "❌"
                    
                    print(f"  {method}:")
                    print(f"    ✓ Configurations: {len(configs)}")
                    print(f"    ✓ Valid entries: {valid_count}")
                    print(f"    ✓ Missing entries: {missing_count}")
                    if valid_validities:
                        print(f"    {validity_check} Validity range: {min(valid_validities):.3f} - {max(valid_validities):.3f}")
                    else:
                        print(f"    ⚠️  No valid validity values")
                else:
                    print(f"  {method}: NO DATA")
            
            print(f"\n  📊 Total configurations in {data_type}: {total_configs}")
        
        return True


# Comprehensive dataset processing functions
def process_all_datasets(save_csv=False):
    """Process all datasets using the unified ExperimentDataReader"""
    datasets = ['german_credit', 'spambase', 'compas', 'heloc']
    
    print("=" * 80)
    print("PROCESSING ALL DATASETS WITH UNIFIED READER")
    print("=" * 80)
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"PROCESSING: {dataset.upper()}")
        print(f"{'='*60}")
        
        try:
            # Create reader for this dataset
            reader = ExperimentDataReader(dataset)
            
            # Load and parse the data
            if reader.load_and_analyze():
                print(f"\n✅ Successfully loaded {dataset}")
                
                # Validate data
                reader.validate_data()
                
                # Create DataFrames (but don't save CSV by default)
                saved_files = reader.export_to_csv(save_files=save_csv)
                
                # Display summary
                print(f"\n📊 PROCESSING SUMMARY FOR {dataset.upper()}:")
                for table_type, df, file_path in saved_files:
                    status = f"→ {os.path.basename(file_path)}" if file_path else "(not saved)"
                    print(f"  {table_type} Perturbations: {len(df)} rows {status}")
                    
                    # Show sample data for each method
                    for method in reader.target_methods:
                        method_data = df[df['Method'] == method]
                        if not method_data.empty:
                            valid_count = method_data['mean_validity'].notna().sum()
                            print(f"    {method}: {valid_count}/{len(method_data)} valid entries")
                
            else:
                print(f"❌ Failed to process {dataset}")
                
        except Exception as e:
            print(f"❌ Error processing {dataset}: {e}")
            import traceback
            traceback.print_exc()

def test_single_dataset(dataset_name, save_csv=False):
    """Test processing of a single dataset"""
    print(f"=" * 60)
    print(f"TESTING SINGLE DATASET: {dataset_name.upper()}")
    print(f"=" * 60)
    
    try:
        reader = ExperimentDataReader(dataset_name)
        
        if reader.load_and_analyze():
            reader.validate_data()
            saved_files = reader.export_to_csv(save_files=save_csv)
            
            print(f"\n🎯 TEST RESULTS FOR {dataset_name.upper()}:")
            print(f"  Model data available: {'✅' if reader.model_data else '❌'}")
            print(f"  Data data available: {'✅' if reader.data_data else '❌'}")
            print(f"  DataFrames created: {len(saved_files)}")
            if save_csv:
                csv_count = sum(1 for _, _, path in saved_files if path is not None)
                print(f"  CSV files exported: {csv_count}")
            
            return reader
        else:
            print(f"❌ Failed to load {dataset_name}")
            return None
            
    except Exception as e:
        print(f"❌ Error testing {dataset_name}: {e}")
        return None

# Main execution functions
def main(save_files=False):
    """Main function demonstrating baseline and experiment data processing"""
    # Test baseline data reader
    print("=" * 80)
    print("STEP 1: TESTING BASELINE DATA READER")
    print("=" * 80)
    
    baseline_reader = BaselineDataReader()
    if baseline_reader.load_data():
        baseline_reader.display_summary()
        baseline_reader.save_summary_csv("all_baseline_values_summary.csv", save_file=save_files)
    
    print("\n" + "=" * 80)
    print("STEP 2: TESTING EXPERIMENT DATA PROCESSING")
    print("=" * 80)
    
    # Process all datasets
    process_all_datasets(save_csv=save_files)

def test_german_credit_only(save_csv=False):
    """Quick test for German Credit dataset only"""
    return test_single_dataset('german_credit', save_csv=save_csv)

if __name__ == "__main__":
    # Run main processing
    main()

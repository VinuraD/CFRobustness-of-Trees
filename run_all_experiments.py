#!/usr/bin/env python3
"""
Sequential Counterfactual Robustness Experiment Runner

This script runs all 20 counterfactual robustness analysis files sequentially,
waiting for each experiment to complete before starting the next one.

Usage:
    python run_all_experiments.py [options]

Features:
- Runs all 5 CF methods (DiCE, CEML, cfxplorer, NICE, feature_tweak) across 4 datasets
- Cross-platform support (Windows, Linux, macOS)
- Sequential execution with comprehensive progress tracking
- Real-time output display for each experiment
- Supports filtering by method or dataset
- Environment validation and dependency checking
- Organized output directory structure
- JSON status tracking with timestamps and durations

Improvements over original:
- Fixed Windows compatibility issues
- Sequential execution prevents resource conflicts
- Enhanced logging and status persistence
- Better error handling and environment validation
- Organized experiment outputs in dedicated directories
"""

import os
import sys
import subprocess
import time
import logging
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import json

class ExperimentRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.experiment_status = {}
        
        # Define all experiments
        self.methods = ['DiCE', 'CEML', 'cfxplorer', 'NICE', 'feature_tweak', 'CERTS']
        self.datasets = [
            ('v2', 'Spambase'),
            ('v3', 'German-Credit'), 
            ('v4_heloc', 'HELOC'),
            ('v5_compas', 'COMPAS')
        ]
        
        # Setup logging
        self.setup_logging()
        
        # Check Python environment
        self.check_environment()
        
    def check_environment(self):
        """Check if Python environment has required dependencies"""
        self.logger.info("Checking Python environment...")
        
        # Check Python version
        python_version = sys.version_info
        self.logger.info(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check for key dependencies
        required_packages = [
            'pandas', 'numpy', 'sklearn', 'matplotlib', 'seaborn'
        ]
        
        # Check for optional but important packages
        optional_packages = [
            'tensorflow', 'torch', 'cfxplorer'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                self.logger.info(f"[OK] {package} is available")
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"[MISSING] {package} is missing")
        
        # Check optional packages but don't fail if missing
        for package in optional_packages:
            try:
                if package == 'tensorflow':
                    # TensorFlow can take a long time to import, so add a timeout mechanism
                    self.logger.info(f"[CHECKING] {package} (may take a moment)...")
                __import__(package)
                self.logger.info(f"[OK] {package} is available")
            except ImportError:
                self.logger.info(f"[OPTIONAL] {package} is not available (some methods may not work)")
            except Exception as e:
                self.logger.warning(f"[WARNING] {package} import failed: {e}")
        
        if missing_packages:
            self.logger.warning(f"Missing packages: {', '.join(missing_packages)}")
            self.logger.warning("Some experiments may fail. Consider installing missing packages.")
        else:
            self.logger.info("[OK] All basic dependencies are available")
            
        # Check if data files exist
        data_dir = self.base_dir / "data"
        if not data_dir.exists():
            self.logger.error(f"[ERROR] Data directory not found: {data_dir}")
            return False
            
        expected_files = ["Spambase.csv", "German-Credit.csv", "HELOC.csv", "COMPAS.csv"]
        for file in expected_files:
            file_path = data_dir / file
            if file_path.exists():
                self.logger.info(f"[OK] Data file found: {file}")
            else:
                self.logger.error(f"[ERROR] Data file missing: {file}")
                
        return True
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"experiment_runner_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment runner started - Log file: {log_file}")
        
    def get_experiment_files(self, methods=None, datasets=None):
        """Get list of experiment files to run"""
        if methods is None:
            methods = self.methods
        if datasets is None:
            datasets = [d[0] for d in self.datasets]
            
        experiments = []
        
        for method in methods:
            method_dir = self.base_dir / method
            if not method_dir.exists():
                self.logger.warning(f"Method directory not found: {method_dir}")
                continue
                
            for dataset_version, dataset_name in self.datasets:
                if dataset_version not in datasets:
                    continue
                    
                # Construct filename - DiCE doesn't include method name in filename
                if method == 'DiCE':
                    if dataset_version == 'v2':
                        filename = f"cf_robustness_analysis_v2.py"
                    elif dataset_version == 'v3':
                        filename = f"cf_robustness_analysis_v3.py"
                    elif dataset_version == 'v4_heloc':
                        filename = f"cf_robustness_analysis_v4_heloc.py"
                    elif dataset_version == 'v5_compas':
                        filename = f"cf_robustness_analysis_v5_compas.py"
                else:
                    # Other methods include method name in filename
                    # Handle special case for feature_tweak (uses 'featuretweak' without underscore)
                    method_name = method.lower().replace('_', '') if method == 'feature_tweak' else method.lower()
                    
                    if dataset_version == 'v2':
                        filename = f"cf_robustness_analysis_{method_name}_v2.py"
                    elif dataset_version == 'v3':
                        filename = f"cf_robustness_analysis_{method_name}_v3.py"
                    elif dataset_version == 'v4_heloc':
                        filename = f"cf_robustness_analysis_{method_name}_v4_heloc.py"
                    elif dataset_version == 'v5_compas':
                        filename = f"cf_robustness_analysis_{method_name}_v5_compas.py"
                
                file_path = method_dir / filename
                
                if file_path.exists():
                    experiments.append({
                        'method': method,
                        'dataset': dataset_name,
                        'version': dataset_version,
                        'file_path': file_path,
                        'filename': filename
                    })
                else:
                    self.logger.warning(f"File not found: {file_path}")
        
        return experiments
    
    def run_experiment(self, experiment, timeout_minutes=None):
        """Run a single experiment sequentially and wait for completion"""
        method = experiment['method']
        dataset = experiment['dataset']
        file_path = experiment['file_path']
        
        experiment_id = f"{method}_{experiment['version']}"
        
        # Create output directory for this experiment
        output_dir = self.base_dir / "experiment_outputs" / experiment_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log files
        log_file = output_dir / f"{experiment_id}.log"
        err_file = output_dir / f"{experiment_id}.err"
        
        self.experiment_status[experiment_id] = {
            'status': 'running',
            'start_time': datetime.now(),
            'method': method,
            'dataset': dataset,
            'output_dir': str(output_dir),
            'log_file': str(log_file),
            'error_file': str(err_file)
        }
        
        self.logger.info(f"[STARTING] {experiment_id}")
        self.logger.info(f"  Working directory: {file_path.parent}")
        self.logger.info(f"  Script: {file_path.name}")
        self.logger.info(f"  Log file: {log_file}")
        self.logger.info(f"  Error file: {err_file}")
        if timeout_minutes is not None:
            self.logger.info(f"  Timeout: {timeout_minutes} minutes")
        else:
            self.logger.info(f"  Timeout: None (unlimited)")
        
        try:
            # Run the experiment and wait for completion
            with open(log_file, 'w') as log_f, open(err_file, 'w') as err_f:
                process = subprocess.Popen(
                    ['python', file_path.name],
                    cwd=file_path.parent,
                    stdout=log_f,
                    stderr=err_f,
                    text=True
                )
                
                # Wait for the process to complete with optional timeout
                if timeout_minutes is not None:
                    timeout = timeout_minutes * 60  # Convert to seconds
                    try:
                        return_code = process.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        self.logger.error(f"[TIMEOUT] {experiment_id} exceeded {timeout_minutes} minute timeout, terminating...")
                        process.kill()
                        process.wait()  # Wait for process to actually terminate
                        return_code = -1  # Set to failure code
                        
                        # Add timeout info to status
                        self.experiment_status[experiment_id]['timeout'] = True
                else:
                    # No timeout - wait indefinitely
                    return_code = process.wait()
                
                end_time = datetime.now()
                duration = end_time - self.experiment_status[experiment_id]['start_time']
                
                # Update status
                self.experiment_status[experiment_id]['end_time'] = end_time
                self.experiment_status[experiment_id]['duration'] = duration
                self.experiment_status[experiment_id]['return_code'] = return_code
                
                if return_code == 0:
                    self.experiment_status[experiment_id]['status'] = 'completed'
                    self.logger.info(f"[COMPLETED] {experiment_id} (Duration: {duration}, Exit code: {return_code})")
                elif return_code == -1 and self.experiment_status[experiment_id].get('timeout', False):
                    self.experiment_status[experiment_id]['status'] = 'timeout'
                    timeout_msg = f"{timeout_minutes} minutes" if timeout_minutes is not None else "unknown"
                    self.logger.error(f"[TIMEOUT] {experiment_id} (Duration: {duration}, Timeout after {timeout_msg})")
                else:
                    self.experiment_status[experiment_id]['status'] = 'failed'
                    self.logger.error(f"[FAILED] {experiment_id} (Duration: {duration}, Exit code: {return_code})")
                    
                    # Show error file contents if there are errors
                    if err_file.exists() and err_file.stat().st_size > 0:
                        self.logger.error(f"Error output for {experiment_id}:")
                        with open(err_file, 'r') as f:
                            error_content = f.read().strip()
                            # Show first few lines of error
                            error_lines = error_content.split('\n')[:5]
                            for line in error_lines:
                                self.logger.error(f"  {line}")
                            if len(error_content.split('\n')) > 5:
                                self.logger.error(f"  ... (see {err_file} for full error)")
                
                return return_code == 0
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to run {experiment_id}: {e}")
            self.experiment_status[experiment_id]['status'] = 'error'
            self.experiment_status[experiment_id]['error'] = str(e)
            return False
    
    def run_all_experiments(self, experiments, timeout_minutes=None):
        """Run all experiments sequentially"""
        self.logger.info(f"[LAUNCH] Starting {len(experiments)} experiments sequentially...")
        
        success_count = 0
        failed_count = 0
        
        for i, experiment in enumerate(experiments, 1):
            experiment_id = f"{experiment['method']}_{experiment['version']}"
            self.logger.info(f"\n[{i}/{len(experiments)}] Running {experiment_id}...")
            
            if self.run_experiment(experiment, timeout_minutes):
                success_count += 1
            else:
                failed_count += 1
            
            # Save status after each experiment
            self.save_status()
            
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"[SUMMARY] Sequential Execution Complete!")
        self.logger.info(f"  Successfully completed: {success_count}")
        self.logger.info(f"  Failed: {failed_count}")
        self.logger.info(f"  Total experiments: {len(experiments)}")
        self.logger.info(f"{'='*80}")
        
        return success_count
    
    def print_final_status(self):
        """Print final status of all experiments"""
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL EXPERIMENT STATUS")
        self.logger.info("="*80)
        
        completed = []
        failed = []
        timeout = []
        error = []
        
        for exp_id, status in self.experiment_status.items():
            if status['status'] == 'completed':
                completed.append(exp_id)
            elif status['status'] == 'failed':
                failed.append(exp_id)
            elif status['status'] == 'timeout':
                timeout.append(exp_id)
            else:
                error.append(exp_id)
        
        self.logger.info(f"[SUCCESS] Completed ({len(completed)}): {', '.join(completed) if completed else 'None'}")
        self.logger.info(f"[FAILED] Failed ({len(failed)}): {', '.join(failed) if failed else 'None'}")
        self.logger.info(f"[TIMEOUT] Timeout ({len(timeout)}): {', '.join(timeout) if timeout else 'None'}")
        self.logger.info(f"[ERROR] Errors ({len(error)}): {', '.join(error) if error else 'None'}")
        
        # Show log file locations
        if self.experiment_status:
            self.logger.info("\nLog file locations:")
            for exp_id, status in self.experiment_status.items():
                log_file = status.get('log_file', '')
                error_file = status.get('error_file', '')
                duration = status.get('duration', 'N/A')
                self.logger.info(f"  {exp_id} ({status['status']}) - Duration: {duration}")
                if log_file:
                    self.logger.info(f"    Log: {log_file}")
                if error_file:
                    self.logger.info(f"    Error: {error_file}")
        
        # Save final status
        self.save_status()
    
    def save_status(self):
        """Save experiment status to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        status_file = self.base_dir / f"experiment_status_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_status = {
            'run_info': {
                'timestamp': timestamp,
                'total_experiments': len(self.experiment_status),
                'base_dir': str(self.base_dir),
                'platform': os.name
            },
            'experiments': {}
        }
        
        for exp_id, status in self.experiment_status.items():
            json_status['experiments'][exp_id] = {
                'status': status['status'],
                'method': status['method'],
                'dataset': status['dataset'],
                'start_time': status['start_time'].isoformat(),
                'output_dir': status.get('output_dir', ''),
                'log_file': status.get('log_file', ''),
                'error_file': status.get('error_file', ''),
                'return_code': status.get('return_code', None)
            }
            if 'end_time' in status:
                json_status['experiments'][exp_id]['end_time'] = status['end_time'].isoformat()
            if 'duration' in status:
                json_status['experiments'][exp_id]['duration_seconds'] = status['duration'].total_seconds()
                json_status['experiments'][exp_id]['duration_str'] = str(status['duration'])
            if 'error' in status:
                json_status['experiments'][exp_id]['error'] = status['error']
        
        with open(status_file, 'w') as f:
            json.dump(json_status, f, indent=2)
        
        # Also save a "latest" version for easy access
        latest_file = self.base_dir / "experiment_status_latest.json"
        with open(latest_file, 'w') as f:
            json.dump(json_status, f, indent=2)
        
        self.logger.info(f"[SAVED] Status saved to: {status_file}")
        return status_file
    
    def load_latest_status(self):
        """Load the latest experiment status if available"""
        latest_file = self.base_dir / "experiment_status_latest.json"
        if latest_file.exists():
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    
                self.logger.info(f"[LOADED] Loaded previous status from: {latest_file}")
                return data['experiments']
            except Exception as e:
                self.logger.warning(f"Failed to load previous status: {e}")
        
        return {}

def main():
    parser = argparse.ArgumentParser(description='Run counterfactual robustness experiments sequentially')
    parser.add_argument('--methods', nargs='+', 
                       choices=['DiCE', 'CEML', 'cfxplorer', 'NICE', 'feature_tweak', 'CERTS'],
                       help='Specific methods to run (default: all)')
    parser.add_argument('--datasets', nargs='+',
                       choices=['v2', 'v3', 'v4_heloc', 'v5_compas'],
                       help='Specific datasets to run (default: all)')
    parser.add_argument('--list-only', action='store_true',
                       help='List experiments that would be run without running them')
    parser.add_argument('--check-env', action='store_true',
                       help='Only check environment and exit')
    parser.add_argument('--show-logs', action='store_true',
                       help='Show log file locations from the latest run')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout for each experiment in minutes (default: no timeout)')
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner()
    
    # If only checking environment, exit after check
    if args.check_env:
        return 0
    
    # Show log files from latest run
    if args.show_logs:
        previous_status = runner.load_latest_status()
        if previous_status:
            runner.logger.info("Log file locations from latest run:")
            for exp_id, status in previous_status.items():
                log_file = status.get('log_file', '')
                error_file = status.get('error_file', '')
                runner.logger.info(f"  {exp_id}:")
                if log_file:
                    runner.logger.info(f"    Log: {log_file}")
                if error_file:
                    runner.logger.info(f"    Error: {error_file}")
        else:
            runner.logger.info("No previous run found")
        return 0
    
    # Get experiments to run
    experiments = runner.get_experiment_files(args.methods, args.datasets)
    
    if not experiments:
        runner.logger.error("[ERROR] No valid experiments found!")
        return 1
    
    # List experiments
    runner.logger.info(f"\n[FOUND] Found {len(experiments)} experiments:")
    for exp in experiments:
        runner.logger.info(f"   {exp['method']} - {exp['dataset']} ({exp['version']})")
    
    if args.list_only:
        return 0
    
    # Confirm execution
    if len(experiments) > 5:
        confirm = input(f"\n[WARNING] About to run {len(experiments)} experiments sequentially. This may take a long time. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            runner.logger.info("Aborted by user")
            return 0
    
    # Run experiments sequentially
    success_count = runner.run_all_experiments(experiments, args.timeout)
    
    if success_count == 0:
        runner.logger.error("[ERROR] No experiments completed successfully!")
        return 1
    
    # Print final status
    runner.print_final_status()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
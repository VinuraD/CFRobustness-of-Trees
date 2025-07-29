#!/usr/bin/env python3
"""
Parallel Counterfactual Robustness Experiment Runner

This script launches all 20 counterfactual robustness analysis files in separate terminals
for parallel execution, significantly reducing total experiment time.

Usage:
    python run_all_experiments.py [options]

Features:
- Runs all 5 CF methods (DiCE, CEML, cfxplorer, NICE, feature_tweak) across 4 datasets
- Opens each experiment in a separate terminal for parallel execution
- Monitors experiment status and provides logging
- Supports filtering by method or dataset
- Handles different terminal emulators (gnome-terminal, xterm, etc.)
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
import signal
import json

class ExperimentRunner:
    def __init__(self):
        self.base_dir = Path(__file__).parent.absolute()
        self.processes = []
        self.experiment_status = {}
        
        # Define all experiments
        self.methods = ['DiCE', 'CEML', 'cfxplorer', 'NICE', 'feature_tweak']
        self.datasets = [
            ('v2', 'Spambase'),
            ('v3', 'German-Credit'), 
            ('v4_heloc', 'HELOC'),
            ('v5_compas', 'COMPAS')
        ]
        
        # Setup logging
        self.setup_logging()
        
        # Detect available terminal
        self.terminal_cmd = self.detect_terminal()
        
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
        
    def detect_terminal(self):
        """Detect available terminal emulator"""
        terminals = [
            ('gnome-terminal', ['gnome-terminal', '--', 'bash', '-c']),
            ('xterm', ['xterm', '-e', 'bash', '-c']),
            ('konsole', ['konsole', '-e', 'bash', '-c']),
            ('xfce4-terminal', ['xfce4-terminal', '-e', 'bash', '-c']),
            ('terminator', ['terminator', '-e', 'bash -c']),
        ]
        
        for name, cmd in terminals:
            if shutil.which(name):
                self.logger.info(f"Detected terminal emulator: {name}")
                return cmd
        
        # Fallback to basic terminal
        self.logger.warning("No GUI terminal detected, using basic approach")
        return None
    
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
    
    def run_experiment(self, experiment):
        """Run a single experiment in a new terminal"""
        method = experiment['method']
        dataset = experiment['dataset']
        file_path = experiment['file_path']
        
        experiment_id = f"{method}_{experiment['version']}"
        
        # Prepare command
        python_cmd = f"cd '{file_path.parent}' && python '{file_path.name}'"
        
        # Add title and keep terminal open after completion
        full_cmd = f"""
        echo "Starting {method} - {dataset} experiment..."
        echo "Working directory: {file_path.parent}"
        echo "File: {file_path.name}"
        echo "Started at: $(date)"
        echo "=" | tr '=' '=' | head -c 80 && echo
        {python_cmd}
        echo 
        echo "=" | tr '=' '=' | head -c 80 && echo
        echo "{method} - {dataset} experiment completed at: $(date)"
        echo "Press Enter to close this terminal..."
        read
        """
        
        try:
            if self.terminal_cmd:
                # Use GUI terminal
                cmd = self.terminal_cmd + [f"'{full_cmd}'"]
                # Set window title
                if 'gnome-terminal' in self.terminal_cmd[0]:
                    cmd = ['gnome-terminal', '--title', f'{method} - {dataset}', '--', 'bash', '-c', full_cmd]
                elif 'xterm' in self.terminal_cmd[0]:
                    cmd = ['xterm', '-title', f'{method} - {dataset}', '-e', 'bash', '-c', full_cmd]
                
                process = subprocess.Popen(cmd, 
                                         cwd=file_path.parent,
                                         stdout=subprocess.PIPE, 
                                         stderr=subprocess.PIPE)
            else:
                # Fallback: run in background with nohup
                process = subprocess.Popen([
                    'nohup', 'python', file_path.name
                ], cwd=file_path.parent,
                   stdout=open(f'{experiment_id}.out', 'w'),
                   stderr=open(f'{experiment_id}.err', 'w'))
            
            self.processes.append({
                'process': process,
                'experiment': experiment,
                'experiment_id': experiment_id,
                'start_time': datetime.now()
            })
            
            self.experiment_status[experiment_id] = {
                'status': 'running',
                'start_time': datetime.now(),
                'method': method,
                'dataset': dataset
            }
            
            self.logger.info(f"✅ Started: {experiment_id} (PID: {process.pid})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start {experiment_id}: {e}")
            return False
    
    def run_all_experiments(self, experiments, delay=2):
        """Run all experiments with optional delay between launches"""
        self.logger.info(f"🚀 Starting {len(experiments)} experiments...")
        
        success_count = 0
        for i, experiment in enumerate(experiments, 1):
            experiment_id = f"{experiment['method']}_{experiment['version']}"
            self.logger.info(f"[{i}/{len(experiments)}] Launching {experiment_id}...")
            
            if self.run_experiment(experiment):
                success_count += 1
                if delay > 0 and i < len(experiments):
                    time.sleep(delay)
            
        self.logger.info(f"🎯 Launch Summary: {success_count}/{len(experiments)} experiments started successfully")
        
        # Save experiment status
        self.save_status()
        
        return success_count
    
    def monitor_experiments(self, check_interval=30):
        """Monitor running experiments"""
        if not self.processes:
            self.logger.info("No experiments to monitor")
            return
            
        self.logger.info(f"👁️ Monitoring {len(self.processes)} experiments (checking every {check_interval}s)")
        self.logger.info("Press Ctrl+C to stop monitoring and view final status")
        
        try:
            while True:
                running_count = 0
                completed_count = 0
                
                for proc_info in self.processes:
                    process = proc_info['process']
                    experiment_id = proc_info['experiment_id']
                    
                    if process.poll() is None:
                        running_count += 1
                        self.experiment_status[experiment_id]['status'] = 'running'
                    else:
                        if self.experiment_status[experiment_id]['status'] == 'running':
                            # Just completed
                            self.experiment_status[experiment_id]['status'] = 'completed'
                            self.experiment_status[experiment_id]['end_time'] = datetime.now()
                            
                            duration = self.experiment_status[experiment_id]['end_time'] - proc_info['start_time']
                            self.logger.info(f"✅ Completed: {experiment_id} (Duration: {duration})")
                        
                        completed_count += 1
                
                # Print status summary
                self.logger.info(f"📊 Status: {running_count} running, {completed_count} completed")
                
                if running_count == 0:
                    self.logger.info("🎉 All experiments completed!")
                    break
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            self.logger.info("\n⏹️ Monitoring stopped by user")
        
        self.print_final_status()
    
    def print_final_status(self):
        """Print final status of all experiments"""
        self.logger.info("\n" + "="*80)
        self.logger.info("FINAL EXPERIMENT STATUS")
        self.logger.info("="*80)
        
        running = []
        completed = []
        
        for exp_id, status in self.experiment_status.items():
            if status['status'] == 'running':
                running.append(exp_id)
            else:
                completed.append(exp_id)
        
        self.logger.info(f"✅ Completed ({len(completed)}): {', '.join(completed) if completed else 'None'}")
        self.logger.info(f"🔄 Still Running ({len(running)}): {', '.join(running) if running else 'None'}")
        
        # Save final status
        self.save_status()
    
    def save_status(self):
        """Save experiment status to JSON file"""
        status_file = f"experiment_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert datetime objects to strings for JSON serialization
        json_status = {}
        for exp_id, status in self.experiment_status.items():
            json_status[exp_id] = {
                'status': status['status'],
                'method': status['method'],
                'dataset': status['dataset'],
                'start_time': status['start_time'].isoformat(),
            }
            if 'end_time' in status:
                json_status[exp_id]['end_time'] = status['end_time'].isoformat()
        
        with open(status_file, 'w') as f:
            json.dump(json_status, f, indent=2)
        
        self.logger.info(f"💾 Status saved to: {status_file}")
    
    def kill_all_experiments(self):
        """Terminate all running experiments"""
        self.logger.info("🛑 Terminating all experiments...")
        
        killed_count = 0
        for proc_info in self.processes:
            process = proc_info['process']
            experiment_id = proc_info['experiment_id']
            
            if process.poll() is None:  # Still running
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    killed_count += 1
                    self.logger.info(f"🔪 Killed: {experiment_id}")
                    self.experiment_status[experiment_id]['status'] = 'terminated'
                except Exception as e:
                    self.logger.error(f"Failed to kill {experiment_id}: {e}")
        
        self.logger.info(f"🛑 Terminated {killed_count} experiments")

def signal_handler(sig, frame, runner):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Received interrupt signal...")
    choice = input("Do you want to (k)ill all experiments, (c)ontinue monitoring, or (q)uit monitoring? [k/c/q]: ").lower()
    
    if choice == 'k':
        runner.kill_all_experiments()
        sys.exit(0)
    elif choice == 'q':
        runner.print_final_status()
        sys.exit(0)
    # else continue monitoring

def main():
    parser = argparse.ArgumentParser(description='Run counterfactual robustness experiments in parallel')
    parser.add_argument('--methods', nargs='+', 
                       choices=['DiCE', 'CEML', 'cfxplorer', 'NICE', 'feature_tweak'],
                       help='Specific methods to run (default: all)')
    parser.add_argument('--datasets', nargs='+',
                       choices=['v2', 'v3', 'v4_heloc', 'v5_compas'],
                       help='Specific datasets to run (default: all)')
    parser.add_argument('--delay', type=int, default=2,
                       help='Delay between experiment launches in seconds (default: 2)')
    parser.add_argument('--no-monitor', action='store_true',
                       help='Launch experiments without monitoring')
    parser.add_argument('--list-only', action='store_true',
                       help='List experiments that would be run without running them')
    
    args = parser.parse_args()
    
    # Create runner
    runner = ExperimentRunner()
    
    # Setup signal handler
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, runner))
    
    # Get experiments to run
    experiments = runner.get_experiment_files(args.methods, args.datasets)
    
    if not experiments:
        runner.logger.error("❌ No valid experiments found!")
        return 1
    
    # List experiments
    runner.logger.info(f"\n📋 Found {len(experiments)} experiments:")
    for exp in experiments:
        runner.logger.info(f"   {exp['method']} - {exp['dataset']} ({exp['version']})")
    
    if args.list_only:
        return 0
    
    # Confirm execution
    if len(experiments) > 10:
        confirm = input(f"\n⚠️  About to launch {len(experiments)} experiments. Continue? [y/N]: ")
        if confirm.lower() != 'y':
            runner.logger.info("Aborted by user")
            return 0
    
    # Run experiments
    success_count = runner.run_all_experiments(experiments, args.delay)
    
    if success_count == 0:
        runner.logger.error("❌ No experiments started successfully!")
        return 1
    
    # Monitor if requested
    if not args.no_monitor:
        runner.monitor_experiments()
    else:
        runner.logger.info("🚀 Experiments launched. Use --monitor to track progress.")
        runner.save_status()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
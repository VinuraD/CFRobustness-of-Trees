#!/usr/bin/env python3
"""
Parse CERTS v3 log file, compute mean/std across 5 folds,
create summary log, and update Excel files with CERTS results.
"""

import re
import numpy as np
import pandas as pd
from collections import defaultdict
import os

LOG_PATH = "/mnt/datassd3/rashinda/CF_Robustness/CFRobustness-of-Trees/experiment_outputs/CERTS_v3/CERTS_v3.log"
SUMMARY_PATH = "/mnt/datassd3/rashinda/CF_Robustness/CFRobustness-of-Trees/experiment_outputs/CERTS_v3/CERTS_v3_summary.log"
DATA_PERTURB_XLSX = "/mnt/datassd3/rashinda/CF_Robustness/CFRobustness-of-Trees/Final_anlysis_and_plotting/german_credit_data_perturb.xlsx"
MODEL_PERTURB_XLSX = "/mnt/datassd3/rashinda/CF_Robustness/CFRobustness-of-Trees/Final_anlysis_and_plotting/German_Credit_model_perturb.xlsx"


def parse_log():
    """Parse all 5 folds from the CERTS v3 log file."""
    with open(LOG_PATH, 'r') as f:
        lines = f.readlines()

    # Data structures to hold per-fold results
    # data_perturb[perturbation_type][bin_num] = list of (validity, accuracy) per fold
    data_perturb = defaultdict(lambda: defaultdict(list))
    # model_perturb[config_key] = list of (validity, accuracy) per fold
    model_perturb = defaultdict(list)

    current_fold = None
    current_section = None  # 'data' or 'model'
    current_perturbation = None  # 'minor_deletion' or 'minor_addition'

    for line in lines:
        line = line.rstrip('\n')

        # Detect fold
        fold_match = re.match(r'^--- FOLD (\d+) ANALYSIS ---', line)
        if fold_match:
            current_fold = int(fold_match.group(1))
            current_section = None
            current_perturbation = None
            continue

        # Detect data perturbation section
        if re.match(r'^Testing data perturbations for fold \d+', line):
            current_section = 'data'
            continue

        # Detect model perturbation section
        if re.match(r'^Testing model perturbations for fold \d+', line):
            current_section = 'model'
            continue

        # Detect perturbation type within data section
        if current_section == 'data':
            if line.strip() == 'minor_deletion:':
                current_perturbation = 'minor_deletion'
                continue
            elif line.strip() == 'minor_addition:':
                current_perturbation = 'minor_addition'
                continue

            # Parse data perturbation bin results
            # Format: "    Bin X: Remove/Use Y% -> validity: V, accuracy: A, L2: ..., L0: ..., LOF: ..."
            bin_match = re.match(
                r'^\s+Bin (\d+): (?:Remove|Use) (\d+)% -> validity: ([\d.]+), accuracy: ([\d.]+)',
                line
            )
            if bin_match and current_perturbation:
                bin_num = int(bin_match.group(1))
                data_pct = int(bin_match.group(2))
                validity = float(bin_match.group(3))
                accuracy = float(bin_match.group(4))
                data_perturb[(current_perturbation, bin_num)]['validity'].append(validity)
                data_perturb[(current_perturbation, bin_num)]['accuracy'].append(accuracy)
                data_perturb[(current_perturbation, bin_num)]['data_pct'] = data_pct
                continue

        # Parse model perturbation results
        if current_section == 'model':
            # Format: "      model_type {params}: validity V, accuracy A, L2: ..., L0: ..., LOF: ..."
            model_match = re.match(
                r'^\s+(random_forest|xgboost|lightgbm|adaboost|catboost)\s+\{([^}]+)\}:\s+validity\s+([\d.]+),\s+accuracy\s+([\d.]+)',
                line
            )
            if model_match:
                model_type = model_match.group(1)
                params_str = model_match.group(2)
                validity = float(model_match.group(3))
                accuracy = float(model_match.group(4))

                # Extract max_depth/depth and n_estimators/iterations
                depth_match = re.search(r"'(?:max_depth|depth)':\s*(\d+)", params_str)
                n_est_match = re.search(r"'(?:n_estimators|iterations)':\s*(\d+)", params_str)
                if depth_match and n_est_match:
                    depth = int(depth_match.group(1))
                    n_est = int(n_est_match.group(1))
                    config_key = f"{model_type}_{depth}_{n_est}"
                    model_perturb[config_key].append({
                        'validity': validity,
                        'accuracy': accuracy,
                        'fold': current_fold
                    })
                continue

    return data_perturb, model_perturb


def compute_stats(data_perturb, model_perturb):
    """Compute mean and std across 5 folds."""

    # Data perturbation stats
    data_stats = {}
    for (perturb_type, bin_num), values in sorted(data_perturb.items()):
        validities = np.array(values['validity'])
        accuracies = np.array(values['accuracy'])
        data_pct = values['data_pct']
        data_stats[(perturb_type, bin_num)] = {
            'data_pct': data_pct,
            'mean_validity': np.mean(validities),
            'std_validity': np.std(validities, ddof=0),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies, ddof=0),
            'n_folds': len(validities),
        }

    # Model perturbation stats
    # Handle duplicates: some configs appear twice per fold (e.g., 3_100)
    # We need to deduplicate by taking only the first occurrence per fold
    model_stats = {}
    for config_key, entries in sorted(model_perturb.items()):
        # Deduplicate: keep first occurrence per fold
        seen_folds = set()
        deduped = []
        for entry in entries:
            if entry['fold'] not in seen_folds:
                seen_folds.add(entry['fold'])
                deduped.append(entry)

        validities = np.array([e['validity'] for e in deduped])
        accuracies = np.array([e['accuracy'] for e in deduped])
        model_stats[config_key] = {
            'mean_validity': np.mean(validities),
            'std_validity': np.std(validities, ddof=0),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies, ddof=0),
            'n_folds': len(deduped),
        }

    return data_stats, model_stats


def write_summary_log(data_stats, model_stats):
    """Write a summary log file with averaged results."""
    with open(SUMMARY_PATH, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CERTS v3 - GERMAN CREDIT DATASET - SUMMARY (Mean +/- Std across 5 folds)\n")
        f.write("=" * 80 + "\n\n")

        # Data perturbation summary
        f.write("DATA PERTURBATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Perturbation':<20} {'Bin':>5} {'Data%':>6} {'Mean Val':>10} {'Std Val':>10} {'Mean Acc':>10} {'Std Acc':>10} {'N':>3}\n")
        f.write("-" * 80 + "\n")

        for (perturb_type, bin_num) in sorted(data_stats.keys()):
            s = data_stats[(perturb_type, bin_num)]
            f.write(f"{perturb_type:<20} {bin_num:>5} {s['data_pct']:>6} "
                    f"{s['mean_validity']:>10.4f} {s['std_validity']:>10.4f} "
                    f"{s['mean_accuracy']:>10.4f} {s['std_accuracy']:>10.4f} "
                    f"{s['n_folds']:>3}\n")

        f.write("\n\n")

        # Model perturbation summary
        f.write("MODEL PERTURBATION RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Model Configuration':<30} {'Mean Val':>10} {'Std Val':>10} {'Mean Acc':>10} {'Std Acc':>10} {'N':>3}\n")
        f.write("-" * 80 + "\n")

        # Sort by model type, depth, n_estimators
        def sort_key(config_key):
            parts = config_key.rsplit('_', 2)
            model = parts[0]
            depth = int(parts[1])
            n_est = int(parts[2])
            model_order = {'random_forest': 0, 'xgboost': 1, 'lightgbm': 2, 'adaboost': 3, 'catboost': 4}
            return (model_order.get(model, 99), depth, n_est)

        for config_key in sorted(model_stats.keys(), key=sort_key):
            s = model_stats[config_key]
            f.write(f"{config_key:<30} "
                    f"{s['mean_validity']:>10.4f} {s['std_validity']:>10.4f} "
                    f"{s['mean_accuracy']:>10.4f} {s['std_accuracy']:>10.4f} "
                    f"{s['n_folds']:>3}\n")

        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")

    print(f"Summary log written to: {SUMMARY_PATH}")


def update_data_perturb_excel(data_stats):
    """Append CERTS columns to the data perturbation Excel file."""
    # Read raw Excel (no header processing)
    df = pd.read_excel(DATA_PERTURB_XLSX, header=None)

    # Current structure:
    # Row 0: [nan, nan, nan, 'NICE', nan, nan, nan, 'DiCE', nan, nan, nan, ...]
    # Row 1: ['Pertbation', 'Bin', 'Data %', 'Mean Validity', 'Std Validity', 'Mean Accuracy', 'Std Accuracy', ...]
    # Rows 2+: data

    n_cols = df.shape[1]

    # Add 4 new columns for CERTS
    new_col_start = n_cols
    df[new_col_start] = None
    df[new_col_start + 1] = None
    df[new_col_start + 2] = None
    df[new_col_start + 3] = None

    # Set header row 0: method name in first new column, NaN for the rest
    df.iloc[0, new_col_start] = 'CERTS'
    df.iloc[0, new_col_start + 1] = None
    df.iloc[0, new_col_start + 2] = None
    df.iloc[0, new_col_start + 3] = None

    # Set header row 1: sub-headers
    df.iloc[1, new_col_start] = 'Mean Validity'
    df.iloc[1, new_col_start + 1] = 'Std Validity'
    df.iloc[1, new_col_start + 2] = 'Mean Accuracy'
    df.iloc[1, new_col_start + 3] = 'Std Accuracy'

    # Fill in data rows (rows 2+)
    for row_idx in range(2, len(df)):
        perturb_type = str(df.iloc[row_idx, 0]).strip()
        bin_num = df.iloc[row_idx, 1]
        try:
            bin_num = int(bin_num)
        except (ValueError, TypeError):
            continue

        key = (perturb_type, bin_num)
        if key in data_stats:
            s = data_stats[key]
            df.iloc[row_idx, new_col_start] = round(s['mean_validity'], 4)
            df.iloc[row_idx, new_col_start + 1] = round(s['std_validity'], 4)
            df.iloc[row_idx, new_col_start + 2] = round(s['mean_accuracy'], 4)
            df.iloc[row_idx, new_col_start + 3] = round(s['std_accuracy'], 4)
        else:
            print(f"WARNING: No CERTS data for data perturbation key {key}")

    # Write back
    df.to_excel(DATA_PERTURB_XLSX, index=False, header=False)
    print(f"Updated data perturbation Excel: {DATA_PERTURB_XLSX}")
    print(f"  Added CERTS columns at positions {new_col_start}-{new_col_start+3}")
    print(f"  Total columns now: {df.shape[1]}")


def update_model_perturb_excel(model_stats):
    """Append CERTS columns to the model perturbation Excel file."""
    # Read raw Excel (no header processing)
    df = pd.read_excel(MODEL_PERTURB_XLSX, header=None)

    # Current structure:
    # Row 0: [nan, 'NICE', nan, nan, nan, 'DiCE', nan, nan, nan, 'Feature Tweak', nan, nan, nan, 'CEML', nan, nan, nan]
    # Row 1: ['Model Configuration', 'Mean Validity', 'Std Validity', 'Mean Accuracy', 'Std Accuracy', ...]
    # Rows 2+: data with config names like 'random_forest_3_50'

    n_cols = df.shape[1]

    # Add 4 new columns for CERTS
    new_col_start = n_cols
    df[new_col_start] = None
    df[new_col_start + 1] = None
    df[new_col_start + 2] = None
    df[new_col_start + 3] = None

    # Set header row 0
    df.iloc[0, new_col_start] = 'CERTS'
    df.iloc[0, new_col_start + 1] = None
    df.iloc[0, new_col_start + 2] = None
    df.iloc[0, new_col_start + 3] = None

    # Set header row 1
    df.iloc[1, new_col_start] = 'Mean Validity'
    df.iloc[1, new_col_start + 1] = 'Std Validity'
    df.iloc[1, new_col_start + 2] = 'Mean Accuracy'
    df.iloc[1, new_col_start + 3] = 'Std Accuracy'

    # Fill in data rows (rows 2+)
    matched = 0
    unmatched = 0
    for row_idx in range(2, len(df)):
        config_name = str(df.iloc[row_idx, 0]).strip()

        if config_name in model_stats:
            s = model_stats[config_name]
            df.iloc[row_idx, new_col_start] = round(s['mean_validity'], 4)
            df.iloc[row_idx, new_col_start + 1] = round(s['std_validity'], 4)
            df.iloc[row_idx, new_col_start + 2] = round(s['mean_accuracy'], 4)
            df.iloc[row_idx, new_col_start + 3] = round(s['std_accuracy'], 4)
            matched += 1
        else:
            # Config exists in Excel but not in CERTS log - leave empty
            unmatched += 1

    print(f"Updated model perturbation Excel: {MODEL_PERTURB_XLSX}")
    print(f"  Added CERTS columns at positions {new_col_start}-{new_col_start+3}")
    print(f"  Matched: {matched} configs, Unmatched (no CERTS data): {unmatched} configs")
    print(f"  Total columns now: {df.shape[1]}")

    # Write back
    df.to_excel(MODEL_PERTURB_XLSX, index=False, header=False)


def main():
    print("=" * 60)
    print("CERTS v3 Log Parser - German Credit Dataset")
    print("=" * 60)

    # Step 1: Parse the log file
    print("\n1. Parsing log file...")
    data_perturb, model_perturb = parse_log()

    print(f"   Data perturbation entries: {len(data_perturb)}")
    print(f"   Model perturbation configs: {len(model_perturb)}")

    # Validate parsing
    print("\n   Data perturbation fold counts:")
    for key in sorted(data_perturb.keys())[:5]:
        n = len(data_perturb[key]['validity'])
        print(f"     {key}: {n} folds")
    print("     ...")

    print("\n   Model perturbation fold counts (sample):")
    for config in sorted(model_perturb.keys())[:5]:
        entries = model_perturb[config]
        folds = [e['fold'] for e in entries]
        print(f"     {config}: {len(entries)} entries, folds={folds}")
    print("     ...")

    # Step 2: Compute stats
    print("\n2. Computing mean and std across 5 folds...")
    data_stats, model_stats = compute_stats(data_perturb, model_perturb)

    print(f"   Data perturbation stats computed: {len(data_stats)} entries")
    print(f"   Model perturbation stats computed: {len(model_stats)} configs")

    # Verify all entries have 5 folds
    for key, s in data_stats.items():
        if s['n_folds'] != 5:
            print(f"   WARNING: {key} has {s['n_folds']} folds instead of 5!")

    for config, s in model_stats.items():
        if s['n_folds'] != 5:
            print(f"   WARNING: {config} has {s['n_folds']} folds instead of 5!")

    # Step 3: Write summary log
    print("\n3. Writing summary log...")
    write_summary_log(data_stats, model_stats)

    # Step 4: Update Excel files
    print("\n4. Updating Excel files...")
    update_data_perturb_excel(data_stats)
    print()
    update_model_perturb_excel(model_stats)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()

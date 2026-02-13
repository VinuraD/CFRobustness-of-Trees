"""Generate LaTeX comparison tables for German Credit dataset experiments."""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PERTURB_CSV = os.path.join(SCRIPT_DIR, "german_credit_data_perturb.csv")
MODEL_PERTURB_CSV = os.path.join(SCRIPT_DIR, "German_Credit_model_perturb.csv")
OUTPUT_TEX = os.path.join(SCRIPT_DIR, "german_credit_latex_tables.tex")

# --- Data perturbation CSV column layout ---
# Row 0 (header): method names at cols 3,7,11,15,19,23,27
# Row 1 (sub-header): Mean Validity, Std Validity, Mean Accuracy, Std Accuracy (repeated)
# Cols 0-2: Perturbation, Bin, Data %
DATA_METHODS = ["NICE", "DiCE", "cfxplorer", "CEML", "Feature Tweak", "OCEAN", "CERTS"]
DATA_METHOD_START_COLS = [3, 7, 11, 15, 19, 23, 27]  # Mean Validity column for each method

# --- Model perturbation CSV column layout ---
# Col 0: Model Configuration
# Methods start at cols 1,5,9,13,17
MODEL_METHODS = ["NICE", "DiCE", "Feature Tweak", "CEML", "CERTS"]
MODEL_METHOD_START_COLS = [1, 5, 9, 13, 17]  # Mean Validity column for each method

MODEL_TYPE_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGB",
    "adaboost": "Ada",
}

MODEL_TYPE_ORDER = ["random_forest", "xgboost", "lightgbm", "adaboost"]


def read_csv_raw(filepath):
    """Read CSV and return all rows as lists of strings."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def parse_float(val):
    """Parse a string to float, return None if empty or invalid."""
    val = val.strip()
    if val == "" or val == "-":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fmt_val(mean, std):
    """Format mean +/- std, bold-ready. Returns (formatted_str, mean_for_comparison)."""
    if mean is None or std is None:
        return ("-", None)
    return (f"{mean:.3f} $\\pm$ {std:.3f}", mean)


def bold_best(entries):
    """Given list of (formatted_str, mean_value), bold the best (highest mean)
    and underline the second best. Returns list of final formatted strings."""
    valid_means = [(i, m) for i, (_, m) in enumerate(entries) if m is not None]
    if not valid_means:
        return [s for s, _ in entries]

    # Find best and second-best means
    unique_means = sorted(set(m for _, m in valid_means), reverse=True)
    best_mean = unique_means[0]
    second_mean = unique_means[1] if len(unique_means) > 1 else None

    result = []
    for i, (s, m) in enumerate(entries):
        if m is not None and abs(m - best_mean) < 1e-9:
            result.append(f"\\textbf{{{s}}}")
        elif second_mean is not None and m is not None and abs(m - second_mean) < 1e-9:
            result.append(f"\\underline{{{s}}}")
        else:
            result.append(s)
    return result


def generate_data_perturb_table(rows, perturbation_type, caption, label):
    """Generate a LaTeX table for data perturbation results."""
    # Filter rows by perturbation type (skip header rows 0,1)
    data_rows = []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        if row[0].strip() == perturbation_type:
            data_rows.append(row)

    if not data_rows:
        return ""

    ncols = len(DATA_METHODS)
    col_spec = "cc" + "c" * ncols  # Bin, Data%, then one col per method

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header row
    header_cells = ["Bin", "Data \\%"] + DATA_METHODS
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")

    # Data rows
    for row in data_rows:
        bin_val = row[1].strip()
        data_pct = row[2].strip()

        entries = []
        for start_col in DATA_METHOD_START_COLS:
            mean = parse_float(row[start_col])
            std = parse_float(row[start_col + 1])
            entries.append(fmt_val(mean, std))

        formatted = bold_best(entries)
        line_cells = [bin_val, data_pct] + formatted
        lines.append(" & ".join(line_cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def parse_model_config(config_str):
    """Parse model config string like 'random_forest_3_100' -> (model_type, depth, n_estimators)."""
    config = config_str.strip()
    # Handle two-word model types
    for mt in MODEL_TYPE_ORDER:
        if config.startswith(mt + "_"):
            rest = config[len(mt) + 1:]
            parts = rest.split("_")
            if len(parts) == 2:
                return (mt, int(parts[0]), int(parts[1]))
    return (None, None, None)


def generate_model_perturb_table(rows, study_type, caption, label):
    """Generate a LaTeX table for model perturbation results.

    study_type: 'depth' (n_estimators=100, vary depth) or 'n_estimators' (depth=3, vary n_estimators)
    """
    # Parse all data rows (skip header rows 0,1)
    parsed = []
    for row in rows[2:]:
        if len(row) < 2 or row[0].strip() == "":
            continue
        model_type, depth, n_est = parse_model_config(row[0])
        if model_type is None:
            continue
        parsed.append((model_type, depth, n_est, row))

    # Filter by study type
    if study_type == "depth":
        filtered = [(mt, d, ne, row) for mt, d, ne, row in parsed if ne == 100]
        row_label_fn = lambda mt, d, ne: f"{MODEL_TYPE_LABELS[mt]} & {d}"
        vary_values = [3, 4, 5, 6]
        vary_key = lambda mt, d, ne: (MODEL_TYPE_ORDER.index(mt), d)
        extra_col_header = "Max Depth"
    else:  # n_estimators
        filtered = [(mt, d, ne, row) for mt, d, ne, row in parsed if d == 3]
        row_label_fn = lambda mt, d, ne: f"{MODEL_TYPE_LABELS[mt]} & {ne}"
        vary_values = [50, 100, 150, 200]
        vary_key = lambda mt, d, ne: (MODEL_TYPE_ORDER.index(mt), ne)
        extra_col_header = "$n_{\\text{estimators}}$"

    # Sort rows
    filtered.sort(key=lambda x: vary_key(x[0], x[1], x[2]))

    ncols = len(MODEL_METHODS)
    col_spec = "cc" + "c" * ncols  # Model, vary_param, then methods

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    # Header
    header_cells = ["Model", extra_col_header] + MODEL_METHODS
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")

    # Data rows
    prev_model = None
    for mt, d, ne, row in filtered:
        # Add separator between model types
        if prev_model is not None and mt != prev_model:
            lines.append("\\hline")
        prev_model = mt

        entries = []
        for start_col in MODEL_METHOD_START_COLS:
            mean = parse_float(row[start_col]) if start_col < len(row) else None
            std = parse_float(row[start_col + 1]) if start_col + 1 < len(row) else None
            entries.append(fmt_val(mean, std))

        formatted = bold_best(entries)
        row_label = row_label_fn(mt, d, ne)
        line_cells = [row_label] + formatted
        lines.append(" & ".join(line_cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    # Read CSVs
    data_rows = read_csv_raw(DATA_PERTURB_CSV)
    model_rows = read_csv_raw(MODEL_PERTURB_CSV)

    tables = []

    # Table 1: Data Perturbation - Minor Addition
    tables.append(generate_data_perturb_table(
        data_rows,
        "minor_addition",
        "German Credit -- Data Perturbation (Minor Addition): Mean Validity $\\pm$ Std",
        "tab:german_data_addition",
    ))

    # Table 2: Data Perturbation - Minor Deletion
    tables.append(generate_data_perturb_table(
        data_rows,
        "minor_deletion",
        "German Credit -- Data Perturbation (Minor Deletion): Mean Validity $\\pm$ Std",
        "tab:german_data_deletion",
    ))

    # Table 3: Model Perturbation - Max Depth Study (n_estimators=100)
    tables.append(generate_model_perturb_table(
        model_rows,
        "depth",
        "German Credit -- Model Perturbation (Max Depth Study, $n_{\\text{estimators}}=100$): Mean Validity $\\pm$ Std",
        "tab:german_model_depth",
    ))

    # Table 4: Model Perturbation - N_estimators Study (max_depth=3)
    tables.append(generate_model_perturb_table(
        model_rows,
        "n_estimators",
        "German Credit -- Model Perturbation ($n_{\\text{estimators}}$ Study, max\\_depth$=3$): Mean Validity $\\pm$ Std",
        "tab:german_model_nestimators",
    ))

    # Write output
    with open(OUTPUT_TEX, "w") as f:
        f.write("% Auto-generated LaTeX tables for German Credit dataset\n")
        f.write("% Generated by generate_latex_tables.py\n\n")
        for i, table in enumerate(tables):
            if table:
                f.write(table)
                f.write("\n\n")

    print(f"LaTeX tables written to {OUTPUT_TEX}")


if __name__ == "__main__":
    main()

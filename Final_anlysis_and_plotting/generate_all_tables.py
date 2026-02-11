"""
Generate LaTeX comparison tables for Spambase, HELOC, and COMPAS datasets.

Steps:
1. Convert xlsx files to csv (preserving 2-row header structure)
2. Parse ARMOR log files to extract data & model perturbation results
3. Append ARMOR columns to the CSVs
4. Generate LaTeX tables for each dataset
5. Create standalone tex wrappers
"""

import csv
import os
import re
import statistics

import openpyxl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ─── Dataset configurations ──────────────────────────────────────────────────

DATASETS = {
    "spambase": {
        "label": "Spambase",
        "data_perturb_xlsx": os.path.join(SCRIPT_DIR, "Spambase_data_perturb.xlsx"),
        "model_perturb_xlsx": os.path.join(SCRIPT_DIR, "Spambase_model_perturb.xlsx"),
        "data_perturb_csv": os.path.join(SCRIPT_DIR, "spambase_data_perturb.csv"),
        "model_perturb_csv": os.path.join(SCRIPT_DIR, "spambase_model_perturb.csv"),
        "armor_log": os.path.join(
            PROJECT_ROOT, "experiment_outputs", "ARMOR_v2", "ARMOR_v2.log"
        ),
        "latex_tables": os.path.join(SCRIPT_DIR, "spambase_latex_tables.tex"),
        "standalone_tex": os.path.join(SCRIPT_DIR, "spambase_tables_standalone.tex"),
    },
    "heloc": {
        "label": "HELOC",
        "data_perturb_xlsx": os.path.join(SCRIPT_DIR, "heloc_data_perturb.xlsx"),
        "model_perturb_xlsx": os.path.join(SCRIPT_DIR, "Heloc_model_perturb.xlsx"),
        "data_perturb_csv": os.path.join(SCRIPT_DIR, "heloc_data_perturb.csv"),
        "model_perturb_csv": os.path.join(SCRIPT_DIR, "heloc_model_perturb.csv"),
        "armor_log": os.path.join(
            PROJECT_ROOT,
            "experiment_outputs",
            "ARMOR_v4_heloc",
            "ARMOR_v4_heloc.log",
        ),
        "latex_tables": os.path.join(SCRIPT_DIR, "heloc_latex_tables.tex"),
        "standalone_tex": os.path.join(SCRIPT_DIR, "heloc_tables_standalone.tex"),
    },
    "compas": {
        "label": "COMPAS",
        "data_perturb_xlsx": os.path.join(SCRIPT_DIR, "compas_data_perturb.xlsx"),
        "model_perturb_xlsx": os.path.join(SCRIPT_DIR, "Compas_model_perturb.xlsx"),
        "data_perturb_csv": os.path.join(SCRIPT_DIR, "compas_data_perturb.csv"),
        "model_perturb_csv": os.path.join(SCRIPT_DIR, "compas_model_perturb.csv"),
        "armor_log": os.path.join(
            PROJECT_ROOT,
            "experiment_outputs",
            "ARMOR_v5_compas",
            "ARMOR_v5_compas.log",
        ),
        "latex_tables": os.path.join(SCRIPT_DIR, "compas_latex_tables.tex"),
        "standalone_tex": os.path.join(SCRIPT_DIR, "compas_tables_standalone.tex"),
    },
}

# ─── Data perturbation CSV layout ────────────────────────────────────────────
# Row 0 (header): method names at cols 3,7,11,15,19,23,27
# Row 1 (sub-header): Mean Validity, Std Validity, Mean Accuracy, Std Accuracy
# Cols 0-2: Perturbation, Bin, Data %
DATA_METHODS = ["NICE", "DiCE", "cfxplorer", "CEML", "Feature Tweak", "OCEAN", "ARMOR"]
DATA_METHOD_START_COLS = [3, 7, 11, 15, 19, 23, 27]

# ─── Model perturbation CSV layout ───────────────────────────────────────────
# Col 0: Model Configuration
# Methods start at cols 1,5,9,13,17
MODEL_METHODS = ["NICE", "DiCE", "Feature Tweak", "CEML", "ARMOR"]
MODEL_METHOD_START_COLS = [1, 5, 9, 13, 17]

MODEL_TYPE_LABELS = {
    "random_forest": "RF",
    "xgboost": "XGB",
    "lightgbm": "LGB",
    "adaboost": "Ada",
}
MODEL_TYPE_ORDER = ["random_forest", "xgboost", "lightgbm", "adaboost"]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Convert XLSX → CSV
# ═══════════════════════════════════════════════════════════════════════════════


def xlsx_to_csv(xlsx_path, csv_path):
    """Convert an xlsx file to csv, preserving the 2-row header structure."""
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    rows = []
    for row in ws.iter_rows(values_only=True):
        rows.append(list(row))
    wb.close()
    return rows


def write_csv(rows, csv_path):
    """Write rows to CSV."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Parse ARMOR logs
# ═══════════════════════════════════════════════════════════════════════════════


def parse_armor_log(log_path):
    """Parse an ARMOR log file and return data and model perturbation results.

    Returns:
        data_perturb: dict[(perturbation_type, bin_num)] -> list of (validity, accuracy)
                      across folds
        model_perturb: dict[(model_type, depth, n_estimators)] -> list of (validity, accuracy)
                       across folds
    """
    with open(log_path, "r") as f:
        content = f.read()

    data_perturb = {}  # (perturb_type, bin) -> [(validity, accuracy), ...]
    model_perturb = {}  # (model_type, depth, n_est) -> [(validity, accuracy), ...]

    # Split by fold
    fold_sections = re.split(r"--- FOLD \d+ ANALYSIS ---", content)

    for section in fold_sections[1:]:  # skip preamble before fold 0
        # Parse data perturbation results
        _parse_data_perturb_section(section, data_perturb)
        # Parse model perturbation results
        _parse_model_perturb_section(section, model_perturb)

    return data_perturb, model_perturb


def _parse_data_perturb_section(section, data_perturb):
    """Extract data perturbation results from a single fold section."""
    # minor_deletion entries
    del_match = re.search(r"minor_deletion:(.*?)(?:minor_addition:|Testing model)", section, re.DOTALL)
    if del_match:
        _extract_bin_results(del_match.group(1), "minor_deletion", data_perturb)

    # minor_addition entries
    add_match = re.search(r"minor_addition:(.*?)(?:Testing model|--- FOLD|\Z)", section, re.DOTALL)
    if add_match:
        _extract_bin_results(add_match.group(1), "minor_addition", data_perturb)


def _extract_bin_results(text, perturb_type, data_perturb):
    """Extract bin results from a text block."""
    pattern = r"Bin (\d+): (?:Remove|Use) \d+% -> validity: ([\d.]+), accuracy: ([\d.]+)"
    for m in re.finditer(pattern, text):
        bin_num = int(m.group(1))
        validity = float(m.group(2))
        accuracy = float(m.group(3))
        key = (perturb_type, bin_num)
        data_perturb.setdefault(key, []).append((validity, accuracy))


def _parse_model_perturb_section(section, model_perturb):
    """Extract model perturbation results from a single fold section."""
    model_section = re.search(r"Testing model perturbations.*?(?:--- FOLD|\Z)", section, re.DOTALL)
    if not model_section:
        return

    text = model_section.group(0)

    # Pattern for result lines (not the "testing" header lines)
    # e.g.: "      random_forest {'max_depth': 3, 'n_estimators': 100, ...}: validity 0.7744, accuracy 0.9034, ..."
    pattern = (
        r"^\s+(\w+) \{.*?'max_depth': (\d+).*?'n_estimators': (\d+).*?\}:\s+"
        r"validity ([\d.]+), accuracy ([\d.]+)"
    )
    # Also handle catboost format: 'depth': X, 'iterations': Y
    pattern_catboost = (
        r"^\s+catboost \{.*?'depth': (\d+).*?'iterations': (\d+).*?\}:\s+"
        r"validity ([\d.]+), accuracy ([\d.]+)"
    )

    for m in re.finditer(pattern, text, re.MULTILINE):
        model_type = m.group(1)
        if model_type == "catboost":
            continue  # skip catboost
        depth = int(m.group(2))
        n_est = int(m.group(3))
        validity = float(m.group(4))
        accuracy = float(m.group(5))
        key = (model_type, depth, n_est)
        model_perturb.setdefault(key, []).append((validity, accuracy))


def compute_mean_std(values):
    """Compute mean and std of a list of values. Uses population std (ddof=0) like numpy."""
    if not values:
        return None, None
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0
    # Use population std (ddof=0) to match numpy behavior
    std = statistics.pstdev(values)
    return mean, std


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Build CSVs with ARMOR columns appended
# ═══════════════════════════════════════════════════════════════════════════════


def build_data_perturb_csv(xlsx_rows, armor_data):
    """Add ARMOR columns to data perturbation xlsx rows.

    xlsx_rows: raw rows from xlsx (6 methods, no ARMOR)
    armor_data: dict[(perturb_type, bin)] -> list of (validity, accuracy) across folds

    Returns rows with ARMOR appended (4 extra columns).
    """
    result = []

    # Row 0: method header - add ARMOR
    row0 = list(xlsx_rows[0])
    # Extend to ensure we have enough columns, then add ARMOR header
    while len(row0) < 27:
        row0.append(None)
    row0.extend(["ARMOR", None, None, None])
    result.append(row0)

    # Row 1: sub-header - add Mean Validity, Std Validity, Mean Accuracy, Std Accuracy
    row1 = list(xlsx_rows[1])
    while len(row1) < 27:
        row1.append(None)
    row1.extend(["Mean Validity", "Std Validity", "Mean Accuracy", "Std Accuracy"])
    result.append(row1)

    # Data rows
    for row in xlsx_rows[2:]:
        new_row = list(row)
        # Extend to 27 cols if needed
        while len(new_row) < 27:
            new_row.append(None)

        perturb_type = str(row[0]).strip() if row[0] else ""
        bin_num = int(row[1]) if row[1] is not None else None

        if perturb_type and bin_num is not None:
            # For minor_addition, invert ARMOR bin mapping:
            # ARMOR log Bin 0 = baseline (50% data), Bin 50 = 100% data
            # CSV convention: Bin 50 = baseline (100% data), Bin 0 = 50% data
            # So CSV Bin X should use ARMOR log Bin (50 - X)
            if perturb_type == "minor_addition":
                armor_bin = 50 - bin_num
            else:
                armor_bin = bin_num
            key = (perturb_type, armor_bin)
            if key in armor_data:
                vals = armor_data[key]
                validities = [v for v, a in vals]
                accuracies = [a for v, a in vals]
                mv, sv = compute_mean_std(validities)
                ma, sa = compute_mean_std(accuracies)
                new_row.extend([mv, sv, ma, sa])
            else:
                new_row.extend([None, None, None, None])
        else:
            new_row.extend([None, None, None, None])

        result.append(new_row)

    return result


def build_model_perturb_csv(xlsx_rows, armor_model):
    """Add ARMOR columns to model perturbation xlsx rows.

    xlsx_rows: raw rows from xlsx (4 methods: NICE, DiCE, Feature Tweak, CEML)
    armor_model: dict[(model_type, depth, n_est)] -> list of (validity, accuracy)

    Returns rows with ARMOR appended (4 extra columns).
    """
    result = []

    # Row 0: method header
    row0 = list(xlsx_rows[0])
    while len(row0) < 17:
        row0.append(None)
    row0.extend(["ARMOR", None, None, None])
    result.append(row0)

    # Row 1: sub-header
    row1 = list(xlsx_rows[1])
    while len(row1) < 17:
        row1.append(None)
    row1.extend(["Mean Validity", "Std Validity", "Mean Accuracy", "Std Accuracy"])
    result.append(row1)

    # Data rows
    for row in xlsx_rows[2:]:
        new_row = list(row)
        while len(new_row) < 17:
            new_row.append(None)

        config_str = str(row[0]).strip() if row[0] else ""
        model_type, depth, n_est = parse_model_config(config_str)

        if model_type is not None:
            key = (model_type, depth, n_est)
            if key in armor_model:
                vals = armor_model[key]
                validities = [v for v, a in vals]
                accuracies = [a for v, a in vals]
                mv, sv = compute_mean_std(validities)
                ma, sa = compute_mean_std(accuracies)
                new_row.extend([mv, sv, ma, sa])
            else:
                new_row.extend([None, None, None, None])
        else:
            new_row.extend([None, None, None, None])

        result.append(new_row)

    return result


def parse_model_config(config_str):
    """Parse model config string like 'random_forest_3_100' -> (model_type, depth, n_estimators)."""
    config = config_str.strip()
    for mt in MODEL_TYPE_ORDER:
        if config.startswith(mt + "_"):
            rest = config[len(mt) + 1 :]
            parts = rest.split("_")
            if len(parts) == 2:
                try:
                    return (mt, int(parts[0]), int(parts[1]))
                except ValueError:
                    pass
    return (None, None, None)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Generate LaTeX tables
# ═══════════════════════════════════════════════════════════════════════════════


def read_csv_raw(filepath):
    """Read CSV and return all rows as lists of strings."""
    with open(filepath, "r") as f:
        reader = csv.reader(f)
        return [row for row in reader]


def parse_float(val):
    """Parse a string to float, return None if empty or invalid."""
    if val is None:
        return None
    val = str(val).strip()
    if val == "" or val == "-" or val == "—" or val == "None":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def fmt_val(mean, std):
    """Format mean +/- std. Returns (formatted_str, mean_for_comparison)."""
    if mean is None or std is None:
        return ("-", None)
    return (f"{mean:.3f} $\\pm$ {std:.3f}", mean)


def bold_best(entries):
    """Given list of (formatted_str, mean_value), bold the best (highest mean)
    and underline the second best. Returns list of final formatted strings."""
    valid_means = [(i, m) for i, (_, m) in enumerate(entries) if m is not None]
    if not valid_means:
        return [s for s, _ in entries]

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
    data_rows = []
    for row in rows[2:]:
        if len(row) < 3:
            continue
        if str(row[0]).strip() == perturbation_type:
            data_rows.append(row)

    if not data_rows:
        return ""

    ncols = len(DATA_METHODS)
    col_spec = "cc" + "c" * ncols

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header_cells = ["Bin", "Data \\%"] + DATA_METHODS
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")

    for row in data_rows:
        bin_val = str(row[1]).strip()
        data_pct = str(row[2]).strip()

        entries = []
        for start_col in DATA_METHOD_START_COLS:
            mean = parse_float(row[start_col]) if start_col < len(row) else None
            std = parse_float(row[start_col + 1]) if start_col + 1 < len(row) else None
            entries.append(fmt_val(mean, std))

        formatted = bold_best(entries)
        line_cells = [bin_val, data_pct] + formatted
        lines.append(" & ".join(line_cells) + " \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def generate_model_perturb_table(rows, study_type, caption, label):
    """Generate a LaTeX table for model perturbation results.

    study_type: 'depth' (n_estimators=100, vary depth) or 'n_estimators' (depth=3, vary n_estimators)
    """
    parsed = []
    for row in rows[2:]:
        if len(row) < 2 or row[0] is None or str(row[0]).strip() == "":
            continue
        model_type, depth, n_est = parse_model_config(str(row[0]))
        if model_type is None:
            continue
        parsed.append((model_type, depth, n_est, row))

    if study_type == "depth":
        filtered = [(mt, d, ne, row) for mt, d, ne, row in parsed if ne == 100]
        row_label_fn = lambda mt, d, ne: f"{MODEL_TYPE_LABELS[mt]} & {d}"
        vary_key = lambda mt, d, ne: (MODEL_TYPE_ORDER.index(mt), d)
        extra_col_header = "Max Depth"
    else:
        filtered = [(mt, d, ne, row) for mt, d, ne, row in parsed if d == 3]
        row_label_fn = lambda mt, d, ne: f"{MODEL_TYPE_LABELS[mt]} & {ne}"
        vary_key = lambda mt, d, ne: (MODEL_TYPE_ORDER.index(mt), ne)
        extra_col_header = "$n_{\\text{estimators}}$"

    filtered.sort(key=lambda x: vary_key(x[0], x[1], x[2]))

    ncols = len(MODEL_METHODS)
    col_spec = "cc" + "c" * ncols

    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\hline")

    header_cells = ["Model", extra_col_header] + MODEL_METHODS
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\hline")

    prev_model = None
    for mt, d, ne, row in filtered:
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


def generate_standalone_tex(latex_tables_filename, standalone_path):
    """Create a standalone LaTeX wrapper for compiling."""
    content = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=0.5in, landscape]{geometry}\n"
        "\\usepackage{graphicx}\n"
        "\\usepackage{amsmath}\n"
        "\n"
        "\\begin{document}\n"
        "\n"
        f"\\input{{{latex_tables_filename}}}\n"
        "\n"
        "\\end{document}\n"
    )
    with open(standalone_path, "w") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def format_csv_value(val):
    """Format a value for CSV output."""
    if val is None:
        return ""
    return str(val)


def process_dataset(name, cfg):
    """Process a single dataset: xlsx->csv, parse log, add ARMOR, generate LaTeX."""
    print(f"\n{'='*60}")
    print(f"Processing {cfg['label']} dataset")
    print(f"{'='*60}")

    # Step 1: Read xlsx files
    print(f"  Reading {cfg['data_perturb_xlsx']}...")
    data_xlsx_rows = xlsx_to_csv(cfg["data_perturb_xlsx"], cfg["data_perturb_csv"])
    print(f"    {len(data_xlsx_rows)} rows (including 2 header rows)")

    print(f"  Reading {cfg['model_perturb_xlsx']}...")
    model_xlsx_rows = xlsx_to_csv(cfg["model_perturb_xlsx"], cfg["model_perturb_csv"])
    # Filter out empty/None rows at the end (HELOC has extra rows)
    model_xlsx_rows_clean = []
    for row in model_xlsx_rows:
        if row[0] is not None or model_xlsx_rows.index(row) < 2:
            model_xlsx_rows_clean.append(row)
        else:
            break
    print(f"    {len(model_xlsx_rows_clean)} rows (including 2 header rows)")

    # Step 2: Parse ARMOR log
    print(f"  Parsing ARMOR log: {cfg['armor_log']}...")
    armor_data, armor_model = parse_armor_log(cfg["armor_log"])
    print(f"    Data perturbation: {len(armor_data)} (type, bin) combinations")
    print(f"    Model perturbation: {len(armor_model)} (model, depth, n_est) combinations")

    # Verify 5 folds
    for key, vals in armor_data.items():
        if len(vals) != 5:
            print(f"    WARNING: {key} has {len(vals)} folds (expected 5)")

    # Step 3: Build CSVs with ARMOR
    print("  Building data perturbation CSV with ARMOR...")
    data_csv_rows = build_data_perturb_csv(data_xlsx_rows, armor_data)
    write_csv(
        [[format_csv_value(v) for v in row] for row in data_csv_rows],
        cfg["data_perturb_csv"],
    )
    print(f"    Written to {cfg['data_perturb_csv']}")

    print("  Building model perturbation CSV with ARMOR...")
    model_csv_rows = build_model_perturb_csv(model_xlsx_rows_clean, armor_model)
    write_csv(
        [[format_csv_value(v) for v in row] for row in model_csv_rows],
        cfg["model_perturb_csv"],
    )
    print(f"    Written to {cfg['model_perturb_csv']}")

    # Step 4: Generate LaTeX tables
    print("  Generating LaTeX tables...")
    data_rows = read_csv_raw(cfg["data_perturb_csv"])
    model_rows = read_csv_raw(cfg["model_perturb_csv"])

    dataset_label = cfg["label"]
    prefix = name

    tables = []

    tables.append(
        generate_data_perturb_table(
            data_rows,
            "minor_addition",
            f"{dataset_label} -- Data Perturbation (Minor Addition): Mean Validity $\\pm$ Std",
            f"tab:{prefix}_data_addition",
        )
    )

    tables.append(
        generate_data_perturb_table(
            data_rows,
            "minor_deletion",
            f"{dataset_label} -- Data Perturbation (Minor Deletion): Mean Validity $\\pm$ Std",
            f"tab:{prefix}_data_deletion",
        )
    )

    tables.append(
        generate_model_perturb_table(
            model_rows,
            "depth",
            f"{dataset_label} -- Model Perturbation (Max Depth Study, $n_{{\\text{{estimators}}}}=100$): Mean Validity $\\pm$ Std",
            f"tab:{prefix}_model_depth",
        )
    )

    tables.append(
        generate_model_perturb_table(
            model_rows,
            "n_estimators",
            f"{dataset_label} -- Model Perturbation ($n_{{\\text{{estimators}}}}$ Study, max\\_depth$=3$): Mean Validity $\\pm$ Std",
            f"tab:{prefix}_model_nestimators",
        )
    )

    with open(cfg["latex_tables"], "w") as f:
        f.write(f"% Auto-generated LaTeX tables for {dataset_label} dataset\n")
        f.write("% Generated by generate_all_tables.py\n\n")
        for table in tables:
            if table:
                f.write(table)
                f.write("\n\n")

    print(f"    LaTeX tables written to {cfg['latex_tables']}")

    # Step 5: Create standalone wrapper
    latex_basename = os.path.basename(cfg["latex_tables"])
    generate_standalone_tex(latex_basename, cfg["standalone_tex"])
    print(f"    Standalone tex written to {cfg['standalone_tex']}")

    # Print sample ARMOR values for verification
    print("\n  Sample ARMOR values for verification:")
    sample_keys = list(armor_data.keys())[:3]
    for key in sample_keys:
        vals = armor_data[key]
        validities = [v for v, a in vals]
        mv, sv = compute_mean_std(validities)
        print(f"    {key}: validity mean={mv:.4f}, std={sv:.4f} (from {len(vals)} folds)")

    sample_model_keys = list(armor_model.keys())[:3]
    for key in sample_model_keys:
        vals = armor_model[key]
        validities = [v for v, a in vals]
        mv, sv = compute_mean_std(validities)
        print(f"    {key}: validity mean={mv:.4f}, std={sv:.4f} (from {len(vals)} folds)")


def main():
    for name, cfg in DATASETS.items():
        process_dataset(name, cfg)

    print(f"\n{'='*60}")
    print("All datasets processed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

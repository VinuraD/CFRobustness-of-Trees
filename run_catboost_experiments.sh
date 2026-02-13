#!/bin/bash
# Master script to run all CatBoost model perturbation experiments
# CatBoost code uses task_type='GPU', devices='3' directly - no CUDA_VISIBLE_DEVICES needed
# Runs 2 datasets at a time per method
# Safe to run with nohup - survives terminal closure

set -e
PROJ=/mnt/datassd3/rashinda/CF_Robustness/CFRobustness-of-Trees
LOGS=$PROJ/logs

# IMPORTANT: Do NOT set CUDA_VISIBLE_DEVICES - CatBoost uses devices='3' directly
unset CUDA_VISIBLE_DEVICES

mkdir -p "$LOGS"

run_pair() {
    local dir1=$1 script1=$2 log1=$3 args1=$4
    local dir2=$5 script2=$6 log2=$7 args2=$8

    echo "[$(date '+%H:%M:%S')] Starting: $log1 + $log2"

    (cd "$PROJ/$dir1" && conda run -n cf-robustness python "$script1" $args1) > "$LOGS/$log1" 2>&1 &
    local pid1=$!

    (cd "$PROJ/$dir2" && conda run -n cf-robustness python "$script2" $args2) > "$LOGS/$log2" 2>&1 &
    local pid2=$!

    wait $pid1
    local rc1=$?
    wait $pid2
    local rc2=$?

    echo "[$(date '+%H:%M:%S')] Done: $log1 (exit=$rc1), $log2 (exit=$rc2)"
}

echo "=========================================="
echo "CatBoost Model Perturbation Experiments"
echo "Started: $(date)"
echo "GPU: device 3 (configured in CatBoost code)"
echo "=========================================="

# --- DiCE ---
echo ""
echo "--- DiCE ---"
run_pair \
    DiCE cf_robustness_analysis_v2.py dice_v2_catboost.log "--model-type catboost --skip-data-perturbation" \
    DiCE cf_robustness_analysis_v3.py dice_v3_catboost.log "--model-type catboost --skip-data-perturbation"

run_pair \
    DiCE cf_robustness_analysis_v4_heloc.py dice_v4_catboost.log "--model-type catboost --skip-data-perturbation" \
    DiCE cf_robustness_analysis_v5_compas.py dice_v5_catboost.log "--model-type catboost --skip-data-perturbation"

# --- NICE ---
echo ""
echo "--- NICE ---"
run_pair \
    NICE cf_robustness_analysis_nice_v2.py nice_v2_catboost.log "--model-type catboost --skip-data-perturbation" \
    NICE cf_robustness_analysis_nice_v3.py nice_v3_catboost.log "--model-type catboost --skip-data-perturbation"

run_pair \
    NICE cf_robustness_analysis_nice_v4_heloc.py nice_v4_catboost.log "--model-type catboost --skip-data-perturbation" \
    NICE cf_robustness_analysis_nice_v5_compas.py nice_v5_catboost.log "--model-type catboost --skip-data-perturbation"

# --- CEML ---
echo ""
echo "--- CEML ---"
run_pair \
    CEML cf_robustness_analysis_ceml_v2.py ceml_v2_catboost.log "--model-type catboost --skip-data-perturbation" \
    CEML cf_robustness_analysis_ceml_v3.py ceml_v3_catboost.log "--model-type catboost --skip-data-perturbation"

run_pair \
    CEML cf_robustness_analysis_ceml_v4_heloc.py ceml_v4_catboost.log "--model-type catboost --skip-data-perturbation" \
    CEML cf_robustness_analysis_ceml_v5_compas.py ceml_v5_catboost.log "--model-type catboost --skip-data-perturbation"

# --- FeatureTweak ---
echo ""
echo "--- FeatureTweak ---"
run_pair \
    feature_tweak cf_robustness_analysis_featuretweak_v2.py ft_v2_catboost.log "--skip-data-perturbation --model-perturbation-filter catboost" \
    feature_tweak cf_robustness_analysis_featuretweak_v3.py ft_v3_catboost.log "--skip-data-perturbation --model-perturbation-filter catboost"

run_pair \
    feature_tweak cf_robustness_analysis_featuretweak_v4_heloc.py ft_v4_catboost.log "--skip-data-perturbation --model-perturbation-filter catboost" \
    feature_tweak cf_robustness_analysis_featuretweak_v5_compas.py ft_v5_catboost.log "--skip-data-perturbation --model-perturbation-filter catboost"

echo ""
echo "=========================================="
echo "All CatBoost experiments completed!"
echo "Finished: $(date)"
echo "=========================================="
echo "Logs in: $LOGS/"
ls -la "$LOGS"/*catboost*.log

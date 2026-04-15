#!/bin/bash
# run_all_experiments.sh
# Master script for BPAN revised paper experiments.
#
# Runs the full experimental pipeline:
#   1. Parameter sweeps (BPAN vs MLP) across all datasets and hidden widths
#   2. Anytime comparison (BPAN, ACT, PonderNet, Multi-Exit) on MNIST, F-MNIST, SVHN
#   3. Balance regulariser ablation study
#   4. Sensitivity analysis (T, lambda_bal, E/I ratio)
#   5. Sequential evidence integration (BPAN, GRU, LSTM, Transformer)
#   6. Pareto plot generation
#
# Usage:
#   chmod +x run_all_experiments.sh
#   ./run_all_experiments.sh [--epochs 15] [--quick]
#
# Options:
#   --epochs: Number of training epochs (default: 15)
#   --quick: Run quick experiments with fewer epochs (5)

set -e  # Exit on error

# Activate the jeet venv
source ~/venvs/jeet/bin/activate
echo "Python: $(python --version)"
echo "Venv: $VIRTUAL_ENV"

# Default parameters
EPOCHS=15
HIDDEN=256
BPAN_T=6
BATCH_SIZE=256
LR=0.001
SEED=42

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --quick)
            EPOCHS=5
            shift
            ;;
        --hidden)
            HIDDEN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "BPAN Revised Paper: Full Experiment Pipeline"
echo "========================================"
echo "Epochs: $EPOCHS"
echo "Hidden: $HIDDEN"
echo "BPAN T: $BPAN_T"
echo "Batch size: $BATCH_SIZE"
echo "========================================"

# Create output directories
mkdir -p results
mkdir -p results_anytime_all
mkdir -p results_balance_analysis
mkdir -p results_sensitivity
mkdir -p results_sequential
mkdir -p plots

# ========================================
# 1. Parameter sweeps: BPAN vs MLP across all datasets
# ========================================
echo ""
echo ">>> Step 1: Parameter sweeps (BPAN vs MLP)..."
for DATASET in mnist fashion_mnist cifar10 svhn; do
    echo "  Dataset: $DATASET"
    python run_param_sweep.py \
        --dataset $DATASET \
        --hidden_list 64 128 256 512 \
        --seeds 1 2 3 \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --bpan_T $BPAN_T \
        --out_prefix ./results
done

# ========================================
# 2. (removed) E/I ratio sweep — superseded by Step 5's sensitivity analysis,
#     which sweeps E/I ratio alongside T and lambda_bal via
#     run_sensitivity_analysis.py.
# ========================================

# ========================================
# 3. Anytime comparison on MNIST, Fashion-MNIST, SVHN
# ========================================
echo ""
echo ">>> Step 3: Anytime comparison (BPAN + baselines)..."
for DATASET in mnist fashion_mnist svhn; do
    echo ""
    echo "  === $DATASET ==="

    # BPAN
    echo "  Training BPAN anytime..."
    python run_anytime_bpan.py \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --bpan_T $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_anytime_all \
        --thresholds "0.5,0.6,0.7,0.8,0.9,0.95,0.99"

    # ACT
    echo "  Training ACT..."
    python run_anytime_baselines.py \
        --dataset $DATASET \
        --model act \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --max_steps $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_anytime_all \
        --act_time_penalty 0.01

    # PonderNet
    echo "  Training PonderNet..."
    python run_anytime_baselines.py \
        --dataset $DATASET \
        --model pondernet \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --max_steps $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_anytime_all \
        --ponder_lambda_p 0.3 \
        --ponder_beta 0.01

    # Multi-Exit
    echo "  Training Multi-Exit..."
    python run_anytime_baselines.py \
        --dataset $DATASET \
        --model multi_exit \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --max_steps $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_anytime_all
done

# ========================================
# 4. Balance Regulariser Ablation
# ========================================
echo ""
echo ">>> Step 4: Balance regulariser ablation..."
for DATASET in mnist fashion_mnist; do
    echo "  Dataset: $DATASET"
    python analyze_balance_regularizer.py \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --bpan_T $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --lambda_bal 5e-4 \
        --seed $SEED \
        --out_dir ./results_balance_analysis
done

# ========================================
# 5. Sensitivity Analysis
# ========================================
echo ""
echo ">>> Step 5: Sensitivity analysis (T, lambda_bal, E/I ratio)..."
for DATASET in mnist fashion_mnist; do
    echo "  Dataset: $DATASET"
    python run_sensitivity_analysis.py \
        --dataset $DATASET \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seeds 1 2 3 \
        --sweep all \
        --out_dir ./results_sensitivity
done

# ========================================
# 6. Sequential Evidence Integration Tasks
# ========================================
echo ""
echo ">>> Step 6: Sequential evidence integration tasks..."

# CT-MNIST
echo "  CT-MNIST experiments..."
for MODEL in bpan gru lstm transformer; do
    echo "    Model: $MODEL"
    python run_sequential_task.py \
        --task ctmnist \
        --model $MODEL \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --n_glimpses $BPAN_T \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_sequential
done

# Streaming classification
echo "  Streaming classification experiments..."
for MODEL in bpan gru lstm transformer; do
    echo "    Model: $MODEL"
    python run_sequential_task.py \
        --task streaming \
        --model $MODEL \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --n_glimpses 8 \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_sequential
done

# Sequential CIFAR patches
echo "  Sequential CIFAR patches experiments..."
for MODEL in bpan gru lstm transformer; do
    echo "    Model: $MODEL"
    python run_sequential_task.py \
        --task cifar_patches \
        --model $MODEL \
        --epochs $EPOCHS \
        --hidden $HIDDEN \
        --n_glimpses 8 \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --seed $SEED \
        --out_dir ./results_sequential
done

# ========================================
# 7. Generate Pareto Plots
# ========================================
echo ""
echo ">>> Step 7: Generating Pareto plots..."
for DATASET in mnist fashion_mnist svhn; do
    python plot_pareto.py \
        --results_dir ./results_anytime_all \
        --dataset $DATASET \
        --out_dir ./plots \
        --max_steps $BPAN_T
done

# ========================================
# Summary
# ========================================
echo ""
echo "========================================"
echo "All experiments completed!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Parameter sweeps: ./results/"
echo "  - Anytime results: ./results_anytime_all/"
echo "  - Balance analysis: ./results_balance_analysis/"
echo "  - Sensitivity analysis: ./results_sensitivity/"
echo "  - Sequential tasks: ./results_sequential/"
echo "  - Plots: ./plots/"
echo ""
echo "Datasets covered: MNIST, Fashion-MNIST, CIFAR-10, SVHN"
echo "Anytime baselines: BPAN, ACT, PonderNet, Multi-Exit"
echo "Sequential baselines: BPAN, GRU, LSTM, Transformer"
echo ""

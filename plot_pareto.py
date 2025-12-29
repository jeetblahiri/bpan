# plot_pareto.py
"""
Generate Pareto plots comparing BPAN vs anytime baselines.
Shows accuracy vs expected compute (average steps).

Usage:
    python plot_pareto.py --results_dir ./results_anytime_all --dataset mnist
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load_anytime_results(results_dir, dataset):
    """
    Load all anytime results for a given dataset.
    Returns: dict[model_name] -> list of {threshold, acc, avg_steps, step_hist}
    """
    results_dir = Path(results_dir)
    results = defaultdict(list)
    
    # Find all JSONL files for this dataset
    for jsonl_path in results_dir.glob(f"*{dataset}*.jsonl"):
        with open(jsonl_path, "r") as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("type") == "anytime":
                    # Get model name - check explicit "model" key first,
                    # then infer from "bpan_T" for legacy BPAN files
                    if "model" in rec:
                        model = rec["model"]
                    elif "bpan_T" in rec:
                        model = "bpan"  # Legacy BPAN files without explicit model key
                    else:
                        model = "unknown"
                    
                    results[model].append({
                        "threshold": rec["threshold"],
                        "acc": rec["acc"],
                        "avg_steps": rec["avg_steps"],
                        "step_hist": rec.get("step_hist", {}),
                    })
    
    return dict(results)


def compute_pareto_frontier(points):
    """
    Given list of (avg_steps, acc) points, return the Pareto frontier.
    A point is Pareto-optimal if no other point has both lower steps AND higher acc.
    """
    # Sort by avg_steps ascending
    sorted_pts = sorted(points, key=lambda p: p[0])
    
    frontier = []
    max_acc_so_far = -1
    
    for steps, acc in sorted_pts:
        if acc > max_acc_so_far:
            frontier.append((steps, acc))
            max_acc_so_far = acc
    
    return frontier


def plot_pareto_curves(results, dataset, save_path=None, max_steps=6):
    """
    Plot accuracy vs avg_steps for all models.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Color and marker scheme
    model_styles = {
        "bpan": {"color": "#2E86AB", "marker": "o", "label": "BPAN (ours)"},
        "act": {"color": "#A23B72", "marker": "s", "label": "ACT"},
        "pondernet": {"color": "#F18F01", "marker": "^", "label": "PonderNet"},
        "multi_exit": {"color": "#C73E1D", "marker": "D", "label": "Multi-Exit"},
    }
    
    for model_name, data_list in results.items():
        if not data_list:
            continue
        
        style = model_styles.get(model_name, {"color": "gray", "marker": "x", "label": model_name})
        
        # Extract points
        points = [(d["avg_steps"], d["acc"]) for d in data_list]
        
        # Sort by threshold (which correlates with avg_steps)
        points = sorted(points, key=lambda p: p[0])
        
        steps = [p[0] for p in points]
        accs = [p[1] for p in points]
        
        # Plot curve
        ax.plot(steps, accs, 
                color=style["color"], 
                marker=style["marker"],
                markersize=8,
                linewidth=2,
                label=style["label"],
                alpha=0.9)
        
        # Compute and plot Pareto frontier
        frontier = compute_pareto_frontier(points)
        if len(frontier) > 1:
            f_steps = [p[0] for p in frontier]
            f_accs = [p[1] for p in frontier]
            ax.plot(f_steps, f_accs, 
                    color=style["color"], 
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.5)
    
    # Add reference lines
    ax.axvline(x=max_steps, color='gray', linestyle=':', alpha=0.5, 
               label=f'Max steps ({max_steps})')
    
    ax.set_xlabel("Average Computation Steps", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title(f"Anytime Accuracy vs Compute — {dataset.upper()}", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Set axis limits
    ax.set_xlim(0.5, max_steps + 0.5)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved Pareto plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_step_histogram(results, dataset, save_path=None, max_steps=6):
    """
    Plot distribution of halting times for each model at a fixed threshold.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    model_styles = {
        "bpan": {"color": "#2E86AB", "label": "BPAN"},
        "act": {"color": "#A23B72", "label": "ACT"},
        "pondernet": {"color": "#F18F01", "label": "PonderNet"},
        "multi_exit": {"color": "#C73E1D", "label": "Multi-Exit"},
    }
    
    # Pick a middle threshold (0.9)
    target_threshold = 0.9
    
    for ax_idx, (model_name, data_list) in enumerate(results.items()):
        if ax_idx >= 4 or not data_list:
            continue
        
        ax = axes[ax_idx]
        style = model_styles.get(model_name, {"color": "gray", "label": model_name})
        
        # Find closest threshold
        closest = min(data_list, key=lambda d: abs(d["threshold"] - target_threshold))
        step_hist = closest.get("step_hist", {})
        
        if step_hist:
            steps = list(range(1, max_steps + 1))
            counts = [step_hist.get(str(t), step_hist.get(t, 0)) for t in steps]
            total = sum(counts)
            if total > 0:
                fracs = [c / total for c in counts]
            else:
                fracs = [0] * len(steps)
            
            ax.bar(steps, fracs, color=style["color"], alpha=0.8, edgecolor="black")
            ax.set_xlabel("Halting Step", fontsize=11)
            ax.set_ylabel("Fraction of Samples", fontsize=11)
            ax.set_title(f"{style['label']} — Threshold {closest['threshold']:.2f}\n"
                        f"Acc: {closest['acc']:.3f}, Avg Steps: {closest['avg_steps']:.2f}",
                        fontsize=11)
            ax.set_xticks(steps)
            ax.set_xlim(0.5, max_steps + 0.5)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
    
    # Hide unused axes
    for ax_idx in range(len(results), 4):
        axes[ax_idx].set_visible(False)
    
    plt.suptitle(f"Halting Time Distributions — {dataset.upper()}", fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved step histogram to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_speedup_vs_accuracy_loss(results, dataset, save_path=None, max_steps=6):
    """
    Plot relative speedup vs accuracy loss compared to full-depth baseline.
    This addresses the "20x latency" concern by showing compute savings.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    model_styles = {
        "bpan": {"color": "#2E86AB", "marker": "o", "label": "BPAN (ours)"},
        "act": {"color": "#A23B72", "marker": "s", "label": "ACT"},
        "pondernet": {"color": "#F18F01", "marker": "^", "label": "PonderNet"},
        "multi_exit": {"color": "#C73E1D", "marker": "D", "label": "Multi-Exit"},
    }
    
    for model_name, data_list in results.items():
        if not data_list:
            continue
        
        style = model_styles.get(model_name, {"color": "gray", "marker": "x", "label": model_name})
        
        # Baseline: highest accuracy (threshold ~1.0 or max steps)
        baseline = max(data_list, key=lambda d: d["acc"])
        baseline_acc = baseline["acc"]
        
        speedups = []
        acc_losses = []
        
        for d in data_list:
            speedup = max_steps / max(d["avg_steps"], 0.1)  # Relative to max steps
            acc_loss = baseline_acc - d["acc"]
            speedups.append(speedup)
            acc_losses.append(acc_loss)
        
        ax.scatter(speedups, acc_losses, 
                   color=style["color"],
                   marker=style["marker"],
                   s=80,
                   alpha=0.8,
                   label=style["label"])
        
        # Connect with line
        sorted_pts = sorted(zip(speedups, acc_losses))
        ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                color=style["color"], linewidth=1.5, alpha=0.5)
    
    ax.axhline(y=0.01, color='gray', linestyle='--', alpha=0.5, 
               label='1% accuracy loss')
    
    ax.set_xlabel("Speedup Factor (vs Max Steps)", fontsize=12)
    ax.set_ylabel("Accuracy Loss (vs Full Depth)", fontsize=12)
    ax.set_title(f"Compute Savings Trade-off — {dataset.upper()}", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, max_steps + 0.5)
    ax.set_ylim(-0.01, 0.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved speedup plot to {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_latex_table(results, dataset, save_path=None):
    """
    Generate LaTeX table summarizing anytime results.
    """
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Anytime accuracy vs compute on " + dataset.upper() + r"}")
    lines.append(r"\label{tab:anytime_" + dataset + r"}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Model & Threshold & Accuracy & Avg Steps & Speedup \\")
    lines.append(r"\midrule")
    
    max_steps = 6  # Assume default
    
    for model_name, data_list in results.items():
        if not data_list:
            continue
        
        # Report at key thresholds
        for target_thr in [0.7, 0.9, 0.99]:
            closest = min(data_list, key=lambda d: abs(d["threshold"] - target_thr))
            speedup = max_steps / max(closest["avg_steps"], 0.1)
            
            lines.append(f"{model_name.upper()} & {closest['threshold']:.2f} & "
                        f"{closest['acc']:.3f} & {closest['avg_steps']:.2f} & "
                        f"{speedup:.2f}$\\times$ \\\\")
        
        lines.append(r"\midrule")
    
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    
    latex_str = "\n".join(lines)
    
    if save_path:
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to {save_path}")
    else:
        print(latex_str)
    
    return latex_str


def main():
    parser = argparse.ArgumentParser(
        description="Generate Pareto plots for anytime classification models."
    )
    parser.add_argument("--results_dir", type=str, default="./results_anytime_all",
                        help="Directory containing JSONL result files.")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "emnist_letters", "cifar10"])
    parser.add_argument("--out_dir", type=str, default="./plots",
                        help="Directory to save plots.")
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Maximum steps for reference.")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = load_anytime_results(args.results_dir, args.dataset)
    
    if not results:
        print(f"No results found for {args.dataset} in {args.results_dir}")
        return
    
    print(f"Loaded results for models: {list(results.keys())}")
    
    # Generate plots
    plot_pareto_curves(
        results, args.dataset,
        save_path=out_dir / f"pareto_{args.dataset}.png",
        max_steps=args.max_steps
    )
    
    plot_step_histogram(
        results, args.dataset,
        save_path=out_dir / f"step_hist_{args.dataset}.png",
        max_steps=args.max_steps
    )
    
    plot_speedup_vs_accuracy_loss(
        results, args.dataset,
        save_path=out_dir / f"speedup_{args.dataset}.png",
        max_steps=args.max_steps
    )
    
    generate_latex_table(
        results, args.dataset,
        save_path=out_dir / f"table_{args.dataset}.tex"
    )


if __name__ == "__main__":
    main()

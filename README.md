# BPAN — Balanced Predictive Attractor Networks

Reference implementation accompanying the manuscript

> **Balanced Predictive Attractor Networks: A Compact Excitatory–Inhibitory Testbed for Anytime and Sequential Inference**
> IEEE Transactions on Cognitive and Developmental Systems, Manuscript ID **TCDS-2026-0088**.

BPAN is a small recurrent network with separate excitatory (E) and inhibitory (I)
populations, softplus sign constraints (a differentiable realisation of Dale's
law), a contractive update, and an auxiliary balance regulariser that encourages
near-balanced E/I currents. The code reproduces every quantitative result and
figure in the paper.

---

## Repository layout

### Core model
| File | Purpose |
|---|---|
| [models.py](models.py) | BPAN, E/I layer, softplus sign constraints, balance regulariser. |
| [models_anytime.py](models_anytime.py) | Anytime wrappers: BPAN-anytime, ACT, PonderNet, Multi-Exit. |

### Training / sweep drivers
| File | Purpose |
|---|---|
| [run_experiment.py](run_experiment.py) | Single training run (BPAN or MLP) on MNIST / F-MNIST / CIFAR-10 / SVHN. |
| [run_param_sweep.py](run_param_sweep.py) | Hidden-width and seed sweep for BPAN vs. MLP (Table 1). |
| [run_ei_ratio_sweep.py](run_ei_ratio_sweep.py) | E/I ratio sweep (legacy helper; superseded by the sensitivity driver). |
| [run_anytime_bpan.py](run_anytime_bpan.py) | Trains BPAN with per-step confidence readouts and threshold-based halting. |
| [run_anytime_baselines.py](run_anytime_baselines.py) | Trains ACT, PonderNet, and Multi-Exit baselines under matched budgets. |
| [run_sensitivity_analysis.py](run_sensitivity_analysis.py) | Sensitivity sweeps over `T`, `lambda_bal`, and E/I ratio (Fig. 11). |
| [analyze_balance_regularizer.py](analyze_balance_regularizer.py) | Ablation of the balance regulariser and spectral / E–I diagnostics. |
| [run_sequential_task.py](run_sequential_task.py) | CT-MNIST, streaming, and sequential-CIFAR-patches tasks for BPAN, GRU, LSTM, Transformer. |

### Analysis / plotting
| File | Purpose |
|---|---|
| [aggregate_results.py](aggregate_results.py) | Collate JSONL outputs into per-dataset tables. |
| [compute_costs.py](compute_costs.py) | Parameter / FLOPs / latency accounting. |
| [plot_pareto.py](plot_pareto.py) | Accuracy-vs-cost Pareto curves (Fig. 10). |
| [analyze_dynamics.py](analyze_dynamics.py) | Step-wise accuracy, halting histograms, trajectory diagnostics. |
| [analyze_pca_tsne.py](analyze_pca_tsne.py) | PCA / t-SNE of E/I state trajectories. |
| [make_interpretability_figures.py](make_interpretability_figures.py) | Attractor / occlusion figures. |
| [aggregate_paper_figures.py](aggregate_paper_figures.py) | Builds the final PNGs used in the manuscript. |

### Orchestration
- [run_all_experiments.sh](run_all_experiments.sh) — master shell driver that runs the full pipeline end to end.

---

## Installation

Tested with Python 3.10 and CUDA 11.8.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The pinned dependencies are minimal:

```
torch>=2.0
torchvision>=0.15
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
```

Datasets (MNIST, Fashion-MNIST, CIFAR-10, SVHN) are pulled automatically through
`torchvision.datasets` into `./data/` on first use.

---

## Quick start

Reproduce every experiment in the paper with a single command:

```bash
chmod +x run_all_experiments.sh
./run_all_experiments.sh --epochs 15
```

For a faster smoke test:

```bash
./run_all_experiments.sh --quick     # 5 epochs
```

The script writes JSONL results and figures into

```
results/                  # parameter sweeps (Table 1)
results_anytime_all/      # anytime comparison (Fig. 10)
results_balance_analysis/ # balance regulariser ablation
results_sensitivity/      # sensitivity sweeps (Fig. 11)
results_sequential/       # sequential tasks (Table 2)
plots/                    # Pareto curves
```

Individual stages can also be run on their own — each driver is a standalone
`argparse` program; pass `-h` to any of them for the full flag list. Example:

```bash
python run_anytime_bpan.py --dataset mnist --epochs 15 --hidden 256 \
    --bpan_T 6 --thresholds "0.5,0.7,0.9,0.95,0.99" \
    --out_dir ./results_anytime_all
```

After the runs finish, generate the manuscript-ready figures with

```bash
python aggregate_paper_figures.py
```

---

## Reproducibility notes

- All training drivers accept `--seed`; the paper reports means ± std over
  seeds 1, 2, 3.
- `run_all_experiments.sh` activates a virtualenv at `~/venvs/venv`; edit
  the `source` line at the top of the script to point at your environment.
- Results in the paper were obtained on Apple M5 Silicon (32 GB VRAM);
  MNIST and Fashion-MNIST runs also complete comfortably on CPU.

---

## Citation

If you use this code, please cite the TCDS paper (BibTeX will be added once
the camera-ready DOI is assigned).

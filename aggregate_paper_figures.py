"""
aggregate_paper_figures.py

One-shot script: reads every result JSON/JSONL produced by the pipeline and
writes paper-ready figures + LaTeX tables into ./plots_paper/.

Produces:
  - sensitivity_mnist.png, sensitivity_fashion_mnist.png (3-panel: T, lambda_bal, E/I)
  - sequential_tasks.png (bar chart: BPAN/GRU/LSTM/Transformer on 3 tasks)
  - param_sweep.png (accuracy-vs-params curves, BPAN vs MLP, 4 datasets)
  - table_sensitivity.tex
  - table_sequential.tex
  - table_param_sweep.tex
  - table_anytime_all.tex  (consolidated: MNIST/F-MNIST/SVHN x 4 models)
  - summary.json (all headline numbers for the paper)
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "plots_paper"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.dpi": 140,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------- helpers
def _mean_std(xs):
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    m = mean(xs)
    s = stdev(xs) if len(xs) > 1 else 0.0
    return m, s


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- 1. Sensitivity
def sensitivity_figure(dataset):
    T_rows = _load_jsonl(ROOT / f"results_sensitivity/{dataset}_T_sweep.jsonl")
    L_rows = _load_jsonl(ROOT / f"results_sensitivity/{dataset}_lambda_bal_sweep.jsonl")
    E_rows = _load_jsonl(ROOT / f"results_sensitivity/{dataset}_ei_ratio_sweep.jsonl")

    def agg(rows, key):
        by_k = defaultdict(list)
        for r in rows:
            by_k[r[key]].append(r["test_acc"])
        keys = sorted(by_k)
        means = [mean(by_k[k]) for k in keys]
        stds = [stdev(by_k[k]) if len(by_k[k]) > 1 else 0.0 for k in keys]
        return keys, means, stds

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))

    keys, m, s = agg(T_rows, "T")
    axes[0].errorbar(keys, m, yerr=s, marker="o", capsize=4, color="#1f77b4")
    axes[0].set_xlabel("Recurrent steps T")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("(a) Depth of pondering")
    axes[0].grid(True, alpha=0.3)

    keys, m, s = agg(L_rows, "lambda_bal")
    axes[1].errorbar(range(len(keys)), m, yerr=s, marker="s", capsize=4, color="#d62728")
    axes[1].set_xticks(range(len(keys)))
    axes[1].set_xticklabels([f"{k:.0e}" if k > 0 else "0" for k in keys])
    axes[1].set_xlabel(r"$\lambda_{\mathrm{bal}}$")
    axes[1].set_title("(b) Balance regulariser weight")
    axes[1].grid(True, alpha=0.3)

    # E/I ratio: use the reported ratio directly. We plot against integer
    # positions rather than a log axis so the tick labels "2:1", "4:1", "8:1",
    # "16:1" don't collide in a narrow sub-panel.
    by_r = defaultdict(list)
    for r in E_rows:
        by_r[round(r["ei_ratio"], 1)].append(r["test_acc"])
    keys = sorted(by_r)
    m = [mean(by_r[k]) for k in keys]
    s = [stdev(by_r[k]) if len(by_r[k]) > 1 else 0.0 for k in keys]
    xpos = list(range(len(keys)))
    axes[2].errorbar(xpos, m, yerr=s, marker="^", capsize=4, color="#2ca02c")
    axes[2].set_xticks(xpos)
    axes[2].set_xticklabels([f"{int(round(k))}:1" for k in keys])
    axes[2].set_xlabel(r"E/I ratio ($n_{\mathrm{exc}}\!:\!n_{\mathrm{inh}}$)")
    axes[2].set_title("(c) E/I population ratio")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"BPAN sensitivity on {dataset.replace('_',' ').title()} (3 seeds)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / f"sensitivity_{dataset}.png")
    plt.close(fig)

    # Return numerical summary for LaTeX
    def _fmt(rows, key):
        by_k = defaultdict(list)
        for r in rows:
            by_k[r[key]].append(r["test_acc"])
        return {str(k): _mean_std(by_k[k]) for k in sorted(by_k)}

    return {
        "T": _fmt(T_rows, "T"),
        "lambda_bal": _fmt(L_rows, "lambda_bal"),
        "ei_ratio": {str(round(k, 1)): _mean_std(by_r[k]) for k in sorted(by_r)},
    }


# ---------------------------------------------------------------- 2. Sequential tasks
def sequential_figure():
    tasks = ["ctmnist", "streaming", "cifar_patches"]
    models = ["bpan", "gru", "lstm", "transformer"]
    task_labels = ["CT-MNIST", "Streaming", "Sequential CIFAR"]
    model_colors = {"bpan": "#1f77b4", "gru": "#ff7f0e",
                    "lstm": "#2ca02c", "transformer": "#d62728"}

    accs = {m: [] for m in models}
    params = {m: [] for m in models}
    for t in tasks:
        for m in models:
            path = ROOT / f"results_sequential/{t}_{m}.json"
            d = json.load(open(path))
            acc = d.get("best_test_acc", d.get("best_acc", 0))
            accs[m].append(acc)
            params[m].append(d.get("params", 0))

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(tasks))
    w = 0.2
    for i, m in enumerate(models):
        offset = (i - 1.5) * w
        ax.bar(x + offset, accs[m], w, label=m.upper(), color=model_colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Sequential evidence integration: BPAN vs. recurrent / attention baselines")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "sequential_tasks.png")
    plt.close(fig)

    return {"tasks": tasks, "models": models, "acc": accs, "params": params}


# ---------------------------------------------------------------- 3. Param sweep
def param_sweep_figure():
    datasets = ["mnist", "fashion_mnist", "cifar10", "svhn"]
    titles = ["MNIST", "Fashion-MNIST", "CIFAR-10", "SVHN"]

    # aggregate across seeds
    agg = {}  # dataset -> model -> hidden -> list[acc]
    agg_params = {}  # dataset -> model -> hidden -> params
    for ds in datasets:
        agg[ds] = {"bpan": defaultdict(list), "mlp": defaultdict(list)}
        agg_params[ds] = {"bpan": {}, "mlp": {}}
        for seed in (1, 2, 3):
            p = ROOT / f"results/{ds}_param_sweep_seed{seed}.jsonl"
            if not p.exists():
                continue
            for row in _load_jsonl(p):
                m = row["model"]
                h = row["hidden"]
                agg[ds][m][h].append(row["best_test_acc"])
                agg_params[ds][m][h] = row["params"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)
    for ax, ds, title in zip(axes, datasets, titles):
        for model, color, marker in [("bpan", "#1f77b4", "o"), ("mlp", "#ff7f0e", "s")]:
            hs = sorted(agg[ds][model])
            if not hs:
                continue
            means = [mean(agg[ds][model][h]) for h in hs]
            stds = [stdev(agg[ds][model][h]) if len(agg[ds][model][h]) > 1 else 0
                    for h in hs]
            ps = [agg_params[ds][model][h] for h in hs]
            ax.errorbar(ps, means, yerr=stds, marker=marker, capsize=3,
                        label=model.upper(), color=color)
        ax.set_title(title)
        ax.set_xlabel("Parameters")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Best test accuracy")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("BPAN vs. MLP: accuracy–parameter trade-off (mean ± std, 3 seeds)",
                 y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "param_sweep.png")
    plt.close(fig)

    # Numerical summary
    summary = {}
    for ds in datasets:
        summary[ds] = {}
        for m in ("bpan", "mlp"):
            summary[ds][m] = {}
            for h, accs in agg[ds][m].items():
                summary[ds][m][h] = {
                    "mean": mean(accs),
                    "std": stdev(accs) if len(accs) > 1 else 0.0,
                    "n_seeds": len(accs),
                    "params": agg_params[ds][m][h],
                }
    return summary


# ---------------------------------------------------------------- 4. Anytime all
def anytime_summary():
    out = {}
    for ds in ("mnist", "fashion_mnist", "svhn"):
        out[ds] = {}
        for model in ("bpan", "act", "pondernet", "multi_exit"):
            p = ROOT / f"results_anytime_all/{ds}_{model}_anytime.jsonl"
            if not p.exists():
                continue
            rows = _load_jsonl(p)
            summary = next(r for r in rows if r["type"] == "summary")
            acc = summary.get("best_test_acc_fixed_T") or summary.get("best_test_acc_fixed")
            out[ds][model] = {"params": summary["params"], "best_acc": acc}
            # Anytime curve at theta=0.9
            anytime_rows = [r for r in rows if r["type"] == "anytime"]
            r09 = next((r for r in anytime_rows
                        if abs(r["threshold"] - 0.9) < 1e-6), None)
            if r09:
                out[ds][model]["acc_at_0.9"] = r09["acc"]
                out[ds][model]["avg_steps_at_0.9"] = r09["avg_steps"]
    return out


# ---------------------------------------------------------------- 5. LaTeX tables
def latex_tables(sens_mnist, sens_fmnist, seq, param, anytime):
    # --- Sensitivity ---
    def _sens_rows(d):
        rows = []
        for k, (m, s) in d["T"].items():
            rows.append(("$T={}$".format(k), f"{m*100:.2f}", f"{s*100:.2f}"))
        rows.append(("", "", ""))
        for k, (m, s) in d["lambda_bal"].items():
            lbl = "$\\lambda_{\\mathrm{bal}}=0$" if float(k) == 0 else \
                  f"$\\lambda_{{\\mathrm{{bal}}}}={float(k):.0e}$"
            rows.append((lbl, f"{m*100:.2f}", f"{s*100:.2f}"))
        rows.append(("", "", ""))
        for k, (m, s) in d["ei_ratio"].items():
            rows.append((f"E/I$={k}$:1", f"{m*100:.2f}", f"{s*100:.2f}"))
        return rows

    tex = ["\\begin{tabular}{lcc|cc}",
           "\\toprule",
           "Setting & \\multicolumn{2}{c|}{MNIST} & \\multicolumn{2}{c}{F-MNIST} \\\\",
           " & Mean (\\%) & Std (\\%) & Mean (\\%) & Std (\\%) \\\\",
           "\\midrule"]
    rows_m = _sens_rows(sens_mnist)
    rows_f = _sens_rows(sens_fmnist)
    for (label, mm, sm), (_, mf, sf) in zip(rows_m, rows_f):
        if label == "":
            tex.append("\\midrule")
            continue
        tex.append(f"{label} & {mm} & {sm} & {mf} & {sf} \\\\")
    tex.extend(["\\bottomrule", "\\end{tabular}"])
    (OUT / "table_sensitivity.tex").write_text("\n".join(tex))

    # --- Sequential ---
    tex = ["\\begin{tabular}{llrr}",
           "\\toprule",
           "Task & Model & Params & Test Acc.\\ (\\%) \\\\",
           "\\midrule"]
    for task, label in zip(seq["tasks"], ["CT-MNIST", "Streaming", "Seq.\\ CIFAR"]):
        for i, m in enumerate(seq["models"]):
            tex.append(f"{label if i==0 else ''} & {m.upper()} "
                       f"& {seq['params'][m][seq['tasks'].index(task)]:,} "
                       f"& {seq['acc'][m][seq['tasks'].index(task)]*100:.2f} \\\\")
        tex.append("\\midrule")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_sequential.tex").write_text("\n".join(tex))

    # --- Param sweep ---
    tex = ["\\begin{tabular}{llrrrr}",
           "\\toprule",
           "Dataset & Model & \\multicolumn{4}{c}{Hidden width} \\\\",
           " & & 64 & 128 & 256 & 512 \\\\",
           "\\midrule"]
    for ds, label in [("mnist", "MNIST"), ("fashion_mnist", "F-MNIST"),
                      ("cifar10", "CIFAR-10"), ("svhn", "SVHN")]:
        for i, m in enumerate(("mlp", "bpan")):
            cells = []
            for h in (64, 128, 256, 512):
                if h in param[ds][m]:
                    mm, ss = param[ds][m][h]["mean"], param[ds][m][h]["std"]
                    cells.append(f"{mm*100:.2f}$\\pm${ss*100:.2f}")
                else:
                    cells.append("--")
            tex.append(f"{label if i==0 else ''} & {m.upper()} & " +
                       " & ".join(cells) + " \\\\")
        tex.append("\\midrule")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_param_sweep.tex").write_text("\n".join(tex))

    # --- Anytime consolidated ---
    tex = ["\\begin{tabular}{llrrr}",
           "\\toprule",
           "Dataset & Model & Params & Best Acc.\\ (\\%) & Acc.\\ @ $\\theta{=}0.9$ (steps) \\\\",
           "\\midrule"]
    for ds, label in [("mnist", "MNIST"), ("fashion_mnist", "F-MNIST"),
                      ("svhn", "SVHN")]:
        for i, m in enumerate(("bpan", "act", "pondernet", "multi_exit")):
            if m not in anytime[ds]:
                continue
            d = anytime[ds][m]
            steps = d.get("avg_steps_at_0.9", None)
            steps_str = f"{d['acc_at_0.9']*100:.2f}\\ ({steps:.2f})" if steps else "--"
            tex.append(f"{label if i==0 else ''} & {m.replace('_',' ').upper()} "
                       f"& {d['params']:,} & {d['best_acc']*100:.2f} & {steps_str} \\\\")
        tex.append("\\midrule")
    tex = tex[:-1] + ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table_anytime_all.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------- main
def main():
    print("Generating sensitivity figures...")
    sens_mnist = sensitivity_figure("mnist")
    sens_fmnist = sensitivity_figure("fashion_mnist")

    print("Generating sequential-task figure...")
    seq = sequential_figure()

    print("Generating parameter-sweep figure...")
    param = param_sweep_figure()

    print("Aggregating anytime results...")
    anytime = anytime_summary()

    print("Writing LaTeX tables...")
    latex_tables(sens_mnist, sens_fmnist, seq, param, anytime)

    summary = {
        "sensitivity_mnist": sens_mnist,
        "sensitivity_fashion_mnist": sens_fmnist,
        "sequential": seq,
        "param_sweep": param,
        "anytime": anytime,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nAll assets written to {OUT}/")
    print("  Figures: sensitivity_*.png, sequential_tasks.png, param_sweep.png")
    print("  Tables:  table_sensitivity.tex, table_sequential.tex, "
          "table_param_sweep.tex, table_anytime_all.tex")
    print("  JSON:    summary.json")


if __name__ == "__main__":
    main()

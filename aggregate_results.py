# aggregate_results.py

import argparse
import json
from collections import defaultdict
from pathlib import Path
import math


def mean_std(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def load_records(files):
    recs = []
    for fname in files:
        path = Path(fname)
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                recs.append(json.loads(line))
    return recs


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate JSONL result files into LaTeX table rows."
    )
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--task", type=str, required=True,
                        choices=["small_data", "param_sweep", "ei_ratio"])
    args = parser.parse_args()

    recs = load_records(args.files)
    # Filter to records with best_test_acc
    recs = [r for r in recs if "best_test_acc" in r]

    if args.dataset is not None:
        recs = [r for r in recs if r.get("dataset") == args.dataset]

    if args.task == "small_data":
        # group by (model, frac_train)
        groups = defaultdict(list)
        for r in recs:
            key = (r["model"], float(r["frac_train"]))
            groups[key].append(r["best_test_acc"])

        fracs = sorted({k[1] for k in groups.keys()})
        print("% Train fraction & MLP acc. & BPAN acc. \\\\")
        print("\\hline")
        for frac in fracs:
            mlp_accs = groups.get(("mlp", frac), [])
            bpan_accs = groups.get(("bpan", frac), [])
            mlp_mean, mlp_std = mean_std(mlp_accs)
            bpan_mean, bpan_std = mean_std(bpan_accs)
            print(
                f"${frac:.4f}$ & "
                f"${mlp_mean:.4f} \\pm {mlp_std:.4f}$ & "
                f"${bpan_mean:.4f} \\pm {bpan_std:.4f}$ \\\\"
            )

    elif args.task == "param_sweep":
        # group by (model, hidden)
        groups = defaultdict(list)
        for r in recs:
            key = (r["model"], int(r["hidden"]))
            groups[key].append(r["best_test_acc"])

        widths = sorted({k[1] for k in groups.keys()})
        print("% Hidden & MLP acc. & BPAN acc. \\\\")
        print("\\hline")
        for h in widths:
            mlp_accs = groups.get(("mlp", h), [])
            bpan_accs = groups.get(("bpan", h), [])
            mlp_mean, mlp_std = mean_std(mlp_accs)
            bpan_mean, bpan_std = mean_std(bpan_accs)
            print(
                f"{h} & "
                f"${mlp_mean:.4f} \\pm {mlp_std:.4f}$ & "
                f"${bpan_mean:.4f} \\pm {bpan_std:.4f}$ \\\\"
            )

    elif args.task == "ei_ratio":
        # group by ratio (n_exc / n_inh)
        groups = defaultdict(list)
        for r in recs:
            if r.get("model") != "bpan":
                continue
            n_exc = r.get("n_exc")
            n_inh = r.get("n_inh")
            if not (isinstance(n_exc, int) and isinstance(n_inh, int) and n_inh > 0):
                continue
            ratio = n_exc / n_inh
            groups[ratio].append(r["best_test_acc"])

        ratios = sorted(groups.keys())
        print("% E/I ratio & BPAN acc. \\\\")
        print("\\hline")
        for ratio in ratios:
            accs = groups[ratio]
            m, sd = mean_std(accs)
            print(f"${ratio:.2f}$ & ${m:.4f} \\pm {sd:.4f}$ \\\\")


if __name__ == "__main__":
    main()

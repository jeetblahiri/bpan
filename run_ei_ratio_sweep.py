# run_ei_ratio_sweep.py

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run E/I ratio sweeps for BPAN."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10", "svhn"])
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ei_ratios", nargs="+", type=float,
                        default=[2.0, 4.0, 8.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--bpan_lambda_act", type=float, default=1e-4)
    parser.add_argument("--bpan_lambda_bal", type=float, default=5e-4)
    parser.add_argument("--out_path", type=str, default="results/ei_ratio.jsonl")
    args = parser.parse_args()

    for seed in args.seeds:
        for ratio in args.ei_ratios:
            cmd = [
                "python", "run_experiment.py",
                "--dataset", args.dataset,
                "--model", "bpan",
                "--hidden", str(args.hidden),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--bpan_T", str(args.bpan_T),
                "--bpan_lambda_act", str(args.bpan_lambda_act),
                "--bpan_lambda_bal", str(args.bpan_lambda_bal),
                "--ei_ratio", str(ratio),
                "--seed", str(seed),
                "--out_path", args.out_path,
            ]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

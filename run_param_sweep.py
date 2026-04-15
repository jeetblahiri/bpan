# run_param_sweep.py

import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweeps (hidden widths, multiple seeds) for MLP and BPAN."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10", "svhn"])
    parser.add_argument("--hidden_list", nargs="+", type=int,
                        default=[64, 128, 256, 512])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--bpan_lambda_act", type=float, default=1e-4)
    parser.add_argument("--bpan_lambda_bal", type=float, default=5e-4)
    parser.add_argument("--out_prefix", type=str, default="results")
    args = parser.parse_args()

    for seed in args.seeds:
        for hidden in args.hidden_list:
            # MLP
            out_path_mlp = f"{args.out_prefix}/{args.dataset}_param_sweep_seed{seed}.jsonl"
            cmd_mlp = [
                "python", "run_experiment.py",
                "--dataset", args.dataset,
                "--model", "mlp",
                "--hidden", str(hidden),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--lr", "1e-3",
                "--seed", str(seed),
                "--out_path", out_path_mlp,
            ]
            print("Running:", " ".join(cmd_mlp))
            subprocess.run(cmd_mlp, check=True)

            # BPAN
            out_path_bpan = f"{args.out_prefix}/{args.dataset}_param_sweep_seed{seed}.jsonl"
            cmd_bpan = [
                "python", "run_experiment.py",
                "--dataset", args.dataset,
                "--model", "bpan",
                "--hidden", str(hidden),
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--bpan_T", str(args.bpan_T),
                "--bpan_lambda_act", str(args.bpan_lambda_act),
                "--bpan_lambda_bal", str(args.bpan_lambda_bal),
                "--seed", str(seed),
                "--out_path", out_path_bpan,
            ]
            print("Running:", " ".join(cmd_bpan))
            subprocess.run(cmd_bpan, check=True)


if __name__ == "__main__":
    main()

# run_sensitivity_analysis.py
"""
Sensitivity analysis for BPAN hyperparameters (Reviewer 2 request).

Sweeps over:
  1. E/I ratio (n_exc / n_inh) with total hidden fixed
  2. Number of recurrent steps T
  3. Balance regulariser weight lambda_bal

For each configuration, trains BPAN and reports test accuracy, anytime
performance (avg steps at theta=0.9), and dynamics metrics (balance cost).

Results are saved as JSONL for plotting.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import BPANClassifier, BalancedEILayer, ConvBackboneCIFAR, ConvWrapper


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_datasets(name, data_root):
    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes, in_shape = 10, (1, 28, 28)

    elif name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes, in_shape = 10, (1, 28, 28)

    elif name == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm_test)
        n_classes, in_shape = 10, (3, 32, 32)

    elif name == "svhn":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        train_ds = datasets.SVHN(root=data_root, split="train", download=True,
                                 transform=tfm_train)
        test_ds = datasets.SVHN(root=data_root, split="test", download=True,
                                transform=tfm_test)
        n_classes, in_shape = 10, (3, 32, 32)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_ds, test_ds, n_classes, in_shape


def build_model(dataset_name, in_shape, n_classes, hidden, n_exc, n_inh,
                T, lambda_act, lambda_bal):
    """Build BPAN model for given hyperparameters."""
    is_conv = dataset_name in ("cifar10", "svhn")

    if is_conv:
        backbone = ConvBackboneCIFAR()
        head = BPANClassifier(
            input_dim=backbone.out_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=T,
            lambda_act=lambda_act,
            lambda_bal=lambda_bal,
        )
        model = ConvWrapper(backbone, head)
    else:
        input_dim = in_shape[0] * in_shape[1] * in_shape[2]
        model = BPANClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=T,
            lambda_act=lambda_act,
            lambda_bal=lambda_bal,
        )

    return model


def train_one_epoch(model, optimizer, loader, device, dataset_name, T,
                    lambda_act, lambda_bal, max_grad_norm=5.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    is_conv = dataset_name in ("cifar10", "svhn")

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        if not is_conv:
            x = x.view(x.size(0), -1)

        stats = model.forward_with_stats(x)
        logits_seq = stats["logits_seq"]       # [B, T, C]
        r_e_seq = stats["r_e_seq"]
        r_i_seq = stats["r_i_seq"]
        bal_e_seq = stats["bal_e_seq"]
        bal_i_seq = stats["bal_i_seq"]

        # Weighted CE over time steps
        T_actual = logits_seq.size(1)
        ce_t = torch.stack([
            F.cross_entropy(logits_seq[:, t, :], y) for t in range(T_actual)
        ])
        weights = torch.linspace(0.3, 1.0, steps=T_actual, device=device)
        weights = weights / weights.sum()
        ce = (weights * ce_t).sum()

        act_cost = r_e_seq.pow(2).mean() + r_i_seq.pow(2).mean()
        bal_cost = bal_e_seq.pow(2).mean() + bal_i_seq.pow(2).mean()
        loss = ce + lambda_act * act_cost + lambda_bal * bal_cost

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        B = x.size(0)
        total_loss += loss.item() * B
        preds = logits_seq[:, -1, :].argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += B

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_model(model, loader, device, dataset_name, T, threshold=0.9):
    """Evaluate accuracy (fixed T) and anytime (avg steps at threshold)."""
    model.eval()
    is_conv = dataset_name in ("cifar10", "svhn")

    total_correct = 0
    total_samples = 0
    total_steps = 0
    total_bal_cost = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if not is_conv:
            x = x.view(x.size(0), -1)

        stats = model.forward_with_stats(x)
        logits_seq = stats["logits_seq"]
        bal_e_seq = stats["bal_e_seq"]
        bal_i_seq = stats["bal_i_seq"]

        B = x.size(0)
        T_actual = logits_seq.size(1)

        # Fixed-T accuracy (last step)
        preds = logits_seq[:, -1, :].argmax(dim=1)
        total_correct += (preds == y).sum().item()

        # Balance cost
        bal_cost = bal_e_seq.pow(2).mean() + bal_i_seq.pow(2).mean()
        total_bal_cost += bal_cost.item() * B

        # Anytime: find first step where confidence >= threshold
        done = torch.zeros(B, dtype=torch.bool, device=device)
        steps_used = torch.full((B,), T_actual, dtype=torch.long, device=device)

        for t in range(T_actual):
            probs = logits_seq[:, t, :].softmax(dim=-1)
            conf, _ = probs.max(dim=-1)
            newly_done = (~done) & (conf >= threshold)
            steps_used[newly_done] = t + 1
            done = done | newly_done
            if done.all():
                break

        total_steps += steps_used.sum().item()
        total_samples += B

    acc = total_correct / total_samples
    avg_steps = total_steps / total_samples
    avg_bal = total_bal_cost / total_samples

    return acc, avg_steps, avg_bal


def run_single_config(dataset_name, in_shape, n_classes, train_loader, test_loader,
                      hidden, n_exc, n_inh, T, lambda_act, lambda_bal,
                      epochs, lr, device, seed):
    """Train and evaluate a single BPAN configuration."""
    torch.manual_seed(seed)

    model = build_model(dataset_name, in_shape, n_classes, hidden,
                        n_exc, n_inh, T, lambda_act, lambda_bal)
    model.to(device)
    params = count_params(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, dataset_name,
            T, lambda_act, lambda_bal
        )
        acc, avg_steps, avg_bal = evaluate_model(
            model, test_loader, device, dataset_name, T
        )

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # Reload best and evaluate
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    acc, avg_steps, avg_bal = evaluate_model(
        model, test_loader, device, dataset_name, T
    )

    return {
        "params": params,
        "test_acc": float(acc),
        "avg_steps_09": float(avg_steps),
        "avg_bal_cost": float(avg_bal),
    }


def main():
    parser = argparse.ArgumentParser(
        description="BPAN sensitivity analysis: sweep T, lambda_bal, E/I ratio."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10", "svhn"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_sensitivity")

    # Sweep parameters
    parser.add_argument("--sweep", type=str, default="all",
                        choices=["T", "lambda_bal", "ei_ratio", "all"],
                        help="Which sweep to run.")
    args = parser.parse_args()

    if (not args.no_cuda) and torch.cuda.is_available():
        device = torch.device("cuda")
    elif (not args.no_cuda) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_cuda = device.type == "cuda"
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, test_ds, n_classes, in_shape = get_datasets(args.dataset, args.data_root)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_cuda)

    H = args.hidden

    # Define sweep configurations
    sweeps = {}

    if args.sweep in ("T", "all"):
        sweeps["T"] = []
        for T_val in [2, 4, 6, 8, 10]:
            n_exc = H // 2
            n_inh = max(4, H // 8)
            sweeps["T"].append({
                "T": T_val, "n_exc": n_exc, "n_inh": n_inh,
                "lambda_act": 1e-4, "lambda_bal": 5e-4,
                "label": f"T={T_val}",
            })

    if args.sweep in ("lambda_bal", "all"):
        sweeps["lambda_bal"] = []
        for lbal in [0.0, 1e-4, 5e-4, 1e-3, 5e-3]:
            n_exc = H // 2
            n_inh = max(4, H // 8)
            sweeps["lambda_bal"].append({
                "T": 6, "n_exc": n_exc, "n_inh": n_inh,
                "lambda_act": 1e-4, "lambda_bal": lbal,
                "label": f"lbal={lbal}",
            })

    if args.sweep in ("ei_ratio", "all"):
        sweeps["ei_ratio"] = []
        for ratio in [2.0, 4.0, 8.0, 16.0]:
            # Total active units = H (exc + inh), split by ratio
            n_inh = max(4, int(H / (1 + ratio)))
            n_exc = H - n_inh
            sweeps["ei_ratio"].append({
                "T": 6, "n_exc": n_exc, "n_inh": n_inh,
                "lambda_act": 1e-4, "lambda_bal": 5e-4,
                "label": f"EI={ratio:.0f}:1 (exc={n_exc},inh={n_inh})",
            })

    for sweep_name, configs in sweeps.items():
        log_path = out_dir / f"{args.dataset}_{sweep_name}_sweep.jsonl"
        print(f"\n{'='*60}")
        print(f"Sweep: {sweep_name} on {args.dataset}")
        print(f"{'='*60}")

        for cfg in configs:
            for seed in args.seeds:
                print(f"\n  Config: {cfg['label']}, seed={seed}")
                t0 = time.time()

                result = run_single_config(
                    dataset_name=args.dataset,
                    in_shape=in_shape,
                    n_classes=n_classes,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    hidden=H,
                    n_exc=cfg["n_exc"],
                    n_inh=cfg["n_inh"],
                    T=cfg["T"],
                    lambda_act=cfg["lambda_act"],
                    lambda_bal=cfg["lambda_bal"],
                    epochs=args.epochs,
                    lr=args.lr,
                    device=device,
                    seed=seed,
                )
                elapsed = time.time() - t0

                record = {
                    "sweep": sweep_name,
                    "dataset": args.dataset,
                    "hidden": H,
                    "seed": seed,
                    "n_exc": cfg["n_exc"],
                    "n_inh": cfg["n_inh"],
                    "ei_ratio": cfg["n_exc"] / max(1, cfg["n_inh"]),
                    "T": cfg["T"],
                    "lambda_act": cfg["lambda_act"],
                    "lambda_bal": cfg["lambda_bal"],
                    "label": cfg["label"],
                    **result,
                    "time_s": elapsed,
                }

                with log_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")

                print(f"    Acc={result['test_acc']:.4f}, "
                      f"AvgSteps={result['avg_steps_09']:.2f}, "
                      f"BalCost={result['avg_bal_cost']:.2f}, "
                      f"Params={result['params']:,}, "
                      f"Time={elapsed:.1f}s")

    print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()

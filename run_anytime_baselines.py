# run_anytime_baselines.py
"""
Train and evaluate anytime baselines: ACT, PonderNet, Multi-Exit MLP.
Produces accuracy vs compute curves comparable to BPAN.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models_anytime import (
    ACTClassifier,
    PonderNetClassifier,
    MultiExitMLP,
    MultiExitCNN,
    evaluate_anytime_model,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------- Dataset helpers ---------------------- #

def get_datasets(name, data_root):
    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif name == "emnist_letters":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        train_ds = datasets.EMNIST(root=data_root, split="letters",
                                   train=True, download=True, transform=tfm)
        test_ds = datasets.EMNIST(root=data_root, split="letters",
                                  train=False, download=True, transform=tfm)
        n_classes = 26
        in_shape = (1, 28, 28)

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
        n_classes = 10
        in_shape = (3, 32, 32)

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_ds, test_ds, n_classes, in_shape


def get_dataloaders(dataset_name, data_root, batch_size, frac_train,
                    num_workers, pin_memory):
    train_ds, test_ds, n_classes, in_shape = get_datasets(dataset_name, data_root)

    if frac_train < 1.0:
        n = len(train_ds)
        k = int(n * frac_train)
        idx = torch.randperm(n)[:k].tolist()
        train_ds = Subset(train_ds, idx)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    return train_loader, test_loader, n_classes, in_shape


# ---------------------- Training loops ---------------------- #

def train_one_epoch(model, optimizer, train_loader, device, dataset_name,
                    max_grad_norm=None, is_cifar=False):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        if dataset_name == "emnist_letters":
            y = y - 1

        if not is_cifar:
            x = x.view(x.size(0), -1)

        out = model(x, targets=y)
        loss = out["loss"]
        ce_loss = out.get("ce_loss", loss)

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        B = x.size(0)
        total_loss += loss.item() * B
        total_ce += ce_loss.item() * B

        preds = out["logits"].argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += B

    return total_loss / total_samples, total_ce / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_fixed(model, loader, device, dataset_name, is_cifar=False):
    """Evaluate using final output only."""
    model.eval()
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if dataset_name == "emnist_letters":
            y = y - 1

        if not is_cifar:
            x = x.view(x.size(0), -1)

        out = model(x, targets=None)
        preds = out["logits"].argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_correct / total_samples


# ---------------------- Main ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train anytime baselines (ACT, PonderNet, Multi-Exit) and evaluate."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "emnist_letters", "cifar10"])
    parser.add_argument("--model", type=str, default="act",
                        choices=["act", "pondernet", "multi_exit"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=6,
                        help="Max pondering/exit steps (comparable to BPAN T).")
    parser.add_argument("--frac_train", type=float, default=1.0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_anytime_baselines")
    parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95,0.99",
                        help="Comma-separated confidence thresholds for anytime eval.")
    # Model-specific hyperparameters
    parser.add_argument("--act_time_penalty", type=float, default=0.01,
                        help="ACT ponder cost penalty (tau).")
    parser.add_argument("--ponder_lambda_p", type=float, default=0.3,
                        help="PonderNet geometric prior parameter.")
    parser.add_argument("--ponder_beta", type=float, default=0.01,
                        help="PonderNet KL penalty weight.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.dataset}_{args.model}_anytime.jsonl"

    train_loader, test_loader, n_classes, in_shape = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.frac_train,
        num_workers=args.num_workers, pin_memory=use_cuda
    )

    is_cifar = (args.dataset == "cifar10")
    input_dim = in_shape[0] * in_shape[1] * in_shape[2]

    # Build model
    if args.model == "act":
        model = ACTClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=args.hidden,
            max_steps=args.max_steps,
            halt_epsilon=0.01,
            time_penalty=args.act_time_penalty,
        )
    elif args.model == "pondernet":
        model = PonderNetClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden_dim=args.hidden,
            max_steps=args.max_steps,
            lambda_p=args.ponder_lambda_p,
            beta=args.ponder_beta,
        )
    elif args.model == "multi_exit":
        if is_cifar:
            model = MultiExitCNN(n_classes=n_classes, n_exits=args.max_steps)
        else:
            model = MultiExitMLP(
                input_dim=input_dim,
                n_classes=n_classes,
                hidden_dim=args.hidden,
                n_exits=args.max_steps,
            )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model.to(device)
    params = count_params(model)
    print(f"\n{args.model.upper()} on {args.dataset}")
    print(f"Params: {params:,}, max_steps={args.max_steps}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_ce, train_acc = train_one_epoch(
            model, optimizer, train_loader, device, args.dataset,
            max_grad_norm=args.max_grad_norm, is_cifar=is_cifar
        )
        epoch_time = time.time() - t0

        test_acc = evaluate_fixed(model, test_loader, device, args.dataset, is_cifar=is_cifar)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[{args.model.upper()}-{args.dataset}] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} (CE {train_ce:.4f}) | "
              f"Train acc: {train_acc:.3f} | "
              f"Test acc: {test_acc:.3f} | "
              f"Time: {epoch_time:.2f}s")

    # Save checkpoint
    ckpt_path = out_dir / f"{args.model}_{args.dataset}_hidden{args.hidden}_best.pth"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path}")

    # Load best and evaluate anytime
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    thresholds = [float(s) for s in args.thresholds.split(",") if s.strip()]
    anytime_results = evaluate_anytime_model(
        model, test_loader, device, thresholds, args.max_steps,
        dataset_name=args.dataset, is_cifar=is_cifar
    )

    print("\n=== Anytime Results ===")
    for thr, res in sorted(anytime_results.items()):
        print(f"Threshold {thr:.2f}: acc={res['acc']:.4f}, "
              f"avg_steps={res['avg_steps']:.3f}")

    # Log results
    with log_path.open("a") as f:
        summary = {
            "type": "summary",
            "model": args.model,
            "dataset": args.dataset,
            "params": int(params),
            "max_steps": args.max_steps,
            "epochs": args.epochs,
            "hidden": args.hidden,
            "best_test_acc_fixed": float(best_test_acc),
        }
        f.write(json.dumps(summary) + "\n")

        for thr, res in anytime_results.items():
            rec = {
                "type": "anytime",
                "model": args.model,
                "dataset": args.dataset,
                "hidden": args.hidden,
                "max_steps": args.max_steps,
                "threshold": thr,
                "acc": float(res["acc"]),
                "avg_steps": float(res["avg_steps"]),
                "step_hist": res["step_hist"],
            }
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

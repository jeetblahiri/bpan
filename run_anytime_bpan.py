# run_anytime_bpan.py
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import BPANClassifier, ConvBackboneCIFAR, ConvWrapper


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------- Dataset helpers ---------------- #

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


# ---------------- Anytime training & evaluation ---------------- #

def train_anytime_epoch(model, optimizer, train_loader, device,
                        dataset_name, T, lambda_act, lambda_bal,
                        is_cifar=False, max_grad_norm=None):
    """
    Train BPAN (or ConvWrapper+BPAN) with cross-entropy at every time step.
    Uses model.forward_with_stats.
    """
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_correct_last = 0
    total_samples = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        if dataset_name == "emnist_letters":
            y = y - 1

        if not is_cifar:
            x_in = x.view(x.size(0), -1)  # [B, D]
        else:
            x_in = x  # images; ConvWrapper will handle backbone

        stats = model.forward_with_stats(x_in)
        logits_seq = stats["logits_seq"]        # [B, T, C]
        r_e_seq = stats["r_e_seq"]              # [B, T, n_exc]
        r_i_seq = stats["r_i_seq"]              # [B, T, n_inh]
        bal_e_seq = stats["bal_e_seq"]          # [B, T, n_exc]
        bal_i_seq = stats["bal_i_seq"]          # [B, T, n_inh]

        # CE at each time step
        ce_t = []
        for t in range(T):
            ce_t.append(F.cross_entropy(logits_seq[:, t, :], y))
        ce_t = torch.stack(ce_t, dim=0)  # [T]

        weights = torch.linspace(0.3, 1.0, steps=T, device=device)
        weights = weights / weights.sum()
        ce = (weights * ce_t).sum()

        # regularization
        act_cost = r_e_seq.pow(2).mean() + r_i_seq.pow(2).mean()
        bal_cost = bal_e_seq.pow(2).mean() + bal_i_seq.pow(2).mean()
        reg = lambda_act * act_cost + lambda_bal * bal_cost

        loss = ce + reg

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        B = x.size(0)
        total_loss += loss.item() * B
        total_ce += ce.item() * B

        logits_last = logits_seq[:, -1, :]
        preds = logits_last.argmax(dim=1)
        total_correct_last += (preds == y).sum().item()
        total_samples += B

    avg_loss = total_loss / total_samples
    avg_ce = total_ce / total_samples
    acc_last = total_correct_last / total_samples
    return avg_loss, avg_ce, acc_last


@torch.no_grad()
def evaluate_fixed_T(model, loader, device, dataset_name, is_cifar=False):
    """
    Evaluate using final time-step logits only.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if dataset_name == "emnist_letters":
            y = y - 1

        if not is_cifar:
            x_in = x.view(x.size(0), -1)
        else:
            x_in = x

        stats = model.forward_with_stats(x_in)
        logits_last = stats["logits_seq"][:, -1, :]
        preds = logits_last.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)

    return total_correct / total_samples


@torch.no_grad()
def anytime_predict_batch(model, x, device, is_cifar, T, threshold):
    """
    Run EI dynamics step-by-step, early-stopping per sample when
    max softmax probability >= threshold.
    Supports both plain BPAN and ConvWrapper+BPAN (CIFAR).
    """
    if is_cifar:
        # x: [B, 3, 32, 32]
        feats = model.backbone(x)      # [B, D]
        ei_layer = model.head.ei_layer
        readout = model.head.readout
        inputs = feats
    else:
        # MNIST/F-MNIST/EMNIST
        inputs = x.view(x.size(0), -1)  # [B, D]
        ei_layer = model.ei_layer
        readout = model.readout

    B = inputs.size(0)
    e = torch.zeros(B, ei_layer.n_exc, device=device)
    i = torch.zeros(B, ei_layer.n_inh, device=device)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    preds = torch.zeros(B, dtype=torch.long, device=device)
    steps_used = torch.zeros(B, dtype=torch.long, device=device)

    for t in range(T):
        e, i, r_e, r_i, bal_e, bal_i = ei_layer.step(inputs, e, i)
        logits = readout(F.relu(e))
        probs = logits.softmax(dim=-1)
        conf, cls = probs.max(dim=-1)

        newly_done = (~done) & (conf >= threshold)
        preds[newly_done] = cls[newly_done]
        steps_used[newly_done] = t + 1
        done = done | newly_done

        if done.all():
            break

    if (~done).any():
        logits = readout(F.relu(e))
        probs = logits.softmax(dim=-1)
        _, cls = probs.max(dim=-1)
        preds[~done] = cls[~done]
        steps_used[~done] = T

    return preds, steps_used


@torch.no_grad()
def evaluate_anytime(model, loader, device, dataset_name, T, thresholds, is_cifar=False):
    """
    For each confidence threshold, compute accuracy and avg steps.
    Returns thr -> {acc, avg_steps, step_hist}.
    """
    model.eval()
    results = {thr: {"correct": 0, "total": 0, "steps_sum": 0,
                     "step_hist": {t: 0 for t in range(1, T+1)}}
               for thr in thresholds}

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if dataset_name == "emnist_letters":
            y = y - 1

        for thr in thresholds:
            preds, steps_used = anytime_predict_batch(
                model, x, device, is_cifar, T, thr
            )
            correct = (preds == y).sum().item()
            results[thr]["correct"] += correct
            results[thr]["total"] += x.size(0)
            results[thr]["steps_sum"] += steps_used.sum().item()
            for t in range(1, T+1):
                results[thr]["step_hist"][t] += (steps_used == t).sum().item()

    final = {}
    for thr, r in results.items():
        total = r["total"]
        acc = r["correct"] / total
        avg_steps = r["steps_sum"] / total
        final[thr] = {
            "acc": acc,
            "avg_steps": avg_steps,
            "step_hist": r["step_hist"],
        }
    return final


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Train BPAN in anytime mode and evaluate early-exit behavior."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "emnist_letters", "cifar10"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--bpan_lambda_act", type=float, default=1e-4)
    parser.add_argument("--bpan_lambda_bal", type=float, default=5e-4)
    parser.add_argument("--frac_train", type=float, default=1.0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_anytime")
    parser.add_argument("--thresholds", type=str, default="0.5,0.6,0.7,0.8,0.9,0.95,0.99",
                        help="Comma-separated confidence thresholds for anytime eval.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.dataset}_bpan_anytime.jsonl"

    train_loader, test_loader, n_classes, in_shape = get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.frac_train,
        num_workers=args.num_workers, pin_memory=use_cuda
    )

    is_cifar = (args.dataset == "cifar10")

    # Build model
    if is_cifar:
        backbone = ConvBackboneCIFAR()
        n_exc = args.hidden // 2
        n_inh = max(4, args.hidden // 8)
        head = BPANClassifier(
            input_dim=backbone.out_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=args.bpan_T,
            lambda_act=args.bpan_lambda_act,
            lambda_bal=args.bpan_lambda_bal,
            use_dales=True,
            use_act_reg=True,
            use_bal_reg=True,
        )
        model = ConvWrapper(backbone, head)
    else:
        input_dim = in_shape[0] * in_shape[1] * in_shape[2]
        n_exc = args.hidden // 2
        n_inh = max(4, args.hidden // 8)
        model = BPANClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=args.bpan_T,
            lambda_act=args.bpan_lambda_act,
            lambda_bal=args.bpan_lambda_bal,
            use_dales=True,
            use_act_reg=True,
            use_bal_reg=True,
        )

    model.to(device)
    params = count_params(model)
    print(f"\nBPAN-anytime on {args.dataset}")
    print(f"Params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_ce, train_acc_last = train_anytime_epoch(
            model, optimizer, train_loader, device,
            args.dataset, args.bpan_T,
            args.bpan_lambda_act, args.bpan_lambda_bal,
            is_cifar=is_cifar, max_grad_norm=args.max_grad_norm
        )
        epoch_time = time.time() - t0

        test_acc_fixed = evaluate_fixed_T(
            model, test_loader, device, args.dataset, is_cifar=is_cifar
        )

        if test_acc_fixed > best_test_acc:
            best_test_acc = test_acc_fixed
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[ANYTIME-BPAN-{args.dataset}] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} (CE {train_ce:.4f}) | "
              f"Train acc (last T): {train_acc_last:.3f} | "
              f"Test acc (fixed T): {test_acc_fixed:.3f} | "
              f"Epoch time: {epoch_time:.2f}s")

    ckpt_path = out_dir / f"bpan_anytime_{args.dataset}_hidden{args.hidden}_best.pth"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved best anytime BPAN checkpoint to {ckpt_path}")

    thresholds = [float(s) for s in args.thresholds.split(",") if s.strip()]
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    anytime_results = evaluate_anytime(
        model, test_loader, device, args.dataset, args.bpan_T, thresholds, is_cifar=is_cifar
    )

    for thr, res in anytime_results.items():
        print(f"Threshold {thr:.2f}: acc={res['acc']:.4f}, "
              f"avg_steps={res['avg_steps']:.3f}, hist={res['step_hist']}")

    with log_path.open("a") as f:
        summary = {
            "type": "summary",
            "model": "bpan",  # Added model key for plotting compatibility
            "dataset": args.dataset,
            "params": int(params),
            "bpan_T": args.bpan_T,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden": args.hidden,
            "lambda_act": args.bpan_lambda_act,
            "lambda_bal": args.bpan_lambda_bal,
            "best_test_acc_fixed_T": float(best_test_acc),
        }
        f.write(json.dumps(summary) + "\n")

        for thr, res in anytime_results.items():
            rec = {
                "type": "anytime",
                "model": "bpan",  # Added model key for plotting compatibility
                "dataset": args.dataset,
                "hidden": args.hidden,
                "bpan_T": args.bpan_T,
                "threshold": thr,
                "acc": float(res["acc"]),
                "avg_steps": float(res["avg_steps"]),
                "step_hist": res["step_hist"],
            }
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()

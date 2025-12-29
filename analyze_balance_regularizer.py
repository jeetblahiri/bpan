# analyze_balance_regularizer.py
"""
Demonstrate what the balance regularizer (L_bal) buys beyond accuracy:
  1. Net current cancellation metrics
  2. Stability under stronger corruptions
  3. Convergence speed
  4. Halting time distributions under anytime inference
  
This script compares BPAN with L_bal=0 vs L_bal>0.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from models import BPANClassifier, ConvBackboneCIFAR, ConvWrapper


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------- Dataset helpers ---------------------- #

def get_test_loader(dataset_name, data_root, batch_size, num_workers=4):
    if dataset_name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif dataset_name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif dataset_name == "emnist_letters":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        test_ds = datasets.EMNIST(root=data_root, split="letters",
                                  train=False, download=True, transform=tfm)
        n_classes = 26
        in_shape = (1, 28, 28)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, n_classes, in_shape


def get_train_loader(dataset_name, data_root, batch_size, frac_train=1.0, num_workers=4):
    if dataset_name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif dataset_name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=tfm)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif dataset_name == "emnist_letters":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        train_ds = datasets.EMNIST(root=data_root, split="letters",
                                   train=True, download=True, transform=tfm)
        n_classes = 26
        in_shape = (1, 28, 28)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if frac_train < 1.0:
        n = len(train_ds)
        k = int(n * frac_train)
        idx = torch.randperm(n)[:k].tolist()
        train_ds = Subset(train_ds, idx)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader, n_classes, in_shape


# ---------------------- Noise augmentation ---------------------- #

def add_noise(x, noise_type, noise_level):
    """Add various types of noise/corruption to input."""
    if noise_type == "none":
        return x
    
    if noise_type == "gaussian":
        noise = torch.randn_like(x) * noise_level
        return x + noise
    
    if noise_type == "sp":
        p = noise_level
        rand = torch.rand_like(x)
        x_noisy = x.clone()
        x_noisy[rand < (p / 2)] = x.min()
        x_noisy[(rand >= (p / 2)) & (rand < p)] = x.max()
        return x_noisy
    
    if noise_type == "occlusion":
        _, C, H, W = x.shape
        side = int(noise_level * min(H, W))
        side = max(1, side)
        y0 = (H - side) // 2
        x0 = (W - side) // 2
        x_occ = x.clone()
        x_occ[:, :, y0:y0+side, x0:x0+side] = 0.0
        return x_occ
    
    raise ValueError(f"Unknown noise type: {noise_type}")


# ---------------------- Analysis functions ---------------------- #

@torch.no_grad()
def compute_balance_metrics(model, loader, device, dataset_name):
    """
    Compute E/I balance metrics:
      - Mean |balance_e| and |balance_i| per time step
      - Net current cancellation ratio
      - Variance of balance signal
    """
    model.eval()
    
    all_bal_e = []
    all_bal_i = []
    all_r_e = []
    all_r_i = []
    
    for x, y in loader:
        x = x.to(device)
        if dataset_name == "emnist_letters":
            y = y - 1
        x = x.view(x.size(0), -1)
        
        stats = model.forward_with_stats(x)
        all_bal_e.append(stats["bal_e_seq"].cpu())
        all_bal_i.append(stats["bal_i_seq"].cpu())
        all_r_e.append(stats["r_e_seq"].cpu())
        all_r_i.append(stats["r_i_seq"].cpu())
    
    bal_e = torch.cat(all_bal_e, dim=0)  # [N, T, n_exc]
    bal_i = torch.cat(all_bal_i, dim=0)  # [N, T, n_inh]
    r_e = torch.cat(all_r_e, dim=0)      # [N, T, n_exc]
    r_i = torch.cat(all_r_i, dim=0)      # [N, T, n_inh]
    
    T = bal_e.shape[1]
    
    # Mean absolute balance per time step
    bal_e_mag = bal_e.abs().mean(dim=(0, 2))  # [T]
    bal_i_mag = bal_i.abs().mean(dim=(0, 2))  # [T]
    
    # Net current cancellation ratio
    # Ideally balance ~ 0 means exc and inh inputs cancel
    exc_input_mag = r_e.abs().mean()
    bal_mag_mean = (bal_e.abs().mean() + bal_i.abs().mean()) / 2
    cancellation_ratio = 1.0 - (bal_mag_mean / (exc_input_mag + 1e-8))
    
    # Variance of balance over samples (for each t)
    bal_e_var = bal_e.var(dim=0).mean(dim=1)  # [T]
    bal_i_var = bal_i.var(dim=0).mean(dim=1)  # [T]
    
    return {
        "bal_e_mag_per_t": bal_e_mag.numpy(),
        "bal_i_mag_per_t": bal_i_mag.numpy(),
        "bal_e_var_per_t": bal_e_var.numpy(),
        "bal_i_var_per_t": bal_i_var.numpy(),
        "cancellation_ratio": float(cancellation_ratio),
        "mean_exc_activity": float(r_e.mean()),
        "mean_inh_activity": float(r_i.mean()),
    }


@torch.no_grad()
def evaluate_robustness(model, loader, device, dataset_name,
                        noise_types, noise_levels):
    """
    Evaluate accuracy under various noise conditions.
    """
    model.eval()
    results = {}
    
    for noise_type in noise_types:
        for noise_level in noise_levels:
            correct = 0
            total = 0
            
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                
                if dataset_name == "emnist_letters":
                    y = y - 1
                
                x_noisy = add_noise(x, noise_type, noise_level)
                x_noisy = x_noisy.view(x_noisy.size(0), -1)
                
                out = model(x_noisy, targets=None)
                preds = out["logits"].argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            
            acc = correct / total
            results[(noise_type, noise_level)] = acc
    
    return results


def train_and_track_convergence(model, train_loader, test_loader, device,
                                dataset_name, epochs, lr, max_grad_norm=5.0):
    """
    Train model and track convergence metrics per epoch.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
        "bal_cost": [],
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_bal = 0.0
        total_correct = 0
        total_samples = 0
        
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            
            if dataset_name == "emnist_letters":
                y = y - 1
            
            x = x.view(x.size(0), -1)
            
            out = model(x, targets=y)
            loss = out["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item() * x.size(0)
            total_bal += out["bal_cost"].item() * x.size(0)
            preds = out["logits"].argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)
        
        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples
        bal_cost = total_bal / total_samples
        
        # Test accuracy
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                if dataset_name == "emnist_letters":
                    y = y - 1
                x = x.view(x.size(0), -1)
                out = model(x, targets=None)
                preds = out["logits"].argmax(dim=1)
                test_correct += (preds == y).sum().item()
                test_total += y.size(0)
        
        test_acc = test_correct / test_total
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["bal_cost"].append(bal_cost)
        
        print(f"  Epoch {epoch+1:02d}: loss={train_loss:.4f}, "
              f"train_acc={train_acc:.3f}, test_acc={test_acc:.3f}, "
              f"bal_cost={bal_cost:.6f}")
    
    return history


@torch.no_grad()
def analyze_halting_distribution(model, loader, device, dataset_name, T, thresholds):
    """
    Analyze halting time distribution under anytime inference.
    """
    model.eval()
    
    results = {thr: {"steps": [], "correct": []} for thr in thresholds}
    
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        
        if dataset_name == "emnist_letters":
            y = y - 1
        
        x_flat = x.view(x.size(0), -1)
        
        for thr in thresholds:
            preds, steps_used = anytime_predict_bpan(model, x_flat, device, T, thr)
            results[thr]["steps"].extend(steps_used.cpu().tolist())
            results[thr]["correct"].extend((preds == y).cpu().tolist())
    
    # Compute stats
    final = {}
    for thr in thresholds:
        steps = np.array(results[thr]["steps"])
        correct = np.array(results[thr]["correct"])
        
        final[thr] = {
            "avg_steps": float(steps.mean()),
            "std_steps": float(steps.std()),
            "median_steps": float(np.median(steps)),
            "acc": float(correct.mean()),
            "step_hist": {t: int((steps == t).sum()) for t in range(1, T + 1)},
        }
    
    return final


def anytime_predict_bpan(model, x, device, T, threshold):
    """
    Early-exit BPAN prediction based on confidence threshold.
    """
    B = x.size(0)
    
    ei_layer = model.ei_layer
    readout = model.readout
    
    e = torch.zeros(B, ei_layer.n_exc, device=device)
    i = torch.zeros(B, ei_layer.n_inh, device=device)
    
    done = torch.zeros(B, dtype=torch.bool, device=device)
    preds = torch.zeros(B, dtype=torch.long, device=device)
    steps_used = torch.zeros(B, dtype=torch.long, device=device)
    
    for t in range(T):
        e, i, r_e, r_i, bal_e, bal_i = ei_layer.step(x, e, i)
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
        _, cls = logits.max(dim=-1)
        preds[~done] = cls[~done]
        steps_used[~done] = T
    
    return preds, steps_used


# ---------------------- Plotting ---------------------- #

def plot_balance_comparison(metrics_with_bal, metrics_no_bal, save_path=None):
    """
    Compare balance metrics with and without L_bal regularizer.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    T = len(metrics_with_bal["bal_e_mag_per_t"])
    t_axis = np.arange(T)
    
    # Plot 1: Mean |balance_e| over time
    ax = axes[0, 0]
    ax.plot(t_axis, metrics_with_bal["bal_e_mag_per_t"], "o-", 
            label="With $L_{bal}$", color="#2E86AB", linewidth=2)
    ax.plot(t_axis, metrics_no_bal["bal_e_mag_per_t"], "s--",
            label="Without $L_{bal}$", color="#C73E1D", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean $|\\text{balance}_E|$")
    ax.set_title("Excitatory Balance Magnitude")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: Mean |balance_i| over time
    ax = axes[0, 1]
    ax.plot(t_axis, metrics_with_bal["bal_i_mag_per_t"], "o-",
            label="With $L_{bal}$", color="#2E86AB", linewidth=2)
    ax.plot(t_axis, metrics_no_bal["bal_i_mag_per_t"], "s--",
            label="Without $L_{bal}$", color="#C73E1D", linewidth=2)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Mean $|\\text{balance}_I|$")
    ax.set_title("Inhibitory Balance Magnitude")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 3: Balance variance over time
    ax = axes[1, 0]
    ax.plot(t_axis, metrics_with_bal["bal_e_var_per_t"], "o-",
            label="E balance var (with $L_{bal}$)", color="#2E86AB")
    ax.plot(t_axis, metrics_no_bal["bal_e_var_per_t"], "s--",
            label="E balance var (no $L_{bal}$)", color="#C73E1D")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Variance")
    ax.set_title("Balance Variance Over Time")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 4: Cancellation ratio comparison
    ax = axes[1, 1]
    models = ["With $L_{bal}$", "Without $L_{bal}$"]
    cancellations = [metrics_with_bal["cancellation_ratio"], 
                     metrics_no_bal["cancellation_ratio"]]
    colors = ["#2E86AB", "#C73E1D"]
    bars = ax.bar(models, cancellations, color=colors, edgecolor="black")
    ax.set_ylabel("E/I Cancellation Ratio")
    ax.set_title("Net Current Cancellation (Higher = Better Balance)")
    ax.set_ylim(0, 1)
    for bar, val in zip(bars, cancellations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha='center', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved balance comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_robustness_comparison(robust_with_bal, robust_no_bal, noise_types, 
                                noise_levels, save_path=None):
    """
    Compare robustness to corruptions.
    """
    n_noise = len(noise_types)
    fig, axes = plt.subplots(1, n_noise, figsize=(5*n_noise, 4))
    if n_noise == 1:
        axes = [axes]
    
    for ax, noise_type in zip(axes, noise_types):
        accs_with = [robust_with_bal[(noise_type, lvl)] for lvl in noise_levels]
        accs_no = [robust_no_bal[(noise_type, lvl)] for lvl in noise_levels]
        
        ax.plot(noise_levels, accs_with, "o-", label="With $L_{bal}$", 
                color="#2E86AB", linewidth=2)
        ax.plot(noise_levels, accs_no, "s--", label="Without $L_{bal}$",
                color="#C73E1D", linewidth=2)
        ax.set_xlabel(f"{noise_type.capitalize()} Level")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Robustness to {noise_type.capitalize()} Noise")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved robustness comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_comparison(hist_with_bal, hist_no_bal, save_path=None):
    """
    Compare training convergence.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = np.arange(1, len(hist_with_bal["train_loss"]) + 1)
    
    # Train loss
    ax = axes[0]
    ax.plot(epochs, hist_with_bal["train_loss"], "o-", label="With $L_{bal}$",
            color="#2E86AB")
    ax.plot(epochs, hist_no_bal["train_loss"], "s--", label="Without $L_{bal}$",
            color="#C73E1D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("Training Loss Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Test accuracy
    ax = axes[1]
    ax.plot(epochs, hist_with_bal["test_acc"], "o-", label="With $L_{bal}$",
            color="#2E86AB")
    ax.plot(epochs, hist_no_bal["test_acc"], "s--", label="Without $L_{bal}$",
            color="#C73E1D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy Convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Balance cost
    ax = axes[2]
    ax.plot(epochs, hist_with_bal["bal_cost"], "o-", label="With $L_{bal}$",
            color="#2E86AB")
    ax.plot(epochs, hist_no_bal["bal_cost"], "s--", label="Without $L_{bal}$",
            color="#C73E1D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Balance Cost")
    ax.set_title("E/I Balance Cost Over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved convergence comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_halting_comparison(halt_with_bal, halt_no_bal, T, save_path=None):
    """
    Compare halting time distributions.
    """
    thresholds = list(halt_with_bal.keys())
    
    fig, axes = plt.subplots(2, len(thresholds), figsize=(4*len(thresholds), 8))
    
    for col, thr in enumerate(thresholds):
        # With L_bal
        ax = axes[0, col]
        hist = halt_with_bal[thr]["step_hist"]
        steps = list(range(1, T + 1))
        counts = [hist.get(t, 0) for t in steps]
        total = sum(counts)
        fracs = [c / total if total > 0 else 0 for c in counts]
        
        ax.bar(steps, fracs, color="#2E86AB", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Halting Step")
        ax.set_ylabel("Fraction")
        ax.set_title(f"With $L_{{bal}}$ — thr={thr:.2f}\n"
                    f"Acc: {halt_with_bal[thr]['acc']:.3f}, "
                    f"Avg: {halt_with_bal[thr]['avg_steps']:.2f}")
        ax.set_xticks(steps)
        ax.set_ylim(0, 1)
        
        # Without L_bal
        ax = axes[1, col]
        hist = halt_no_bal[thr]["step_hist"]
        counts = [hist.get(t, 0) for t in steps]
        total = sum(counts)
        fracs = [c / total if total > 0 else 0 for c in counts]
        
        ax.bar(steps, fracs, color="#C73E1D", alpha=0.8, edgecolor="black")
        ax.set_xlabel("Halting Step")
        ax.set_ylabel("Fraction")
        ax.set_title(f"Without $L_{{bal}}$ — thr={thr:.2f}\n"
                    f"Acc: {halt_no_bal[thr]['acc']:.3f}, "
                    f"Avg: {halt_no_bal[thr]['avg_steps']:.2f}")
        ax.set_xticks(steps)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved halting comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


# ---------------------- Main ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Analyze what the balance regularizer L_bal buys."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "emnist_letters"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--lambda_bal", type=float, default=5e-4,
                        help="L_bal weight for 'with balance' model.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./results_balance_analysis")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, n_classes, in_shape = get_train_loader(
        args.dataset, args.data_root, args.batch_size
    )
    test_loader, _, _ = get_test_loader(
        args.dataset, args.data_root, args.batch_size
    )

    input_dim = in_shape[0] * in_shape[1] * in_shape[2]
    n_exc = args.hidden // 2
    n_inh = max(4, args.hidden // 8)

    # ========== Train model WITH L_bal ==========
    print("\n========== Training BPAN WITH L_bal ==========")
    model_with_bal = BPANClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        n_exc=n_exc,
        n_inh=n_inh,
        T=args.bpan_T,
        lambda_act=1e-4,
        lambda_bal=args.lambda_bal,
        use_dales=True,
        use_act_reg=True,
        use_bal_reg=True,
    ).to(device)

    hist_with_bal = train_and_track_convergence(
        model_with_bal, train_loader, test_loader, device,
        args.dataset, args.epochs, args.lr
    )

    # ========== Train model WITHOUT L_bal ==========
    print("\n========== Training BPAN WITHOUT L_bal ==========")
    model_no_bal = BPANClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        n_exc=n_exc,
        n_inh=n_inh,
        T=args.bpan_T,
        lambda_act=1e-4,
        lambda_bal=0.0,
        use_dales=True,
        use_act_reg=True,
        use_bal_reg=False,
    ).to(device)

    hist_no_bal = train_and_track_convergence(
        model_no_bal, train_loader, test_loader, device,
        args.dataset, args.epochs, args.lr
    )

    # ========== Compute balance metrics ==========
    print("\n========== Computing balance metrics ==========")
    metrics_with_bal = compute_balance_metrics(
        model_with_bal, test_loader, device, args.dataset
    )
    metrics_no_bal = compute_balance_metrics(
        model_no_bal, test_loader, device, args.dataset
    )

    print(f"Cancellation ratio WITH L_bal: {metrics_with_bal['cancellation_ratio']:.4f}")
    print(f"Cancellation ratio WITHOUT L_bal: {metrics_no_bal['cancellation_ratio']:.4f}")

    # ========== Evaluate robustness ==========
    print("\n========== Evaluating robustness ==========")
    noise_types = ["gaussian", "sp", "occlusion"]
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    robust_with_bal = evaluate_robustness(
        model_with_bal, test_loader, device, args.dataset,
        noise_types, noise_levels
    )
    robust_no_bal = evaluate_robustness(
        model_no_bal, test_loader, device, args.dataset,
        noise_types, noise_levels
    )

    # ========== Analyze halting distributions ==========
    print("\n========== Analyzing halting distributions ==========")
    thresholds = [0.7, 0.9, 0.95]

    halt_with_bal = analyze_halting_distribution(
        model_with_bal, test_loader, device, args.dataset, args.bpan_T, thresholds
    )
    halt_no_bal = analyze_halting_distribution(
        model_no_bal, test_loader, device, args.dataset, args.bpan_T, thresholds
    )

    # ========== Generate plots ==========
    print("\n========== Generating plots ==========")
    
    plot_balance_comparison(
        metrics_with_bal, metrics_no_bal,
        save_path=out_dir / f"balance_comparison_{args.dataset}.png"
    )

    plot_robustness_comparison(
        robust_with_bal, robust_no_bal, noise_types, noise_levels,
        save_path=out_dir / f"robustness_comparison_{args.dataset}.png"
    )

    plot_convergence_comparison(
        hist_with_bal, hist_no_bal,
        save_path=out_dir / f"convergence_comparison_{args.dataset}.png"
    )

    plot_halting_comparison(
        halt_with_bal, halt_no_bal, args.bpan_T,
        save_path=out_dir / f"halting_comparison_{args.dataset}.png"
    )

    # ========== Save summary ==========
    summary = {
        "dataset": args.dataset,
        "epochs": args.epochs,
        "lambda_bal": args.lambda_bal,
        "metrics_with_bal": metrics_with_bal,
        "metrics_no_bal": metrics_no_bal,
        "final_test_acc_with_bal": hist_with_bal["test_acc"][-1],
        "final_test_acc_no_bal": hist_no_bal["test_acc"][-1],
        "robustness_with_bal": {str(k): v for k, v in robust_with_bal.items()},
        "robustness_no_bal": {str(k): v for k, v in robust_no_bal.items()},
        "halting_with_bal": halt_with_bal,
        "halting_no_bal": halt_no_bal,
    }

    with open(out_dir / f"summary_{args.dataset}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()

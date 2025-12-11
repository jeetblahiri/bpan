# run_glimpse_experiment.py
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from models import MLPClassifier, BPANClassifier  # existing from your repo


# ---------------- Glimpse dataset ---------------- #

class GlimpseMNIST(Dataset):
    """
    Wraps MNIST/Fashion-MNIST/EMNIST and returns a sequence of L glimpses.

    Each glimpse is a full 28x28 image with only a random k x k patch visible
    (rest zero). Glimpses are independent random patches from the same image.

    __getitem__ returns:
        seq: [L, 1, 28, 28]
        label: int
    """
    def __init__(self, base_ds, L=4, patch_size=10):
        self.base_ds = base_ds
        self.L = L
        self.patch_size = patch_size

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]       # img: [1, 28, 28]
        C, H, W = img.shape
        L = self.L
        k = self.patch_size

        seq = []
        for _ in range(L):
            x_t = torch.zeros_like(img)
            # sample random patch location
            if H == k:
                y0 = 0
            else:
                y0 = torch.randint(0, H - k + 1, (1,)).item()
            if W == k:
                x0 = 0
            else:
                x0 = torch.randint(0, W - k + 1, (1,)).item()
            x_t[:, y0:y0+k, x0:x0+k] = img[:, y0:y0+k, x0:x0+k]
            seq.append(x_t)

        seq = torch.stack(seq, dim=0)  # [L, 1, 28, 28]
        return seq, label


def get_base_dataset(name, data_root, train):
    if name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = datasets.MNIST(root=data_root, train=train, download=True, transform=tfm)
        n_classes = 10
    elif name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        ds = datasets.FashionMNIST(root=data_root, train=train, download=True, transform=tfm)
        n_classes = 10
    elif name == "emnist_letters":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        ds = datasets.EMNIST(root=data_root, split="letters",
                             train=train, download=True, transform=tfm)
        n_classes = 26
    else:
        raise ValueError(f"Unsupported dataset for glimpses: {name}")
    return ds, n_classes


def get_glimpse_loaders(dataset_name, data_root, batch_size,
                        L, patch_size, num_workers, pin_memory):
    base_train, n_classes = get_base_dataset(dataset_name, data_root, train=True)
    base_test, _ = get_base_dataset(dataset_name, data_root, train=False)

    train_ds = GlimpseMNIST(base_train, L=L, patch_size=patch_size)
    test_ds = GlimpseMNIST(base_test, L=L, patch_size=patch_size)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, test_loader, n_classes


# ---------------- Baseline: GRU classifier ---------------- #

class GRUClassifier(nn.Module):
    """
    Simple GRU over flattened glimpses.

    x_seq: [B, L, D]
    """
    def __init__(self, input_dim, n_classes, hidden=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x_seq, targets=None):
        """
        x_seq: [B, L, D]
        """
        out, h_n = self.gru(x_seq)     # out: [B, L, H]
        h_last = out[:, -1, :]         # [B, H]
        logits = self.fc(h_last)       # [B, C]
        out_dict = {"logits": logits}

        if targets is not None:
            ce = F.cross_entropy(logits, targets)
            out_dict["loss"] = ce
            out_dict["ce_loss"] = ce

        return out_dict

    def forward_seq(self, x_seq):
        """
        Return logits at each time step.

        x_seq: [B, L, D]
        returns: logits_seq [B, L, C]
        """
        out, _ = self.gru(x_seq)       # [B, L, H]
        logits_seq = self.fc(out)      # [B, L, C]
        return logits_seq


# ---------------- BPAN sequence model ---------------- #

class BPANSequence(nn.Module):
    """
    BPAN used as a sequential evidence integrator.

    On each external time step t, we:
        - feed glimpse x_t
        - update EI state
        - read out class logits from excitatory activity.

    All EI parameters are shared across time; we train with CE at every step.
    """
    def __init__(self, input_dim, n_classes,
                 n_exc=128, n_inh=32,
                 lambda_act=1e-4, lambda_bal=5e-4):
        super().__init__()
        # Reuse BPANClassifier's internal structure to avoid guessing signatures
        base = BPANClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            T=1,                     # we'll control external time ourselves
            lambda_act=lambda_act,
            lambda_bal=lambda_bal,
            use_dales=True,
            use_act_reg=True,
            use_bal_reg=True,
        )
        self.ei_layer = base.ei_layer
        self.readout = base.readout
        self.lambda_act = lambda_act
        self.lambda_bal = lambda_bal
        self.n_exc = n_exc
        self.n_inh = n_inh

    def forward(self, x_seq, targets=None):
        """
        x_seq: [B, L, D]
        """
        device = x_seq.device
        B, L, D = x_seq.shape

        e = torch.zeros(B, self.n_exc, device=device)
        i = torch.zeros(B, self.n_inh, device=device)

        logits_list = []
        r_e_list = []
        r_i_list = []
        bal_e_list = []
        bal_i_list = []

        for t in range(L):
            x_t = x_seq[:, t, :]            # [B, D]
            e, i, r_e, r_i, bal_e, bal_i = self.ei_layer.step(x_t, e, i)
            logits_t = self.readout(F.relu(e))   # [B, C]

            logits_list.append(logits_t)
            r_e_list.append(r_e)
            r_i_list.append(r_i)
            bal_e_list.append(bal_e)
            bal_i_list.append(bal_i)

        logits_seq = torch.stack(logits_list, dim=1)   # [B, L, C]
        r_e_seq = torch.stack(r_e_list, dim=1)         # [B, L, n_exc]
        r_i_seq = torch.stack(r_i_list, dim=1)         # [B, L, n_inh]
        bal_e_seq = torch.stack(bal_e_list, dim=1)     # [B, L, n_exc]
        bal_i_seq = torch.stack(bal_i_list, dim=1)     # [B, L, n_inh]

        out = {
            "logits_seq": logits_seq,
            "logits": logits_seq[:, -1, :],   # final step
            "r_e_seq": r_e_seq,
            "r_i_seq": r_i_seq,
            "bal_e_seq": bal_e_seq,
            "bal_i_seq": bal_i_seq,
        }

        if targets is not None:
            ce_t = []
            for t in range(L):
                ce_t.append(F.cross_entropy(logits_seq[:, t, :], targets))
            ce_t = torch.stack(ce_t, dim=0)  # [L]

            # later glimpses weighted slightly more
            weights = torch.linspace(0.3, 1.0, steps=L, device=device)
            weights = weights / weights.sum()
            ce = (weights * ce_t).sum()

            act_cost = r_e_seq.pow(2).mean() + r_i_seq.pow(2).mean()
            bal_cost = bal_e_seq.pow(2).mean() + bal_i_seq.pow(2).mean()
            reg = self.lambda_act * act_cost + self.lambda_bal * bal_cost

            loss = ce + reg

            out["loss"] = loss
            out["ce_loss"] = ce
            out["reg_loss"] = reg

        return out

    def forward_seq(self, x_seq):
        """
        For evaluation: return logits at each time step.
        """
        out = self.forward(x_seq, targets=None)
        return out["logits_seq"]


# ---------------- Training / evaluation loops ---------------- #

def train_one_epoch(model_type, model, optimizer, loader, device):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_correct = 0
    total_samples = 0

    for seq, labels in loader:
        seq = seq.to(device)    # [B, L, 1, 28, 28]
        labels = labels.to(device)

        if model_type == "mlp_final":
            B, L, C, H, W = seq.shape
            x_last = seq[:, -1, :, :, :].view(B, -1)   # [B, 784]
            out = model(x_last, targets=labels)
            logits = out["logits"]
            loss = out["loss"]
            ce_loss = loss          # <--- use loss as CE
        else:
            B, L, C, H, W = seq.shape
            x_seq = seq.view(B, L, -1)   # [B, L, D]

            if model_type == "gru":
                out = model(x_seq, targets=labels)
                logits = out["logits"]
                loss = out["loss"]
                ce_loss = out["ce_loss"]
            elif model_type == "bpan_seq":
                out = model(x_seq, targets=labels)
                logits = out["logits"]
                loss = out["loss"]
                ce_loss = out["ce_loss"]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        total_ce += ce_loss.item() * B

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += B

    avg_loss = total_loss / total_samples
    avg_ce = total_ce / total_samples
    acc = total_correct / total_samples
    return avg_loss, avg_ce, acc



@torch.no_grad()
def evaluate_final(model_type, model, loader, device):
    """
    Evaluate only on final prediction after all glimpses.
    """
    model.eval()
    total_correct = 0
    total_samples = 0

    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)

        if model_type == "mlp_final":
            B, L, C, H, W = seq.shape
            x_last = seq[:, -1, :, :, :].view(B, -1)
            out = model(x_last, targets=labels)
            logits = out["logits"]

        else:
            B, L, C, H, W = seq.shape
            x_seq = seq.view(B, L, -1)

            if model_type == "gru":
                out = model(x_seq, targets=None)
                logits = out["logits"]
            elif model_type == "bpan_seq":
                out = model(x_seq, targets=None)
                logits = out["logits"]
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_correct / total_samples


@torch.no_grad()
def evaluate_seq(model_type, model, loader, device, L):
    """
    For 'gru' and 'bpan_seq': accuracy after each glimpse t=1..L.
    For 'mlp_final': only final step accuracy (others None).

    Returns:
        acc_per_t: list length L (floats or None).
    """
    model.eval()

    if model_type == "mlp_final":
        # Only final step meaningful
        final_acc = evaluate_final(model_type, model, loader, device)
        acc_per_t = [None] * (L - 1) + [final_acc]
        return acc_per_t

    # For sequential models
    correct_t = torch.zeros(L, dtype=torch.long)
    total = 0

    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        B, L_, C, H, W = seq.shape
        assert L_ == L
        x_seq = seq.view(B, L, -1)   # [B, L, D]

        if model_type == "gru":
            logits_seq = model.forward_seq(x_seq)     # [B, L, C]
        elif model_type == "bpan_seq":
            logits_seq = model.forward_seq(x_seq)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        preds_seq = logits_seq.argmax(dim=-1)         # [B, L]
        for t in range(L):
            correct_t[t] += (preds_seq[:, t] == labels).sum().item()
        total += B

    acc_per_t = [(correct_t[t].item() / total) for t in range(L)]
    return acc_per_t


# ---------------- Main ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Sequential glimpse experiment: MLP-final vs GRU vs BPAN-seq."
    )
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "emnist_letters"])
    parser.add_argument("--model", type=str, default="bpan_seq",
                        choices=["mlp_final", "gru", "bpan_seq"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256,
                        help="Hidden size for MLP, GRU, and BPAN excitatory+inhib.")
    parser.add_argument("--L", type=int, default=4,
                        help="Number of glimpses per sample.")
    parser.add_argument("--patch_size", type=int, default=10,
                        help="Side length of square glimpse patch.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_glimpse")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.dataset}_{args.model}_L{args.L}_p{args.patch_size}.jsonl"

    train_loader, test_loader, n_classes = get_glimpse_loaders(
        args.dataset, args.data_root, args.batch_size,
        L=args.L, patch_size=args.patch_size,
        num_workers=args.num_workers, pin_memory=use_cuda
    )

    # Build model
    input_dim = 28 * 28
    if args.model == "mlp_final":
        model = MLPClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden=args.hidden,
        )
    elif args.model == "gru":
        model = GRUClassifier(
            input_dim=input_dim,
            n_classes=n_classes,
            hidden=args.hidden,
        )
    elif args.model == "bpan_seq":
        n_exc = args.hidden // 2
        n_inh = max(4, args.hidden // 8)
        model = BPANSequence(
            input_dim=input_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            lambda_act=1e-4,
            lambda_bal=5e-4,
        )
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model} on {args.dataset} (glimpses)")
    print(f"Glimpses L={args.L}, patch_size={args.patch_size}")
    print(f"Trainable params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_final_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_ce, train_acc = train_one_epoch(
            args.model, model, optimizer, train_loader, device
        )
        epoch_time = time.time() - t0

        final_acc = evaluate_final(args.model, model, test_loader, device)

        if final_acc > best_final_acc:
            best_final_acc = final_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[{args.model.upper()}-{args.dataset}-GLIMPSE] "
              f"Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} (CE {train_ce:.4f}) | "
              f"Train acc: {train_acc:.3f} | "
              f"Final test acc: {final_acc:.3f} | "
              f"Epoch time: {epoch_time:.2f}s")

    # Save best checkpoint
    ckpt_path = out_dir / f"{args.model}_{args.dataset}_L{args.L}_p{args.patch_size}_best.pth"
    if best_state is not None:
        torch.save(best_state, ckpt_path)
        print(f"Saved best checkpoint to {ckpt_path}")

    # Load best and compute accuracy per glimpse (for sequential models)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    acc_per_t = evaluate_seq(args.model, model, test_loader, device, args.L)

    print("Accuracy per glimpse step:", acc_per_t)

    # Log JSON
    with log_path.open("a") as f:
        record = {
            "dataset": args.dataset,
            "model": args.model,
            "params": int(params),
            "L": args.L,
            "patch_size": args.patch_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "best_final_acc": float(best_final_acc),
            "acc_per_t": acc_per_t,
        }
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

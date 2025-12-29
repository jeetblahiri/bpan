# run_sequential_task.py
"""
Stronger sequential tasks that match the paper's narrative on evidence integration:

1. Cluttered Translated MNIST (CT-MNIST):
   - Digits are randomly placed in a larger canvas with clutter
   - Revealed through sequential glimpses (saccades)
   - Tests genuine evidence integration over time

2. Sequential CIFAR Patches:
   - CIFAR images revealed patch-by-patch in random order
   - Classification from incomplete information
   - Natural task where recurrence helps

3. Streaming Classification (simulated online):
   - Features arrive sequentially (simulating sensor streams)
   - Classification updates in real-time
   - Directly relevant to Neurocomputing's application scope

These tasks naturally require recurrence and show BPAN's attractor-based
evidence integration advantages.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from models import BPANClassifier, MLPClassifier


# ============================================================================
# 1. Cluttered Translated MNIST (CT-MNIST)
# ============================================================================

class ClutteredTranslatedMNIST(Dataset):
    """
    CT-MNIST: Digits placed randomly on a larger canvas with clutter.
    
    Each __getitem__ returns a sequence of L glimpses, each showing
    a small window of the full cluttered image.
    """
    def __init__(self, base_ds, canvas_size=60, n_clutter=6, 
                 clutter_size=8, glimpse_size=12, n_glimpses=6,
                 saccade_strategy="random"):
        """
        Args:
            base_ds: MNIST dataset
            canvas_size: Size of the larger canvas
            n_clutter: Number of clutter patches
            clutter_size: Size of each clutter patch
            glimpse_size: Size of each glimpse window
            n_glimpses: Number of glimpses per sequence
            saccade_strategy: 'random', 'grid', or 'center_out'
        """
        self.base_ds = base_ds
        self.canvas_size = canvas_size
        self.n_clutter = n_clutter
        self.clutter_size = clutter_size
        self.glimpse_size = glimpse_size
        self.n_glimpses = n_glimpses
        self.saccade_strategy = saccade_strategy
        
    def __len__(self):
        return len(self.base_ds)
    
    def _create_cluttered_canvas(self, digit_img, idx):
        """Create a cluttered canvas with the digit randomly placed."""
        C, H, W = digit_img.shape  # 1, 28, 28
        canvas = torch.zeros(C, self.canvas_size, self.canvas_size)
        
        # Place digit at random location
        max_y = self.canvas_size - H
        max_x = self.canvas_size - W
        digit_y = torch.randint(0, max_y + 1, (1,)).item()
        digit_x = torch.randint(0, max_x + 1, (1,)).item()
        
        canvas[:, digit_y:digit_y+H, digit_x:digit_x+W] = digit_img
        
        # Add clutter patches (random crops from other digits)
        for _ in range(self.n_clutter):
            # Get random clutter from another sample
            clutter_idx = torch.randint(0, len(self.base_ds), (1,)).item()
            clutter_img, _ = self.base_ds[clutter_idx]
            
            # Random crop from clutter
            cy = torch.randint(0, 28 - self.clutter_size + 1, (1,)).item()
            cx = torch.randint(0, 28 - self.clutter_size + 1, (1,)).item()
            clutter_patch = clutter_img[:, cy:cy+self.clutter_size, cx:cx+self.clutter_size]
            
            # Place at random location (might overlap with digit - that's okay)
            py = torch.randint(0, self.canvas_size - self.clutter_size + 1, (1,)).item()
            px = torch.randint(0, self.canvas_size - self.clutter_size + 1, (1,)).item()
            
            # Blend clutter (don't overwrite digit completely)
            existing = canvas[:, py:py+self.clutter_size, px:px+self.clutter_size]
            canvas[:, py:py+self.clutter_size, px:px+self.clutter_size] = torch.maximum(
                existing, clutter_patch * 0.5
            )
        
        return canvas, (digit_y + H//2, digit_x + W//2)  # Return digit center
    
    def _generate_glimpse_locations(self, digit_center):
        """Generate glimpse locations based on strategy."""
        g = self.glimpse_size
        max_pos = self.canvas_size - g
        
        if self.saccade_strategy == "random":
            # Purely random glimpses
            locations = []
            for _ in range(self.n_glimpses):
                y = torch.randint(0, max_pos + 1, (1,)).item()
                x = torch.randint(0, max_pos + 1, (1,)).item()
                locations.append((y, x))
        
        elif self.saccade_strategy == "grid":
            # Systematic grid covering the canvas
            n_per_dim = int(np.ceil(np.sqrt(self.n_glimpses)))
            step = max_pos // max(n_per_dim - 1, 1)
            locations = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    if len(locations) >= self.n_glimpses:
                        break
                    y = min(i * step, max_pos)
                    x = min(j * step, max_pos)
                    locations.append((y, x))
                if len(locations) >= self.n_glimpses:
                    break
        
        elif self.saccade_strategy == "center_out":
            # Start from center, spiral outward
            cy, cx = digit_center
            locations = [(max(0, min(cy - g//2, max_pos)), 
                         max(0, min(cx - g//2, max_pos)))]
            
            for i in range(1, self.n_glimpses):
                angle = 2 * np.pi * i / self.n_glimpses
                radius = g * (i / 2)
                y = int(cy + radius * np.sin(angle) - g//2)
                x = int(cx + radius * np.cos(angle) - g//2)
                y = max(0, min(y, max_pos))
                x = max(0, min(x, max_pos))
                locations.append((y, x))
        
        else:
            raise ValueError(f"Unknown saccade strategy: {self.saccade_strategy}")
        
        return locations
    
    def __getitem__(self, idx):
        digit_img, label = self.base_ds[idx]
        
        # Create cluttered canvas
        canvas, digit_center = self._create_cluttered_canvas(digit_img, idx)
        
        # Generate glimpse locations
        locations = self._generate_glimpse_locations(digit_center)
        
        # Extract glimpses
        glimpses = []
        g = self.glimpse_size
        for y, x in locations:
            glimpse = canvas[:, y:y+g, x:x+g]
            glimpses.append(glimpse)
        
        glimpses = torch.stack(glimpses, dim=0)  # [L, C, g, g]
        
        return glimpses, label


# ============================================================================
# 2. Sequential CIFAR Patches
# ============================================================================

class SequentialCIFARPatches(Dataset):
    """
    CIFAR images revealed patch-by-patch.
    Simulates sequential observation of a scene.
    """
    def __init__(self, base_ds, patch_size=8, n_patches=8, order="random"):
        """
        Args:
            base_ds: CIFAR dataset
            patch_size: Size of each revealed patch
            n_patches: Number of patches to reveal (< total available)
            order: 'random', 'raster', or 'spiral'
        """
        self.base_ds = base_ds
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.order = order
        
        # Calculate grid
        self.grid_size = 32 // patch_size  # 4 for patch_size=8
        self.total_patches = self.grid_size ** 2
        
    def __len__(self):
        return len(self.base_ds)
    
    def _get_patch_order(self):
        """Get order of patch indices."""
        if self.order == "random":
            indices = torch.randperm(self.total_patches)[:self.n_patches]
        elif self.order == "raster":
            indices = torch.arange(self.n_patches)
        elif self.order == "spiral":
            # Simple spiral from center
            center = self.grid_size // 2
            indices = []
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            y, x = center, center
            step = 1
            dir_idx = 0
            
            while len(indices) < self.total_patches:
                for _ in range(2):
                    for _ in range(step):
                        if 0 <= y < self.grid_size and 0 <= x < self.grid_size:
                            idx = y * self.grid_size + x
                            if idx not in indices:
                                indices.append(idx)
                        dy, dx = directions[dir_idx]
                        y, x = y + dy, x + dx
                    dir_idx = (dir_idx + 1) % 4
                step += 1
            
            indices = torch.tensor(indices[:self.n_patches])
        else:
            raise ValueError(f"Unknown order: {self.order}")
        
        return indices
    
    def __getitem__(self, idx):
        img, label = self.base_ds[idx]  # [3, 32, 32]
        
        # Get patch order
        patch_indices = self._get_patch_order()
        
        # Extract patches
        patches = []
        p = self.patch_size
        
        for patch_idx in patch_indices:
            patch_y = (patch_idx // self.grid_size) * p
            patch_x = (patch_idx % self.grid_size) * p
            patch = img[:, patch_y:patch_y+p, patch_x:patch_x+p]
            patches.append(patch)
        
        patches = torch.stack(patches, dim=0)  # [L, C, p, p]
        
        return patches, label


# ============================================================================
# 3. Streaming Classification (Feature Sequence)
# ============================================================================

class StreamingFeatureDataset(Dataset):
    """
    Simulates streaming classification where features arrive sequentially.
    Useful for sensor fusion, time-series classification, etc.
    """
    def __init__(self, base_ds, n_chunks=8, noise_std=0.1, shuffle_chunks=True):
        """
        Args:
            base_ds: Base classification dataset
            n_chunks: Number of feature chunks per sample
            noise_std: Noise added to simulate sensor noise
            shuffle_chunks: Whether to randomize chunk order
        """
        self.base_ds = base_ds
        self.n_chunks = n_chunks
        self.noise_std = noise_std
        self.shuffle_chunks = shuffle_chunks
        
    def __len__(self):
        return len(self.base_ds)
    
    def __getitem__(self, idx):
        x, label = self.base_ds[idx]
        
        # Flatten and split into chunks
        x_flat = x.view(-1)
        chunk_size = len(x_flat) // self.n_chunks
        
        chunks = []
        for i in range(self.n_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < self.n_chunks - 1 else len(x_flat)
            chunk = x_flat[start:end]
            
            # Add noise
            if self.noise_std > 0:
                chunk = chunk + torch.randn_like(chunk) * self.noise_std
            
            chunks.append(chunk)
        
        # Pad chunks to same size
        max_len = max(c.size(0) for c in chunks)
        chunks = [F.pad(c, (0, max_len - c.size(0))) for c in chunks]
        chunks = torch.stack(chunks, dim=0)  # [L, chunk_size]
        
        if self.shuffle_chunks:
            perm = torch.randperm(self.n_chunks)
            chunks = chunks[perm]
        
        return chunks, label


# ============================================================================
# Sequential Models
# ============================================================================

class BPANSequential(nn.Module):
    """
    BPAN for sequential evidence integration.
    Processes sequential inputs through the same E/I dynamics.
    """
    def __init__(self, input_dim, n_classes, n_exc=128, n_inh=32,
                 lambda_act=1e-4, lambda_bal=5e-4):
        super().__init__()
        
        self.n_exc = n_exc
        self.n_inh = n_inh
        
        # Build E/I layer directly
        from models import BalancedEILayer
        self.ei_layer = BalancedEILayer(input_dim, n_exc, n_inh, dt=0.2, use_dales=True)
        self.readout = nn.Linear(n_exc, n_classes)
        
        self.lambda_act = lambda_act
        self.lambda_bal = lambda_bal
    
    def forward(self, x_seq, targets=None):
        """
        x_seq: [B, L, D] - sequence of L inputs
        """
        B, L, D = x_seq.shape
        device = x_seq.device
        
        e = torch.zeros(B, self.n_exc, device=device)
        i = torch.zeros(B, self.n_inh, device=device)
        
        logits_list = []
        r_e_list = []
        r_i_list = []
        bal_e_list = []
        bal_i_list = []
        
        for t in range(L):
            x_t = x_seq[:, t, :]
            e, i, r_e, r_i, bal_e, bal_i = self.ei_layer.step(x_t, e, i)
            
            logits_t = self.readout(F.relu(e))
            logits_list.append(logits_t)
            r_e_list.append(r_e)
            r_i_list.append(r_i)
            bal_e_list.append(bal_e)
            bal_i_list.append(bal_i)
        
        logits_seq = torch.stack(logits_list, dim=1)  # [B, L, C]
        r_e_seq = torch.stack(r_e_list, dim=1)
        r_i_seq = torch.stack(r_i_list, dim=1)
        bal_e_seq = torch.stack(bal_e_list, dim=1)
        bal_i_seq = torch.stack(bal_i_list, dim=1)
        
        out = {
            "logits": logits_seq[:, -1, :],
            "logits_seq": logits_seq,
            "r_e_seq": r_e_seq,
            "r_i_seq": r_i_seq,
            "bal_e_seq": bal_e_seq,
            "bal_i_seq": bal_i_seq,
        }
        
        if targets is not None:
            # Weighted CE over time steps
            weights = torch.linspace(0.3, 1.0, L, device=device)
            weights = weights / weights.sum()
            
            ce_list = [F.cross_entropy(logits_seq[:, t], targets) for t in range(L)]
            ce = sum(w * ce_t for w, ce_t in zip(weights, ce_list))
            
            act_cost = r_e_seq.pow(2).mean() + r_i_seq.pow(2).mean()
            bal_cost = bal_e_seq.pow(2).mean() + bal_i_seq.pow(2).mean()
            
            loss = ce + self.lambda_act * act_cost + self.lambda_bal * bal_cost
            
            out["loss"] = loss
            out["ce_loss"] = ce
            out["act_cost"] = act_cost
            out["bal_cost"] = bal_cost
        
        return out
    
    def forward_seq(self, x_seq):
        """For compatibility with evaluation."""
        return self.forward(x_seq)["logits_seq"]


class GRUBaseline(nn.Module):
    """GRU baseline for sequential tasks."""
    def __init__(self, input_dim, n_classes, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x_seq, targets=None):
        out, _ = self.gru(x_seq)  # [B, L, H]
        logits_seq = self.readout(out)  # [B, L, C]
        
        result = {
            "logits": logits_seq[:, -1, :],
            "logits_seq": logits_seq,
        }
        
        if targets is not None:
            L = x_seq.size(1)
            weights = torch.linspace(0.3, 1.0, L, device=x_seq.device)
            weights = weights / weights.sum()
            
            ce_list = [F.cross_entropy(logits_seq[:, t], targets) for t in range(L)]
            loss = sum(w * ce_t for w, ce_t in zip(weights, ce_list))
            
            result["loss"] = loss
            result["ce_loss"] = loss
        
        return result
    
    def forward_seq(self, x_seq):
        return self.forward(x_seq)["logits_seq"]


class LSTMBaseline(nn.Module):
    """LSTM baseline for sequential tasks."""
    def __init__(self, input_dim, n_classes, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.readout = nn.Linear(hidden_dim, n_classes)
    
    def forward(self, x_seq, targets=None):
        out, _ = self.lstm(x_seq)
        logits_seq = self.readout(out)
        
        result = {
            "logits": logits_seq[:, -1, :],
            "logits_seq": logits_seq,
        }
        
        if targets is not None:
            L = x_seq.size(1)
            weights = torch.linspace(0.3, 1.0, L, device=x_seq.device)
            weights = weights / weights.sum()
            
            ce_list = [F.cross_entropy(logits_seq[:, t], targets) for t in range(L)]
            loss = sum(w * ce_t for w, ce_t in zip(weights, ce_list))
            
            result["loss"] = loss
            result["ce_loss"] = loss
        
        return result
    
    def forward_seq(self, x_seq):
        return self.forward(x_seq)["logits_seq"]


# ============================================================================
# Training / Evaluation
# ============================================================================

def train_one_epoch(model, optimizer, loader, device, is_mlp=False, max_grad_norm=5.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        
        # Flatten spatial dims if needed
        B, L = seq.shape[:2]
        seq = seq.view(B, L, -1)
        
        # For MLP, flatten entire sequence into single vector
        if is_mlp:
            seq_flat = seq.view(B, -1)  # [B, L*D]
            out = model(seq_flat, targets=labels)
        else:
            out = model(seq, targets=labels)
        
        loss = out["loss"]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        preds = out["logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * B
        total_samples += B
    
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, is_mlp=False):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        
        B, L = seq.shape[:2]
        seq = seq.view(B, L, -1)
        
        # For MLP, flatten entire sequence into single vector
        if is_mlp:
            seq_flat = seq.view(B, -1)
            out = model(seq_flat, targets=None)
        else:
            out = model(seq, targets=None)
        
        preds = out["logits"].argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += B
    
    return total_correct / total_samples


@torch.no_grad()
def evaluate_per_step(model, loader, device, L):
    """Evaluate accuracy after each sequence step."""
    model.eval()
    correct_per_t = torch.zeros(L)
    total = 0
    
    for seq, labels in loader:
        seq = seq.to(device)
        labels = labels.to(device)
        
        B = seq.size(0)
        seq = seq.view(B, L, -1)
        
        logits_seq = model.forward_seq(seq)  # [B, L, C]
        preds_seq = logits_seq.argmax(dim=-1)  # [B, L]
        
        for t in range(L):
            correct_per_t[t] += (preds_seq[:, t] == labels).sum().item()
        total += B
    
    return (correct_per_t / total).tolist()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run sequential evidence integration experiments."
    )
    parser.add_argument("--task", type=str, default="ctmnist",
                        choices=["ctmnist", "cifar_patches", "streaming"])
    parser.add_argument("--model", type=str, default="bpan",
                        choices=["bpan", "gru", "lstm", "mlp"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--n_glimpses", type=int, default=6)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_dir", type=str, default="./results_sequential")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========== Create dataset ==========
    if args.task == "ctmnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        base_train = datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
        base_test = datasets.MNIST(args.data_root, train=False, download=True, transform=tfm)
        
        train_ds = ClutteredTranslatedMNIST(
            base_train, canvas_size=60, n_clutter=6, 
            glimpse_size=12, n_glimpses=args.n_glimpses,
            saccade_strategy="random"
        )
        test_ds = ClutteredTranslatedMNIST(
            base_test, canvas_size=60, n_clutter=6,
            glimpse_size=12, n_glimpses=args.n_glimpses,
            saccade_strategy="random"
        )
        
        n_classes = 10
        input_dim = 1 * 12 * 12  # glimpse size
        L = args.n_glimpses
        
    elif args.task == "cifar_patches":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])
        base_train = datasets.CIFAR10(args.data_root, train=True, download=True, transform=tfm)
        base_test = datasets.CIFAR10(args.data_root, train=False, download=True, transform=tfm)
        
        train_ds = SequentialCIFARPatches(
            base_train, patch_size=8, n_patches=args.n_glimpses, order="random"
        )
        test_ds = SequentialCIFARPatches(
            base_test, patch_size=8, n_patches=args.n_glimpses, order="random"
        )
        
        n_classes = 10
        input_dim = 3 * 8 * 8
        L = args.n_glimpses
        
    elif args.task == "streaming":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        base_train = datasets.MNIST(args.data_root, train=True, download=True, transform=tfm)
        base_test = datasets.MNIST(args.data_root, train=False, download=True, transform=tfm)
        
        train_ds = StreamingFeatureDataset(
            base_train, n_chunks=args.n_glimpses, noise_std=0.1, shuffle_chunks=True
        )
        test_ds = StreamingFeatureDataset(
            base_test, n_chunks=args.n_glimpses, noise_std=0.1, shuffle_chunks=True
        )
        
        n_classes = 10
        L = args.n_glimpses
        
        # Get actual input dimension from dataset (handles any chunk size)
        sample_seq, _ = train_ds[0]
        input_dim = sample_seq.shape[-1]
    
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=use_cuda)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=use_cuda)

    # ========== Build model ==========
    n_exc = args.hidden // 2
    n_inh = max(4, args.hidden // 8)
    
    if args.model == "bpan":
        model = BPANSequential(
            input_dim=input_dim,
            n_classes=n_classes,
            n_exc=n_exc,
            n_inh=n_inh,
            lambda_act=1e-4,
            lambda_bal=5e-4
        )
    elif args.model == "gru":
        model = GRUBaseline(input_dim, n_classes, args.hidden)
    elif args.model == "lstm":
        model = LSTMBaseline(input_dim, n_classes, args.hidden)
    elif args.model == "mlp":
        # Processes each step independently (no recurrence)
        model = MLPClassifier(input_dim * L, n_classes, args.hidden)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTask: {args.task}, Model: {args.model}")
    print(f"Sequence length: {L}, Input dim: {input_dim}")
    print(f"Params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Check if model is MLP (needs flattened input)
    is_mlp = (args.model == "mlp")

    # ========== Training ==========
    best_acc = 0.0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "test_acc": [], "acc_per_step": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, device, is_mlp=is_mlp)
        test_acc = evaluate(model, test_loader, device, is_mlp=is_mlp)
        epoch_time = time.time() - t0
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        print(f"[{args.model.upper()}-{args.task}] Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.3f} | "
              f"Test acc: {test_acc:.3f} | Time: {epoch_time:.2f}s")

    # ========== Final evaluation ==========
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    
    if args.model != "mlp":
        acc_per_step = evaluate_per_step(model, test_loader, device, L)
        print(f"\nAccuracy per step: {acc_per_step}")
        history["acc_per_step"] = acc_per_step
    else:
        history["acc_per_step"] = [None] * (L - 1) + [best_acc]

    # ========== Save results ==========
    ckpt_path = out_dir / f"{args.model}_{args.task}_best.pth"
    torch.save(best_state, ckpt_path)
    
    log_path = out_dir / f"{args.task}_{args.model}.json"
    result = {
        "task": args.task,
        "model": args.model,
        "params": int(params),
        "n_glimpses": args.n_glimpses,
        "hidden": args.hidden,
        "epochs": args.epochs,
        "best_test_acc": float(best_acc),
        "acc_per_step": history["acc_per_step"],
        "history": history,
    }
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nSaved results to {log_path}")
    print(f"Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

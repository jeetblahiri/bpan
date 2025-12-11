# run_experiment.py
import time
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ei_bpan.models_old import (
    MLPClassifier,
    BPANClassifier,
    ConvBackboneCIFAR,
    ConvWrapper,
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------- Dataset setup ----------

def get_datasets(name, data_root):
    """
    Returns train_dataset, test_dataset, n_classes, in_shape for a dataset name.
    """
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_ds = datasets.MNIST(root=data_root, train=True, download=True,
                                  transform=transform)
        test_ds = datasets.MNIST(root=data_root, train=False, download=True,
                                 transform=transform)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_ds = datasets.FashionMNIST(root=data_root, train=True, download=True,
                                         transform=transform)
        test_ds = datasets.FashionMNIST(root=data_root, train=False, download=True,
                                        transform=transform)
        n_classes = 10
        in_shape = (1, 28, 28)

    elif name == "emnist_letters":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        train_ds = datasets.EMNIST(root=data_root, split="letters",
                                   train=True, download=True,
                                   transform=transform)
        test_ds = datasets.EMNIST(root=data_root, split="letters",
                                  train=False, download=True,
                                  transform=transform)
        n_classes = 26
        in_shape = (1, 28, 28)

    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_ds = datasets.CIFAR10(root=data_root, train=True, download=True,
                                    transform=transform_train)
        test_ds = datasets.CIFAR10(root=data_root, train=False, download=True,
                                   transform=transform_test)
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
        indices = torch.randperm(n)[:k].tolist()
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, n_classes, in_shape


# ---------- Noise / occlusion for eval ----------

def add_noise(x, noise_type, noise_level, dataset_name):
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
        if dataset_name in ["mnist", "fashion_mnist", "emnist_letters"]:
            _, C, H, W = x.shape
        else:
            _, C, H, W = x.shape

        side = int(noise_level * min(H, W))
        side = max(1, side)
        y0 = (H - side) // 2
        x0 = (W - side) // 2

        x_occ = x.clone()
        x_occ[:, :, y0:y0+side, x0:x0+side] = 0.0
        return x_occ

    raise ValueError(f"Unknown noise type: {noise_type}")


# ---------- Training / evaluation loops ----------

def train_one_epoch(model, optimizer, train_loader, device,
                    dataset_name, max_grad_norm=None, is_bpan=False):
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

        if dataset_name != "cifar10":
            x = x.view(x.size(0), -1)

        if is_bpan:
            out = model(x, targets=y)
            loss = out["loss"]
            ce_loss = out["ce_loss"]
        else:
            out = model(x, targets=y)
            loss = out["loss"]
            ce_loss = loss

        optimizer.zero_grad()
        loss.backward()

        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size

        preds = out["logits"].argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_ce = total_ce / total_samples
    acc = total_correct / total_samples
    return avg_loss, avg_ce, acc


@torch.no_grad()
def evaluate(model, data_loader, device, dataset_name,
             is_bpan=False, noise_type="none", noise_level=0.0):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)

        if dataset_name == "emnist_letters":
            y = y - 1

        if noise_type != "none":
            x = add_noise(x, noise_type, noise_level, dataset_name)

        if dataset_name != "cifar10":
            x = x.view(x.size(0), -1)

        if is_bpan:
            out = model(x, targets=y)
            loss = out["loss"]
        else:
            out = model(x, targets=y)
            loss = out["loss"]

        preds = out["logits"].argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_samples += x.size(0)
        total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


# ---------- Main experiment ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "fashion_mnist", "cifar10", "emnist_letters"])
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "bpan"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--bpan_T", type=int, default=6)
    parser.add_argument("--bpan_lambda_act", type=float, default=1e-4)
    parser.add_argument("--bpan_lambda_bal", type=float, default=5e-4)
    parser.add_argument("--frac_train", type=float, default=1.0,
                        help="Fraction of training data to use (small-data regime)")
    parser.add_argument("--noise_type", type=str, default="none",
                        choices=["none", "gaussian", "sp", "occlusion"])
    parser.add_argument("--noise_level", type=float, default=0.0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out_path", type=str, default="./results/exp_results.jsonl")
    # Ablations
    parser.add_argument("--no_dales", action="store_true",
                        help="Disable Dale's law in BPAN")
    parser.add_argument("--no_balance_reg", action="store_true",
                        help="Disable E/I balance regularizer")
    parser.add_argument("--no_activity_reg", action="store_true",
                        help="Disable activity regularizer")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader, n_classes, in_shape = get_dataloaders(
        args.dataset,
        args.data_root,
        args.batch_size,
        args.frac_train,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    # Model
    if args.dataset == "cifar10":
        backbone = ConvBackboneCIFAR()
        if args.model == "mlp":
            head = MLPClassifier(
                input_dim=backbone.out_dim,
                n_classes=n_classes,
                hidden=args.hidden,
            )
            model = ConvWrapper(backbone, head)
            is_bpan = False
        else:
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
                use_dales=not args.no_dales,
                use_act_reg=not args.no_activity_reg,
                use_bal_reg=not args.no_balance_reg,
            )
            model = ConvWrapper(backbone, head)
            is_bpan = True
    else:
        input_dim = in_shape[0] * in_shape[1] * in_shape[2]
        if args.model == "mlp":
            model = MLPClassifier(
                input_dim=input_dim,
                n_classes=n_classes,
                hidden=args.hidden,
            )
            is_bpan = False
        else:
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
                use_dales=not args.no_dales,
                use_act_reg=not args.no_activity_reg,
                use_bal_reg=not args.no_balance_reg,
            )
            is_bpan = True

    model.to(device)

    params = count_params(model)
    print(f"\nModel: {args.model} on {args.dataset}")
    print(f"Train fraction: {args.frac_train}")
    print(f"Params: {params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_times = []
    test_accs = []
    best_test_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_ce, train_acc = train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            dataset_name=args.dataset,
            max_grad_norm=args.max_grad_norm,
            is_bpan=is_bpan,
        )
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            device,
            dataset_name=args.dataset,
            is_bpan=is_bpan,
            noise_type=args.noise_type,
            noise_level=args.noise_level,
        )
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"[{args.model.upper()}-{args.dataset}] "
              f"Epoch {epoch:02d} | "
              f"Train loss: {train_loss:.4f} (CE {train_ce:.4f}) | "
              f"Train acc: {train_acc:.3f} | "
              f"Test acc: {test_acc:.3f} | "
              f"Epoch time: {epoch_time:.2f}s")

    avg_epoch_time = sum(epoch_times) / len(epoch_times)

    print("\n=== Summary ===")
    print(f"{args.model.upper()} on {args.dataset} | "
          f"params: {params:,}, "
          f"best test acc: {best_test_acc:.3f}, "
          f"avg epoch time: {avg_epoch_time:.2f}s")

    # Always save best BPAN checkpoint to a deterministic path
    if args.model == "bpan" and best_state is not None:
        ckpt_path = Path(f"./results/bpan_{args.dataset}_hidden{args.hidden}_best.pth")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt_path)
        print(f"Saved best BPAN checkpoint to {ckpt_path}")

    # Log JSON line
    result_record = {
        "dataset": args.dataset,
        "model": args.model,
        "params": int(params),
        "best_test_acc": float(best_test_acc),
        "avg_epoch_time": float(avg_epoch_time),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden": args.hidden,
        "bpan_T": args.bpan_T if args.model == "bpan" else None,
        "bpan_lambda_act": args.bpan_lambda_act if args.model == "bpan" else None,
        "bpan_lambda_bal": args.bpan_lambda_bal if args.model == "bpan" else None,
        "frac_train": args.frac_train,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "no_dales": args.no_dales,
        "no_balance_reg": args.no_balance_reg,
        "no_activity_reg": args.no_activity_reg,
        "seed": args.seed,
        "device": str(device),
    }

    with out_path.open("a") as f:
        f.write(json.dumps(result_record) + "\n")


if __name__ == "__main__":
    main()

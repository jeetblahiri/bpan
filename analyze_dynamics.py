# analyze_dynamics.py
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ei_bpan.models_old import BPANClassifier


# ---------------------- Dataset + noise helpers ---------------------- #

def get_test_batch(dataset_name, data_root, batch_size, device):
    """
    Load a single batch from the test set for a 28x28 grayscale dataset:
    mnist, fashion_mnist, emnist_letters.

    Returns:
        x_img:   [B, 1, 28, 28] (on device)
        x_flat:  [B, 784]       (on device)
        labels:  [B]            (on device, 0-based)
    """
    if dataset_name == "mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        ds = datasets.MNIST(root=data_root, train=False, download=True,
                            transform=tfm)

    elif dataset_name == "fashion_mnist":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        ds = datasets.FashionMNIST(root=data_root, train=False, download=True,
                                   transform=tfm)

    elif dataset_name == "emnist_letters":
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1736,), (0.3317,))
        ])
        ds = datasets.EMNIST(root=data_root, split="letters",
                             train=False, download=True,
                             transform=tfm)
    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported in analyze_dynamics.py "
            f"(expected one of: mnist, fashion_mnist, emnist_letters)."
        )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    x, y = next(iter(loader))        # one batch
    x = x.to(device)                 # [B, 1, 28, 28]
    y = y.to(device)

    if dataset_name == "emnist_letters":
        # EMNIST letters labels are 1..26; shift to 0..25
        y = y - 1

    x_flat = x.view(x.size(0), -1)   # [B, 784]
    return x, x_flat, y


def get_num_classes(dataset_name):
    if dataset_name in ["mnist", "fashion_mnist"]:
        return 10
    elif dataset_name == "emnist_letters":
        return 26
    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported for class count here."
        )


def add_noise(x_img, noise_type, noise_level, dataset_name):
    """
    x_img: [B, 1, 28, 28] (for these datasets)
    Returns noisy version on the same device.
    """
    if noise_type == "none":
        return x_img

    if noise_type == "gaussian":
        # x_img is already normalized
        noise = torch.randn_like(x_img) * noise_level
        return x_img + noise

    if noise_type == "sp":
        # salt-and-pepper: probability = noise_level
        p = noise_level
        rand = torch.rand_like(x_img)
        x_noisy = x_img.clone()
        x_noisy[rand < (p / 2)] = x_img.min()
        x_noisy[(rand >= (p / 2)) & (rand < p)] = x_img.max()
        return x_noisy

    if noise_type == "occlusion":
        # central square occlusion; noise_level = side / 28
        _, _, H, W = x_img.shape
        side = int(noise_level * min(H, W))
        side = max(1, side)
        y0 = (H - side) // 2
        x0 = (W - side) // 2

        x_occ = x_img.clone()
        x_occ[:, :, y0:y0+side, x0:x0+side] = 0.0
        return x_occ

    raise ValueError(f"Unknown noise_type: {noise_type}")


# ---------------------- Main analysis script ---------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Analyze BPAN dynamics (E/I balance, attractors) "
                    "on a batch of MNIST-like data."
    )
    parser.add_argument(
        "--ckpt", type=str, required=True,
        help="Path to BPAN checkpoint (.pth) saved by run_experiment.py "
             "(e.g. results/bpan_mnist_hidden256_best.pth)."
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "fashion_mnist", "emnist_letters"],
        help="Dataset used for this BPAN (must be 1x28x28 grayscale)."
    )
    parser.add_argument(
        "--data_root", type=str, default="./data",
        help="Root directory for datasets."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64,
        help="Number of test samples to analyze."
    )
    parser.add_argument(
        "--hidden", type=int, default=256,
        help="Hidden size used when training the BPAN head. "
             "We reconstruct n_exc = hidden/2, n_inh = hidden/8."
    )
    parser.add_argument(
        "--bpan_T", type=int, default=6,
        help="Number of recurrent steps T used when training."
    )
    parser.add_argument(
        "--lambda_act", type=float, default=1e-4,
        help="Activity regularization lambda used during training (for completeness)."
    )
    parser.add_argument(
        "--lambda_bal", type=float, default=5e-4,
        help="Balance regularization lambda used during training (for completeness)."
    )
    parser.add_argument(
        "--noise_type", type=str, default="none",
        choices=["none", "gaussian", "sp", "occlusion"],
        help="If not 'none', also run dynamics on a noisy/occluded copy of the batch."
    )
    parser.add_argument(
        "--noise_level", type=float, default=0.0,
        help="Strength of noise/occlusion (e.g., sigma for gaussian; prob for sp; "
             "fraction of image side for occlusion)."
    )
    parser.add_argument(
        "--no_cuda", action="store_true",
        help="Force CPU even if CUDA is available."
    )
    parser.add_argument(
        "--out_path", type=str, default="results/bpan_dynamics_stats.pt",
        help="Where to save the collected dynamics tensors."
    )
    args = parser.parse_args()

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # ---------------- Build model to match training setup ---------------- #

    input_dim = 28 * 28
    n_classes = get_num_classes(args.dataset)

    n_exc = args.hidden // 2
    n_inh = max(4, args.hidden // 8)

    model = BPANClassifier(
        input_dim=input_dim,
        n_classes=n_classes,
        n_exc=n_exc,
        n_inh=n_inh,
        T=args.bpan_T,
        lambda_act=args.lambda_act,
        lambda_bal=args.lambda_bal,
        use_dales=True,
        use_act_reg=True,
        use_bal_reg=True,
    ).to(device)

    # Load checkpoint
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded BPAN checkpoint from {ckpt_path}")

    # ---------------- Get a clean batch ---------------- #

    x_img_clean, x_flat_clean, labels = get_test_batch(
        args.dataset, args.data_root, args.batch_size, device
    )

    # ---------------- Optionally build noisy/occluded copy ---------------- #

    has_noisy = args.noise_type != "none"
    if has_noisy:
        x_img_noisy = add_noise(
            x_img_clean, args.noise_type, args.noise_level, args.dataset
        )
        x_flat_noisy = x_img_noisy.view(x_img_noisy.size(0), -1)
        print(f"Generated noisy batch: type={args.noise_type}, level={args.noise_level}")
    else:
        x_img_noisy = None
        x_flat_noisy = None

    # ---------------- Run BPAN dynamics (clean) ---------------- #

    with torch.no_grad():
        stats_clean = model.forward_with_stats(x_flat_clean)
        r_e_clean = stats_clean["r_e_seq"]        # [B, T, n_exc]
        r_i_clean = stats_clean["r_i_seq"]        # [B, T, n_inh]
        bal_e_clean = stats_clean["bal_e_seq"]    # [B, T, n_exc]
        bal_i_clean = stats_clean["bal_i_seq"]    # [B, T, n_inh]
        logits_clean = stats_clean["logits_seq"]  # [B, T, n_classes]

    # Accuracy per time step (clean)
    preds_clean = logits_clean.argmax(dim=-1)     # [B, T]
    correct_clean = (preds_clean == labels.unsqueeze(1)).float()
    acc_per_t_clean = correct_clean.mean(dim=0)   # [T]

    print("Accuracy per time step (clean):", acc_per_t_clean.cpu().numpy())

    # E/I balance magnitude over time (clean)
    bal_e_mag_clean = bal_e_clean.abs().mean(dim=(0, 2))  # [T]
    bal_i_mag_clean = bal_i_clean.abs().mean(dim=(0, 2))  # [T]
    print("Mean |balance_e| per t (clean):", bal_e_mag_clean.cpu().numpy())
    print("Mean |balance_i| per t (clean):", bal_i_mag_clean.cpu().numpy())

    # ---------------- Run BPAN dynamics (noisy), if requested ---------------- #

    r_e_noisy = None
    r_i_noisy = None
    bal_e_noisy = None
    bal_i_noisy = None
    logits_noisy = None
    acc_per_t_noisy = None
    final_state_cosine = None

    if has_noisy:
        with torch.no_grad():
            stats_noisy = model.forward_with_stats(x_flat_noisy)
            r_e_noisy = stats_noisy["r_e_seq"]       # [B, T, n_exc]
            r_i_noisy = stats_noisy["r_i_seq"]       # [B, T, n_inh]
            bal_e_noisy = stats_noisy["bal_e_seq"]   # [B, T, n_exc]
            bal_i_noisy = stats_noisy["bal_i_seq"]   # [B, T, n_inh]
            logits_noisy = stats_noisy["logits_seq"] # [B, T, n_classes]

        preds_noisy = logits_noisy.argmax(dim=-1)
        correct_noisy = (preds_noisy == labels.unsqueeze(1)).float()
        acc_per_t_noisy = correct_noisy.mean(dim=0)
        print("Accuracy per time step (noisy):", acc_per_t_noisy.cpu().numpy())

        # Attractor-ish measure: cosine similarity between clean vs noisy final excitatory state
        r_e_final_clean = r_e_clean[:, -1, :]   # [B, n_exc]
        r_e_final_noisy = r_e_noisy[:, -1, :]   # [B, n_exc]

        # Normalize and compute cosine sim per sample
        clean_norm = r_e_final_clean / (r_e_final_clean.norm(dim=1, keepdim=True) + 1e-8)
        noisy_norm = r_e_final_noisy / (r_e_final_noisy.norm(dim=1, keepdim=True) + 1e-8)
        cos_sim = (clean_norm * noisy_norm).sum(dim=1)  # [B]
        final_state_cosine = cos_sim.mean()
        print("Mean cosine similarity of final excitatory state (clean vs noisy):",
              float(final_state_cosine.cpu()))

    # ---------------- Save everything for plotting / deeper analysis ---------------- #

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "dataset": args.dataset,
        "ckpt": str(ckpt_path),
        "hidden": args.hidden,
        "bpan_T": args.bpan_T,
        "lambda_act": args.lambda_act,
        "lambda_bal": args.lambda_bal,
        "noise_type": args.noise_type,
        "noise_level": args.noise_level,
        "labels": labels.cpu(),
        # Clean trajectories
        "r_e_clean": r_e_clean.cpu(),
        "r_i_clean": r_i_clean.cpu(),
        "bal_e_clean": bal_e_clean.cpu(),
        "bal_i_clean": bal_i_clean.cpu(),
        "logits_clean": logits_clean.cpu(),
        "acc_per_t_clean": acc_per_t_clean.cpu(),
        "bal_e_mag_clean": bal_e_mag_clean.cpu(),
        "bal_i_mag_clean": bal_i_mag_clean.cpu(),
    }

    if has_noisy:
        save_dict.update({
            "r_e_noisy": r_e_noisy.cpu(),
            "r_i_noisy": r_i_noisy.cpu(),
            "bal_e_noisy": bal_e_noisy.cpu(),
            "bal_i_noisy": bal_i_noisy.cpu(),
            "logits_noisy": logits_noisy.cpu(),
            "acc_per_t_noisy": acc_per_t_noisy.cpu(),
            "final_state_cosine": final_state_cosine.cpu(),
        })

    torch.save(save_dict, out_path)
    print(f"Saved dynamics stats (clean{'+noisy' if has_noisy else ''}) to {out_path}")


if __name__ == "__main__":
    main()

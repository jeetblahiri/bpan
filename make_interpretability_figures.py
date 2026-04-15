"""
Produces interpretability figures for the BPAN manuscript:
  1. pca_trajectories_mnist.png   -- PCA of excitatory trajectories over T=6
                                     recurrent steps; curves colored by digit
                                     class. Re-creates the figure that was in
                                     the original manuscript.
  2. pca_final_states_mnist.png   -- PCA of final excitatory states; scatter
                                     colored by class. Shows class-conditioned
                                     clustering after the network has settled.
  3. confidence_evolution_mnist.png -- Per-step max-softmax confidence for a
                                     sample of test examples, with the 0.9
                                     halting threshold and mean curves.
  4. attractor_occlusion_mnist.png -- PCA trajectories for the same examples
                                     presented clean vs. 50% centrally occluded.
                                     Illustrates attractor-like pattern
                                     completion: perturbed inputs converge into
                                     the same class-specific regions.

All figures are written straight into bpan_figures/ so the manuscript's
\graphicspath picks them up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from models import BPANClassifier

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
ROOT = Path("/Users/jeetblahiri/Documents/PhD/Ongoing works/Revised_Bpan")
CKPT = ROOT / "results" / "bpan_mnist_hidden256_best.pth"
FIG_DIR = ROOT / "bpan_figures"
DATA_ROOT = ROOT / "data"
BATCH_SIZE = 1024          # enough points for a stable PCA fit
N_PER_CLASS_TRAJ = 4       # curves per class in trajectory plots
CONFIDENCE_EXAMPLES = 120  # examples overlaid in the confidence plot
OCCLUSION_EXAMPLES = 3     # examples per class for the occlusion figure
OCCLUSION_FRAC = 0.5       # central square side, as fraction of image side
HIDDEN = 256
T_STEPS = 6
SEED = 0

DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# -------------------------------------------------------------------------
# Data and model loading
# -------------------------------------------------------------------------
def load_mnist_batch(batch_size: int):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(root=str(DATA_ROOT), train=False, download=True,
                        transform=tfm)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    x, y = next(iter(loader))
    return x.to(DEVICE), y.to(DEVICE)


def load_model():
    n_exc = HIDDEN // 2
    n_inh = max(4, HIDDEN // 8)
    model = BPANClassifier(
        input_dim=28 * 28,
        n_classes=10,
        n_exc=n_exc,
        n_inh=n_inh,
        T=T_STEPS,
    ).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def occlude_centre(x_img: torch.Tensor, frac: float) -> torch.Tensor:
    """Zero out a central square of side `frac * 28`."""
    _, _, H, W = x_img.shape
    side = max(1, int(frac * min(H, W)))
    y0 = (H - side) // 2
    x0 = (W - side) // 2
    out = x_img.clone()
    out[:, :, y0:y0 + side, x0:x0 + side] = 0.0
    return out


# -------------------------------------------------------------------------
# Figure 1: PCA trajectories
# -------------------------------------------------------------------------
def fig_pca_trajectories(r_e_seq: np.ndarray, labels: np.ndarray, pca: PCA):
    """r_e_seq: [B, T, n_exc]; labels: [B]."""
    B, T, _ = r_e_seq.shape
    rng = np.random.default_rng(SEED)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    for cls in range(10):
        idx_cls = np.where(labels == cls)[0]
        rng.shuffle(idx_cls)
        chosen = idx_cls[:N_PER_CLASS_TRAJ]
        color = cmap(cls)
        for i in chosen:
            traj = pca.transform(r_e_seq[i])        # [T, 2]
            ax.plot(traj[:, 0], traj[:, 1],
                    color=color, alpha=0.75, linewidth=1.3, zorder=2)
            ax.scatter(traj[0, 0], traj[0, 1],
                       s=22, marker="o", facecolor="white",
                       edgecolor=color, linewidths=1.2, zorder=3)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       s=38, marker="*", color=color, zorder=4)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "Excitatory trajectories over $T{=}6$ steps (PCA), MNIST",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    class_handles = [
        Line2D([0], [0], color=cmap(c), lw=2.0, label=str(c))
        for c in range(10)
    ]
    marker_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor="white", markeredgecolor="gray",
               markersize=6, label="$t{=}0$"),
        Line2D([0], [0], marker="*", linestyle="",
               color="gray", markersize=9, label="$t{=}T$"),
    ]
    leg1 = ax.legend(handles=class_handles, title="Class",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=marker_handles, loc="upper left",
              bbox_to_anchor=(1.02, 0.35), fontsize=8, frameon=False)

    fig.tight_layout()
    out = FIG_DIR / "pca_trajectories_mnist.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[1/4] wrote {out}")


# -------------------------------------------------------------------------
# Figure 2: PCA of final states
# -------------------------------------------------------------------------
def fig_pca_final_states(r_e_final: np.ndarray, labels: np.ndarray, pca: PCA):
    z = pca.transform(r_e_final)
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    scatter = ax.scatter(z[:, 0], z[:, 1],
                         c=labels, cmap="tab10",
                         s=10, alpha=0.75, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Final excitatory states at $t{=}T$ (PCA), MNIST",
                 fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(10))
    cbar.set_label("Digit class")
    fig.tight_layout()
    out = FIG_DIR / "pca_final_states_mnist.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[2/4] wrote {out}")


# -------------------------------------------------------------------------
# Figure 3: Confidence evolution
# -------------------------------------------------------------------------
def fig_confidence_evolution(logits_seq: np.ndarray, labels: np.ndarray):
    """logits_seq: [B, T, C]; labels: [B]."""
    B, T, C = logits_seq.shape
    probs = _softmax_np(logits_seq, axis=-1)    # [B, T, C]
    preds = probs.argmax(axis=-1)               # [B, T]
    confs = probs.max(axis=-1)                  # [B, T]
    final_correct = preds[:, -1] == labels

    rng = np.random.default_rng(SEED)
    chosen = rng.choice(B, size=min(CONFIDENCE_EXAMPLES, B), replace=False)
    t_axis = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i in chosen:
        color = "#2a7fb0" if final_correct[i] else "#c44e52"
        ax.plot(t_axis, confs[i], color=color, alpha=0.25, linewidth=0.9)

    mean_correct = confs[final_correct].mean(axis=0)
    mean_wrong = confs[~final_correct].mean(axis=0)
    ax.plot(t_axis, mean_correct, color="#1f3d5c", linewidth=2.4,
            label=f"mean (correct, n={int(final_correct.sum())})")
    if (~final_correct).any():
        ax.plot(t_axis, mean_wrong, color="#8b1a1a", linewidth=2.4,
                label=f"mean (incorrect, n={int((~final_correct).sum())})")

    ax.axhline(0.9, color="k", linestyle="--", linewidth=1.0,
               label=r"halting threshold $\theta{=}0.9$")
    ax.set_xlabel("Recurrent step $t$")
    ax.set_ylabel("Max-softmax confidence")
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(t_axis)
    ax.set_title("Confidence evolution over recurrent steps, MNIST",
                 fontsize=11)
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out = FIG_DIR / "confidence_evolution_mnist.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[3/4] wrote {out}")


# -------------------------------------------------------------------------
# Figure 4: Attractor behaviour under occlusion
# -------------------------------------------------------------------------
def fig_attractor_occlusion(model, x_img, labels, pca: PCA):
    rng = np.random.default_rng(SEED + 1)
    y_np = labels.detach().cpu().numpy()
    classes = [0, 3, 7]  # three well-separated digits for a clean figure
    chosen = []
    for c in classes:
        idx_c = np.where(y_np == c)[0]
        rng.shuffle(idx_c)
        chosen.extend(idx_c[:OCCLUSION_EXAMPLES])
    chosen = np.array(chosen)

    x_sel = x_img[chosen]
    y_sel = y_np[chosen]
    x_occ = occlude_centre(x_sel, OCCLUSION_FRAC)

    with torch.no_grad():
        clean = model.forward_with_stats(x_sel.view(x_sel.size(0), -1))
        noisy = model.forward_with_stats(x_occ.view(x_occ.size(0), -1))
    r_clean = clean["r_e_seq"].detach().cpu().numpy()  # [N, T, n_exc]
    r_noisy = noisy["r_e_seq"].detach().cpu().numpy()

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for k, idx in enumerate(chosen):
        cls = y_sel[k]
        color = cmap(cls)
        z_c = pca.transform(r_clean[k])
        z_n = pca.transform(r_noisy[k])
        ax.plot(z_c[:, 0], z_c[:, 1], color=color, linewidth=1.5, alpha=0.95)
        ax.plot(z_n[:, 0], z_n[:, 1], color=color, linewidth=1.5, alpha=0.95,
                linestyle="--")
        ax.scatter(z_c[-1, 0], z_c[-1, 1], s=48, marker="*",
                   color=color, zorder=4, edgecolors="k", linewidths=0.5)
        ax.scatter(z_n[-1, 0], z_n[-1, 1], s=36, marker="X",
                   color=color, zorder=4, edgecolors="k", linewidths=0.5)

    class_handles = [
        Line2D([0], [0], color=cmap(c), lw=2.0, label=f"class {c}")
        for c in classes
    ]
    style_handles = [
        Line2D([0], [0], color="gray", lw=2.0, label="clean"),
        Line2D([0], [0], color="gray", lw=2.0, linestyle="--",
               label="occluded (50% centre)"),
        Line2D([0], [0], marker="*", linestyle="", color="gray",
               markersize=9, markeredgecolor="k", label="clean endpoint"),
        Line2D([0], [0], marker="X", linestyle="", color="gray",
               markersize=8, markeredgecolor="k", label="occluded endpoint"),
    ]
    leg1 = ax.legend(handles=class_handles, title="Class",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     fontsize=8, title_fontsize=9, frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=style_handles, loc="upper left",
              bbox_to_anchor=(1.02, 0.55), fontsize=8, frameon=False)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "Clean vs. occluded trajectories converge to class-specific regions",
        fontsize=11,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "attractor_occlusion_mnist.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[4/4] wrote {out}")


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Device: {DEVICE}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    x_img, labels = load_mnist_batch(BATCH_SIZE)
    x_flat = x_img.view(x_img.size(0), -1)

    with torch.no_grad():
        stats = model.forward_with_stats(x_flat)

    r_e_seq = stats["r_e_seq"].detach().cpu().numpy()    # [B, T, n_exc]
    logits_seq = stats["logits_seq"].detach().cpu().numpy()  # [B, T, C]
    y = labels.detach().cpu().numpy()

    # Fit a single PCA on all (B*T, n_exc) excitatory states so the four
    # figures live in a consistent projection.
    B, T, n_exc = r_e_seq.shape
    pca = PCA(n_components=2)
    pca.fit(r_e_seq.reshape(B * T, n_exc))
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    fig_pca_trajectories(r_e_seq, y, pca)
    fig_pca_final_states(r_e_seq[:, -1, :], y, pca)
    fig_confidence_evolution(logits_seq, y)
    fig_attractor_occlusion(model, x_img, labels, pca)


if __name__ == "__main__":
    main()

# analyze_pca_tsne.py
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def load_stats(path):
    path = Path(path)
    data = torch.load(path, map_location="cpu")
    print(f"Loaded {path}")
    print("Keys:", list(data.keys()))
    return data


def to_np(x):
    return x.detach().cpu().numpy()


# ---------------- PCA helpers ---------------- #

def pca_on_final_states(data, save_prefix=None, max_points=2000):
    """
    PCA on final excitatory states (clean, and noisy if present).
    """
    r_e_clean = data["r_e_clean"]      # [B, T, n_exc]
    labels = data["labels"]            # [B]
    B, T, n_exc = r_e_clean.shape

    # Final time-step excitatory state (clean)
    r_final_clean = r_e_clean[:, -1, :]    # [B, n_exc]

    has_noisy = "r_e_noisy" in data
    if has_noisy:
        r_e_noisy = data["r_e_noisy"]
        r_final_noisy = r_e_noisy[:, -1, :]
        print("Noisy trajectories present; will include in PCA.")

    # Subsample if too many points for plotting
    idx = np.arange(B)
    if B > max_points:
        np.random.shuffle(idx)
        idx = idx[:max_points]

    X_clean = to_np(r_final_clean[idx])
    y = to_np(labels[idx])

    if has_noisy:
        X_noisy = to_np(r_final_noisy[idx])
        X_all = np.concatenate([X_clean, X_noisy], axis=0)
    else:
        X_all = X_clean

    print("Running PCA on final excitatory states...")
    pca = PCA(n_components=2)
    Z_all = pca.fit_transform(X_all)  # [B (or 2B), 2]

    if has_noisy:
        Z_clean = Z_all[: len(X_clean)]
        Z_noisy = Z_all[len(X_clean):]
    else:
        Z_clean = Z_all
        Z_noisy = None

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # ---------- Plot: clean states in PC1-PC2 ---------- #
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z_clean[:, 0], Z_clean[:, 1],
                          c=y, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, label="Digit label")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clean final excitatory states (PCA)")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_pca_clean_states.png", dpi=200)
    else:
        plt.show()
    plt.close()

    # ---------- Plot: if noisy, overlay clean vs noisy ---------- #
    if has_noisy:
        plt.figure(figsize=(6, 5))
        plt.scatter(Z_clean[:, 0], Z_clean[:, 1],
                    c="blue", s=6, alpha=0.5, label="clean")
        plt.scatter(Z_noisy[:, 0], Z_noisy[:, 1],
                    c="red", s=6, alpha=0.5, label="noisy")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Clean vs noisy final excitatory states (PCA)")
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_pca_clean_vs_noisy.png", dpi=200)
        else:
            plt.show()
        plt.close()

    return pca, Z_clean, Z_noisy, y


def plot_attractor_trajectories(data, pca=None, save_prefix=None,
                                n_per_class=3):
    """
    Plot trajectories in PC1-PC2 for a subset of samples (clean only).
    """
    r_e_clean = data["r_e_clean"]   # [B, T, n_exc]
    labels = data["labels"]         # [B]
    B, T, n_exc = r_e_clean.shape

    X = to_np(r_e_clean.reshape(B * T, n_exc))  # [B*T, n_exc]
    y = to_np(labels)

    # If no PCA provided, fit one on all time steps
    if pca is None:
        print("Fitting PCA on all clean excitatory states for trajectories...")
        pca = PCA(n_components=2)
        pca.fit(X)

    # Choose a few samples per class
    unique_labels = np.unique(y)
    chosen_indices = []
    rng = np.random.default_rng(0)

    for c in unique_labels:
        idx_c = np.where(y == c)[0]
        if len(idx_c) == 0:
            continue
        rng.shuffle(idx_c)
        chosen_indices.extend(idx_c[: n_per_class])

    chosen_indices = np.array(chosen_indices)
    print(f"Plotting trajectories for {len(chosen_indices)} samples.")

    # Project trajectories
    plt.figure(figsize=(8, 6))
    for idx in chosen_indices:
        traj = to_np(r_e_clean[idx])   # [T, n_exc]
        Z_traj = pca.transform(traj)   # [T, 2]
        label = int(y[idx])

        plt.plot(Z_traj[:, 0], Z_traj[:, 1], "-o", markersize=3, alpha=0.7,
                 label=f"class {label}")

    # Avoid insane legend; show unique once
    handles, labels_ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels_, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=3)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("BPAN clean trajectories in PC space (attractor-like behavior)")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_pca_trajectories.png", dpi=200)
    else:
        plt.show()
    plt.close()


# ---------------- t-SNE helpers ---------------- #

def tsne_on_final_states(data, save_prefix=None, max_points=2000, perplexity=30.0):
    """
    t-SNE on final excitatory states for visualization.
    """
    r_e_clean = data["r_e_clean"]      # [B, T, n_exc]
    labels = data["labels"]            # [B]
    B, T, n_exc = r_e_clean.shape

    r_final_clean = r_e_clean[:, -1, :]    # [B, n_exc]
    has_noisy = "r_e_noisy" in data
    if has_noisy:
        r_e_noisy = data["r_e_noisy"]
        r_final_noisy = r_e_noisy[:, -1, :]
        print("Noisy trajectories present; including in t-SNE.")

    idx = np.arange(B)
    if B > max_points:
        np.random.shuffle(idx)
        idx = idx[:max_points]

    X_clean = to_np(r_final_clean[idx])
    y = to_np(labels[idx])

    if has_noisy:
        X_noisy = to_np(r_final_noisy[idx])
        X_all = np.concatenate([X_clean, X_noisy], axis=0)
    else:
        X_all = X_clean

    print("Running t-SNE on final excitatory states (this can take a while)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto")
    Z_all = tsne.fit_transform(X_all)  # [N, 2]

    if has_noisy:
        Z_clean = Z_all[: len(X_clean)]
        Z_noisy = Z_all[len(X_clean):]
    else:
        Z_clean = Z_all
        Z_noisy = None

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(Z_clean[:, 0], Z_clean[:, 1],
                          c=y, cmap="tab10", s=8, alpha=0.7)
    plt.colorbar(scatter, label="Digit label")
    plt.title("Clean final excitatory states (t-SNE)")
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_tsne_clean_states.png", dpi=200)
    else:
        plt.show()
    plt.close()

    if has_noisy:
        plt.figure(figsize=(6, 5))
        plt.scatter(Z_clean[:, 0], Z_clean[:, 1],
                    c="blue", s=6, alpha=0.5, label="clean")
        plt.scatter(Z_noisy[:, 0], Z_noisy[:, 1],
                    c="red", s=6, alpha=0.5, label="noisy")
        plt.title("Clean vs noisy final excitatory states (t-SNE)")
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_tsne_clean_vs_noisy.png", dpi=200)
        else:
            plt.show()
        plt.close()


# ---------------- E/I balance plots ---------------- #

def plot_ei_balance_over_time(data, save_prefix=None):
    """
    Plot mean |balance_e| and |balance_i| over time, clean (and noisy if present).
    """
    bal_e_mag_clean = data["bal_e_mag_clean"]   # [T]
    bal_i_mag_clean = data["bal_i_mag_clean"]   # [T]
    T = bal_e_mag_clean.shape[0]

    t_axis = np.arange(T)

    plt.figure(figsize=(6, 4))
    plt.plot(t_axis, to_np(bal_e_mag_clean), "-o", label="E balance (clean)")
    plt.plot(t_axis, to_np(bal_i_mag_clean), "-o", label="I balance (clean)")
    plt.xlabel("time step")
    plt.ylabel("mean |balance|")
    plt.title("E/I balance magnitude over time (clean)")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_balance_clean.png", dpi=200)
    else:
        plt.show()
    plt.close()

    # If noisy balance magnitudes are not precomputed, we can approximate
    if "bal_e_noisy" in data:
        bal_e_noisy = data["bal_e_noisy"]   # [B, T, n_exc]
        bal_i_noisy = data["bal_i_noisy"]   # [B, T, n_inh]
        bal_e_mag_noisy = bal_e_noisy.abs().mean(dim=(0, 2))  # [T]
        bal_i_mag_noisy = bal_i_noisy.abs().mean(dim=(0, 2))

        plt.figure(figsize=(6, 4))
        plt.plot(t_axis, to_np(bal_e_mag_noisy), "-o", label="E balance (noisy)")
        plt.plot(t_axis, to_np(bal_i_mag_noisy), "-o", label="I balance (noisy)")
        plt.xlabel("time step")
        plt.ylabel("mean |balance|")
        plt.title("E/I balance magnitude over time (noisy)")
        plt.legend()
        plt.tight_layout()
        if save_prefix:
            plt.savefig(f"{save_prefix}_balance_noisy.png", dpi=200)
        else:
            plt.show()
        plt.close()


def plot_accuracy_over_time(data, save_prefix=None):
    acc_clean = data["acc_per_t_clean"]  # [T]
    T = acc_clean.shape[0]
    t_axis = np.arange(T)

    plt.figure(figsize=(6, 4))
    plt.plot(t_axis, to_np(acc_clean), "-o", label="clean")

    if "acc_per_t_noisy" in data:
        acc_noisy = data["acc_per_t_noisy"]
        plt.plot(t_axis, to_np(acc_noisy), "-o", label="noisy")

    plt.xlabel("time step")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.05)
    plt.title("Accuracy vs time")
    plt.legend()
    plt.tight_layout()
    if save_prefix:
        plt.savefig(f"{save_prefix}_accuracy_vs_time.png", dpi=200)
    else:
        plt.show()
    plt.close()


# ---------------- Main CLI ---------------- #

def main():
    parser = argparse.ArgumentParser(
        description="PCA/t-SNE/attractor/EI-balance analysis for BPAN .pt dynamics files."
    )
    parser.add_argument("--pt_path", type=str, required=True,
                        help="Path to .pt file saved by analyze_dynamics.py")
    parser.add_argument("--no_tsne", action="store_true",
                        help="Skip t-SNE (can be slow).")
    parser.add_argument("--save_prefix", type=str, default="results/analysis",
                        help="Prefix for saved PNG figures (directory will be created).")
    args = parser.parse_args()

    data = load_stats(args.pt_path)

    save_prefix = args.save_prefix
    Path(save_prefix).parent.mkdir(parents=True, exist_ok=True)

    # PCA on final states (clean + noisy)
    pca, Z_clean, Z_noisy, labels = pca_on_final_states(
        data, save_prefix=save_prefix
    )

    # Attractor trajectories (clean)
    plot_attractor_trajectories(
        data, pca=pca, save_prefix=save_prefix, n_per_class=3
    )

    # t-SNE
    if not args.no_tsne:
        tsne_on_final_states(
            data, save_prefix=save_prefix, max_points=2000, perplexity=30.0
        )

    # E/I balance
    plot_ei_balance_over_time(data, save_prefix=save_prefix)

    # Accuracy vs time
    plot_accuracy_over_time(data, save_prefix=save_prefix)

    # Extra: print final-state cosine from file if present
    if "final_state_cosine" in data:
        print("final_state_cosine (mean over batch):",
              float(data["final_state_cosine"].cpu()))


if __name__ == "__main__":
    main()

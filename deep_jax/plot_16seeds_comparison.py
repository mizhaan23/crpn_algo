import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    files = {
        "ACRPN (Tanh)": os.path.join(data_dir, "acrpn_Acrobot-v1_tanh_h64_64_16seeds.npz"),
        "REINFORCE (Tanh)": os.path.join(data_dir, "reinforce_Acrobot-v1_tanh_h64_64_16seeds.npz"),
        "ACRPN (ReLU)": os.path.join(data_dir, "acrpn_Acrobot-v1_relu_h64_64_16seeds.npz"),
        "REINFORCE (ReLU)": os.path.join(data_dir, "reinforce_Acrobot-v1_relu_h64_64_16seeds.npz"),
    }
    
    data = {}
    for name, path in files.items():
        if os.path.exists(path):
            d = np.load(path)
            data[name] = d["returns"] # (seeds, updates)
            print(f"Loaded {name}: {data[name].shape}")
        else:
            print(f"File not found: {path}")
            return

    # Style configuration
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
    
    colors = {
        "ACRPN": "#1f77b4",       # Deep blue
        "REINFORCE": "#d62728",   # Crimson red
    }
    
    updates = np.arange(1, data["ACRPN (Tanh)"].shape[1] + 1)

    # ----------------- Subplot 1: Tanh Activation -----------------
    ax = axes[0]
    for algo, col in colors.items():
        key = f"{algo} (Tanh)"
        ret = data[key]
        mean = np.mean(ret, axis=0)
        std = np.std(ret, axis=0)
        ax.plot(updates, mean, label=algo, color=col, lw=2.2)
        ax.fill_between(updates, mean - std, mean + std, color=col, alpha=0.18)
    
    ax.axhline(-100, color="gray", linestyle=":", lw=1.2, label="Solved Threshold (~ -100)")
    ax.set_title("Acrobot-v1: Tanh Activation (16 Seeds)", fontweight="bold")
    ax.set_xlabel("Policy Updates")
    ax.set_ylabel("Average Episode Return")
    ax.set_ylim(-510, -50)
    ax.legend(loc="lower right", framealpha=0.9)

    # ----------------- Subplot 2: ReLU Activation -----------------
    ax = axes[1]
    for algo, col in colors.items():
        key = f"{algo} (ReLU)"
        ret = data[key]
        mean = np.mean(ret, axis=0)
        std = np.std(ret, axis=0)
        ax.plot(updates, mean, label=algo, color=col, lw=2.2)
        ax.fill_between(updates, mean - std, mean + std, color=col, alpha=0.18)

    ax.axhline(-100, color="gray", linestyle=":", lw=1.2, label="Solved Threshold (~ -100)")
    ax.set_title("Acrobot-v1: ReLU Activation (16 Seeds)", fontweight="bold")
    ax.set_xlabel("Policy Updates")
    ax.set_ylim(-510, -50)
    ax.legend(loc="lower right", framealpha=0.9)

    # ----------------- Subplot 3: Final Return Distribution (Boxplot) -----------------
    ax = axes[2]
    plot_labels = ["ACRPN\n(Tanh)", "REINFORCE\n(Tanh)", "ACRPN\n(ReLU)", "REINFORCE\n(ReLU)"]
    box_data = [
        data["ACRPN (Tanh)"][:, -1],
        data["REINFORCE (Tanh)"][:, -1],
        data["ACRPN (ReLU)"][:, -1],
        data["REINFORCE (ReLU)"][:, -1],
    ]
    
    box_colors = [colors["ACRPN"], colors["REINFORCE"], colors["ACRPN"], colors["REINFORCE"]]
    bplot = ax.boxplot(box_data, patch_artist=True, tick_labels=plot_labels,
                       medianprops=dict(color="black", lw=1.8),
                       flierprops=dict(marker='o', markersize=5, alpha=0.6))
    
    for patch, col in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
        patch.set_edgecolor(col)
        patch.set_linewidth(1.5)

    # Overlay individual seed points with jitter
    for i, pts in enumerate(box_data):
        jitter = np.random.normal(0, 0.05, size=len(pts))
        ax.scatter(np.ones(len(pts)) * (i + 1) + jitter, pts, color=box_colors[i], alpha=0.75, s=25, edgecolor='black', linewidth=0.5, zorder=3)

    ax.axhline(-100, color="gray", linestyle=":", lw=1.2)
    ax.set_title("Final Return Distribution Across 16 Seeds", fontweight="bold")
    ax.set_ylabel("Final Episode Return")
    ax.set_ylim(-515, -50)

    plt.suptitle("Acrobot-v1: ACRPN (Second-Order) vs REINFORCE (First-Order) Benchmark (16 Seeds)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    # Save to deep_jax, repo root, and artifact directory
    repo_root = os.path.abspath(os.path.join(data_dir, "..", ".."))
    out_root = os.path.join(repo_root, "acrobot_16seeds_relu_vs_tanh_benchmark.png")
    plt.savefig(out_root, bbox_inches='tight')
    print(f"Saved: {out_root}")

    # Copy to artifacts directory
    artifact_dir = r"C:\Users\mizha\.gemini\antigravity-ide\brain\0f1f779b-f308-425c-acf8-759cfeca2868"
    if os.path.exists(artifact_dir):
        dest = os.path.join(artifact_dir, "acrobot_16seeds_relu_vs_tanh_benchmark.png")
        shutil.copyfile(out_root, dest)
        print(f"Copied to artifact: {dest}")

if __name__ == "__main__":
    main()

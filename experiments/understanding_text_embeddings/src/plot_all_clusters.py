import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
import os

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
PLOT_DIR = EXP_ROOT / "reports/phase1/visuals"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_clusters():
    print("🎨 Generating 2D Cluster Plots for all 8 variants...")
    
    datasets = load_all_text_datasets(split="val")
    
    # We will create a grid of plots
    fig, axes = plt.subplots(4, 2, figsize=(20, 25))
    axes = axes.flatten()
    
    for i, ds in enumerate(datasets):
        print(f"   -> Plotting: {ds.name}")
        X, y, label_names = ds.get_data()
        
        # 1. Project to 2D using PCA
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        
        # 2. Calculate Centroids in 2D
        centroids_2d = []
        for label_idx in range(len(label_names)):
            centroids_2d.append(np.mean(X_vis[y == label_idx], axis=0))
        centroids_2d = np.array(centroids_2d)
        
        # 3. Scatter Plot
        ax = axes[i]
        for label_idx, label_name in enumerate(label_names):
            mask = (y == label_idx)
            ax.scatter(X_vis[mask, 0], X_vis[mask, 1], label=label_name, alpha=0.4, s=10)
        
        # 4. Plot Centroids
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='black', marker='X', s=100, label='Centroids', edgecolors='white')
        
        ax.set_title(f"Cluster Geometry: {ds.name}", fontsize=14)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc='upper right', markerscale=2)

    plt.tight_layout()
    plt.savefig(EXP_ROOT / "reports/phase1/all_clusters_comparison.png", dpi=150)
    plt.close()
    
    # Also save individual high-res plots for the report
    for ds in datasets:
        X, y, label_names = ds.get_data()
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        for label_idx, label_name in enumerate(label_names):
            mask = (y == label_idx)
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], label=label_name, alpha=0.4, s=12)
        
        # Centroids
        c_2d = np.array([np.mean(X_vis[y == l], axis=0) for l in range(len(label_names))])
        plt.scatter(c_2d[:, 0], c_2d[:, 1], c='black', marker='X', s=150, edgecolors='white', linewidth=2)
        
        plt.title(f"Detailed Geometry: {ds.name}", fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"cluster_{ds.name}.png")
        plt.close()

    print(f"✅ Plots saved to {PLOT_DIR} and overall comparison at reports/phase1/")

if __name__ == "__main__":
    plot_clusters()

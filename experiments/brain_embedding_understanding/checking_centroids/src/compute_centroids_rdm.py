import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from pathlib import Path

# Reuse existing loaders
import sys
REPO_ROOT = Path(__file__).resolve().parents[4]
TEXT_SRC = REPO_ROOT / "experiments/understanding_text_embeddings/src"
if str(TEXT_SRC) not in sys.path:
    sys.path.append(str(TEXT_SRC))
from loader_text import load_all_text_datasets
from loader_brain import load_brain_data

# --- CONFIG ---
EXP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_ROOT / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_centroids(X, y, label_names):
    """Computes class centroids."""
    unique_labels = np.unique(y)
    centroids = []
    names = []
    for label in unique_labels:
        name = label_names[label] if isinstance(label, (int, np.integer)) else label
        if name == "joy": name = "happiness"
        pts = X[y == label]
        if len(pts) > 0:
            centroids.append(np.mean(pts, axis=0))
            names.append(name)
    return np.array(centroids), names

def plot_centroid_map(centroids, names, title, filename):
    """Plots centroids in 2D using PCA."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(centroids)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=200, c='red', marker='o', edgecolors='black', alpha=0.7)
    
    for i, name in enumerate(names):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), xytext=(10, 10), 
                     textcoords='offset points', fontsize=12, fontweight='bold')
        
    plt.title(title, fontsize=14)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.grid(alpha=0.3)
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.axvline(0, color='grey', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

def main():
    print("🚀 Starting Centroid Relational and PCA Analysis...")
    
    # 1. Process LLM Data
    datasets = load_all_text_datasets()
    target_models = ["qwen-768", "mpnet-balanced"]
    
    for ds in datasets:
        if any(m in ds.name.lower() for m in target_models):
            print(f"   Processing LLM: {ds.name}")
            X, y, names = ds.get_data()
            
            # Get Centroids
            c_vals, c_names = get_centroids(X, y, names)
            
            # Plot PCA Map
            plot_centroid_map(c_vals, c_names, f"Centroid Map (PCA): {ds.name}", f"centroid_map_{ds.name}.png")
            
            # Heatmaps
            dist_cos = squareform(pdist(c_vals, metric='cosine'))
            df_cos = pd.DataFrame(dist_cos, index=c_names, columns=c_names)
            plt.figure(figsize=(8, 7))
            sns.heatmap(df_cos, annot=True, fmt=".2f", cmap="YlGnBu_r")
            plt.title(f"Cosine RDM: {ds.name}")
            plt.savefig(OUTPUT_DIR / f"rdm_cosine_{ds.name}.png")
            plt.close()

    # 2. Process Brain Data
    print("   Processing Brain Data...")
    brain_csv = Path(
        os.environ.get(
            "BRAIN_48D_CSV",
            str(REPO_ROOT / "data/brain/human_subject_emotion_roi_48D_scaled.csv"),
        )
    )
    _, _, df_b = load_brain_data(brain_csv)
    
    roi_cols = [c for c in df_b.columns if c not in ['subject', 'emotion']]
    X_brain = df_b[roi_cols].values
    y_brain = df_b['emotion'].values
    
    # Get Centroids
    c_brain, c_names_brain = get_centroids(X_brain, y_brain, {l: l for l in np.unique(y_brain)})
    
    # Plot PCA Map
    plot_centroid_map(c_brain, c_names_brain, "Centroid Map (PCA): Human Brain", "centroid_map_brain.png")
    
    # Heatmap
    dist_brain = squareform(pdist(c_brain, metric='cosine'))
    df_brain = pd.DataFrame(dist_brain, index=c_names_brain, columns=c_names_brain)
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_brain, annot=True, fmt=".2f", cmap="YlGnBu_r")
    plt.title("Cosine RDM: Human Brain")
    plt.savefig(OUTPUT_DIR / "rdm_cosine_brain.png")
    plt.close()

    print(f"\n✅ Analysis complete. Maps and RDMs saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

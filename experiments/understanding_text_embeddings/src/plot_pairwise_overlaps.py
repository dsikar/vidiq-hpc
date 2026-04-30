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
VIS_DIR = EXP_ROOT / "reports/phase2/visuals/pairwise"
os.makedirs(VIS_DIR, exist_ok=True)

def plot_pairwise_overlaps():
    print("🎨 Generating Targeted Pairwise Overlap Plots (Fear-Anger, Love-Happiness, Surprise-Love)...")
    
    datasets = load_all_text_datasets(split="val")
    
    # Pairs defined by user for deep dive
    target_pairs = [
        ("fear", "anger"),
        ("love", "happiness"),
        ("surprise", "love")
    ]
    
    for ds in datasets:
        print(f"   -> Processing: {ds.name}")
        X, y, label_names = ds.get_data()
        
        for p1_name, p2_name in target_pairs:
            try:
                idx1 = label_names.index(p1_name)
                idx2 = label_names.index(p2_name)
            except ValueError: continue
            
            # 1. Filter for just these two emotions
            mask = (y == idx1) | (y == idx2)
            X_p, y_p = X[mask], y[mask]
            
            # 2. PCA to 2D for this specific pair
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X_p)
            
            # 3. Calculate Centroids
            c1 = np.mean(X_vis[y_p == idx1], axis=0)
            c2 = np.mean(X_vis[y_p == idx2], axis=0)
            
            # 4. Identify Geometric Overlaps
            # (Points of class 1 closer to c2, and points of class 2 closer to c1)
            dist1 = np.linalg.norm(X_vis - c1, axis=1)
            dist2 = np.linalg.norm(X_vis - c2, axis=1)
            
            is_ov_1 = (y_p == idx1) & (dist2 < dist1)
            is_ov_2 = (y_p == idx2) & (dist1 < dist2)
            
            # 5. Plot
            plt.figure(figsize=(10, 8))
            
            # Background points
            plt.scatter(X_vis[y_p == idx1, 0], X_vis[y_p == idx1, 1], label=p1_name, alpha=0.3, s=15, color='C0')
            plt.scatter(X_vis[y_p == idx2, 0], X_vis[y_p == idx2, 1], label=p2_name, alpha=0.3, s=15, color='C1')
            
            # Overlap points (bold)
            plt.scatter(X_vis[is_ov_1, 0], X_vis[is_ov_1, 1], color='red', s=25, label=f'Overlap ({p1_name} near {p2_name})', edgecolors='black', linewidth=0.5)
            plt.scatter(X_vis[is_ov_2, 0], X_vis[is_ov_2, 1], color='darkred', s=25, label=f'Overlap ({p2_name} near {p1_name})', edgecolors='black', linewidth=0.5)
            
            # Centroids
            plt.scatter([c1[0], c2[0]], [c1[1], c2[1]], c='black', marker='X', s=200, edgecolors='white', label='Centroids')
            
            plt.title(f"Pairwise Overlap: {ds.name}\n{p1_name.capitalize()} vs {p2_name.capitalize()}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(alpha=0.2)
            plt.tight_layout()
            
            plt.savefig(VIS_DIR / f"ov_{ds.name}_{p1_name}_{p2_name}.png")
            plt.close()

    print(f"✅ Target pairwise plots saved to {VIS_DIR}")

if __name__ == "__main__":
    plot_pairwise_overlaps()

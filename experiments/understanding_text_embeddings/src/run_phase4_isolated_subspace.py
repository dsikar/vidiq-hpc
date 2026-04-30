import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from scipy.linalg import svd

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase4"
VIS_DIR = REPORT_DIR / "visuals"
os.makedirs(VIS_DIR, exist_ok=True)

np.random.seed(42)

def isolate_and_evaluate(X, y, n_dims=20):
    """
    Isolates the top n_dims directions and evaluates them.
    """
    # 1. Identify Top Directions
    clf_full = LogisticRegression(max_iter=1000)
    clf_full.fit(X, y)
    U, S, Vh = svd(clf_full.coef_, full_matrices=False)
    V_top = Vh[:n_dims, :].T # (768, 20)
    
    # 2. Project Data
    X_20D = X @ V_top
    
    # 3. Silhouette Score
    sil_20D = silhouette_score(X_20D, y)
    
    # 4. 5-Fold Cross Validation Accuracy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs = []
    for train_idx, test_idx in skf.split(X_20D, y):
        clf_sub = LogisticRegression(max_iter=1000)
        clf_sub.fit(X_20D[train_idx], y[train_idx])
        accs.append(accuracy_score(y[test_idx], clf_sub.predict(X_20D[test_idx])))
    
    return X_20D, {
        "silhouette_20D": float(sil_20D),
        "accuracy_20D": float(np.mean(accs)),
        "accuracy_std_20D": float(np.std(accs))
    }

def plot_comparison(name, X_full, X_20D, y, label_names):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Plot Full 768D (via PCA)
    pca_full = PCA(n_components=2)
    X_full_vis = pca_full.fit_transform(X_full)
    for i, label in enumerate(label_names):
        mask = (y == i)
        axes[0].scatter(X_full_vis[mask, 0], X_full_vis[mask, 1], label=label, alpha=0.4, s=15)
    axes[0].set_title(f"{name}: Full 768D Manifold", fontsize=14)
    axes[0].grid(alpha=0.2)
    
    # 2. Plot Isolated 20D (via PCA)
    pca_20 = PCA(n_components=2)
    X_20_vis = pca_20.fit_transform(X_20D)
    for i, label in enumerate(label_names):
        mask = (y == i)
        axes[1].scatter(X_20_vis[mask, 0], X_20_vis[mask, 1], label=label, alpha=0.4, s=15)
    axes[1].set_title(f"{name}: Isolated 20D Subspace", fontsize=14)
    axes[1].grid(alpha=0.2)
    
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"comparison_scatter_{name}.png", dpi=150)
    plt.close()

def main():
    print("🚀 Starting Phase 4: Isolated Subspace Analysis (The Top 20 Test)...")
    datasets = load_all_text_datasets(split="val")
    
    # Load Phase 1 results for comparison
    with open(EXP_ROOT / "reports/phase1/baseline_metrics.json", "r") as f:
        p1_metrics = json.load(f)

    results = {}

    for ds in datasets:
        if "Mid" in ds.name: continue
        
        print(f"\n🔬 Processing: {ds.name}")
        X, y, label_names = ds.get_data()
        
        # Perform Isolation
        X_20D, metrics_20D = isolate_and_evaluate(X, y, n_dims=20)
        
        # Generate Visual Comparison
        print(f"      - Generating scatter comparison plot...")
        plot_comparison(ds.name, X, X_20D, y, label_names)
        
        results[ds.name] = {
            "baseline_768D": p1_metrics.get(ds.name, {}),
            "isolated_20D": metrics_20D
        }
        
        print(f"      Result: Silhouette={metrics_20D['silhouette_20D']:.4f}, Accuracy={metrics_20D['accuracy_20D']:.2%}")

    # Save metrics
    with open(REPORT_DIR / "subspace_metrics_20D.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Phase 4 Complete. Results and Plots saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

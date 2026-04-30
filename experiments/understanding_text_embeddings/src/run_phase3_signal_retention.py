import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.linalg import svd

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase3"
os.makedirs(REPORT_DIR, exist_ok=True)

np.random.seed(42)

def remove_direction(X, direction):
    u = direction / (np.linalg.norm(direction) + 1e-10)
    X_proj = (X @ u[:, np.newaxis]) @ u[np.newaxis, :]
    return X - X_proj

def run_iterative_direction_erasure(X, y, n_steps=200):
    """
    Surgically removes SVD directions one-by-one and re-evaluates.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    accuracies = []
    removed_weights = []
    
    clf = LogisticRegression(max_iter=1000)

    # 0. Baseline
    clf.fit(X_train, y_train)
    accuracies.append(float(accuracy_score(y_test, clf.predict(X_test))))

    for i in range(1, n_steps + 1):
        # 1. Fit to find CURRENT best direction
        clf.fit(X_train, y_train)
        U, S, Vh = svd(clf.coef_, full_matrices=False)
        
        top_dir = Vh[0, :]
        top_weight = S[0]
        
        removed_weights.append(float(top_weight))
        
        # 2. Remove from both train and test
        X_train = remove_direction(X_train, top_dir)
        X_test = remove_direction(X_test, top_dir)
        
        # 3. Fresh evaluation
        new_clf = LogisticRegression(max_iter=1000)
        new_clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, new_clf.predict(X_test))
        accuracies.append(float(acc))
        
        if i % 50 == 0:
            print(f"      - Step {i}/200 complete. Signal Weight: {top_weight:.4f}, Acc: {acc:.4f}")

    return accuracies, removed_weights

def main():
    print("🚀 Starting Phase 3: Recursive Directional Erasure (SVD)...")
    datasets = load_all_text_datasets(split="val")
    
    results = {}
    chance_level = 1.0 / 6

    for ds in datasets:
        if "Mid" in ds.name: continue
        
        print(f"\n🔬 Analyzing Manifold: {ds.name}")
        X, y, _ = ds.get_data()
        
        accs, weights = run_iterative_direction_erasure(X, y, n_steps=200)
        results[ds.name] = {
            "accuracies": accs,
            "weights": weights
        }

    # --- PLOTTING 1: The Signal Cliff (First 25) ---
    print("\n📈 Plotting Signal Cliff (0-25)...")
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        linestyle = '-' if "FT" in name else '--'
        plt.plot(range(26), data["accuracies"][:26], label=name, marker='o', markersize=4, linestyle=linestyle, linewidth=2)
    
    plt.axhline(y=chance_level, color='black', linestyle=':', label='Chance (16.7%)')
    plt.title("The Signal Cliff: First 25 Directions Removed (SVD)", fontsize=15)
    plt.xlabel("Number of Directions Removed")
    plt.ylabel("Accuracy")
    plt.xticks(range(26))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "cliff_zoom_top25.png", dpi=200)
    plt.close()

    # --- PLOTTING 2: Full Erasure (0-200) ---
    print("📈 Plotting Full Decay (0-200)...")
    plt.figure(figsize=(12, 8))
    for name, data in results.items():
        linestyle = '-' if "FT" in name else '--'
        plt.plot(range(201), data["accuracies"], label=name, linestyle=linestyle, linewidth=2)
    
    plt.axhline(y=chance_level, color='black', linestyle=':', label='Chance (16.7%)')
    plt.title("Directional Signal Erasure: 200 Iterations", fontsize=15)
    plt.xlabel("Number of Directions Removed")
    plt.ylabel("Accuracy")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "full_erasure_200.png", dpi=200)
    plt.close()

    # Save metrics
    with open(REPORT_DIR / "retention_metrics_top200_directions.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Phase 3 Complete. Results in {REPORT_DIR}")

if __name__ == "__main__":
    main()

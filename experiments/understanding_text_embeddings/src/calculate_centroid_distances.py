import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from scipy.linalg import svd
from scipy.spatial.distance import pdist, squareform

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase4"
os.makedirs(REPORT_DIR, exist_ok=True)

def get_20d_centroids(X, y, n_dims=20):
    # 1. Isolate Top 20 Directions
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    U, S, Vh = svd(clf.coef_, full_matrices=False)
    V_top = Vh[:n_dims, :].T
    X_20D = X @ V_top
    
    # 2. Calculate Centroids
    unique_labels = np.unique(y)
    centroids = np.array([np.mean(X_20D[y == i], axis=0) for i in unique_labels])
    return centroids

def main():
    print("🚀 Calculating Centroid Distances in 20D Subspace...")
    datasets = load_all_text_datasets(split="val")
    
    all_distance_data = {}

    for ds in datasets:
        if "Mid" in ds.name: continue
        
        print(f"   -> Processing: {ds.name}")
        X, y, label_names = ds.get_data()
        
        centroids = get_20d_centroids(X, y)
        
        # Calculate Pairwise Euclidean Distances
        dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        
        # Normalize by mean distance to make them comparable
        # (Since absolute scales differ between models)
        mean_dist = np.mean(dist_matrix[np.triu_indices(6, k=1)])
        norm_dist_matrix = dist_matrix / mean_dist
        
        # Convert to DataFrame for easier display
        df = pd.DataFrame(norm_dist_matrix, index=label_names, columns=label_names)
        
        all_distance_data[ds.name] = {
            "raw_matrix": dist_matrix.tolist(),
            "norm_matrix": norm_dist_matrix.tolist(),
            "labels": label_names
        }

    # Save to JSON
    with open(REPORT_DIR / "centroid_distances_20D.json", "w") as f:
        json.dump(all_distance_data, f, indent=4)
        
    print(f"✅ Centroid distances calculated and saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

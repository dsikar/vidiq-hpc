import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase1"
os.makedirs(REPORT_DIR, exist_ok=True)

def main():
    print("🚀 Calculating Centroid Distances in Full 768D Space (Phase 1)...")
    datasets = load_all_text_datasets(split="val")
    
    all_distance_data = {}

    for ds in datasets:
        print(f"   -> Processing: {ds.name}")
        X, y, label_names = ds.get_data()
        
        # Calculate Centroids in full 768D
        unique_labels = np.unique(y)
        centroids = np.array([np.mean(X[y == i], axis=0) for i in unique_labels])
        
        # Calculate Pairwise Euclidean Distances
        dist_matrix = squareform(pdist(centroids, metric='euclidean'))
        
        # Normalize by mean distance of the non-diagonal elements
        mean_dist = np.mean(dist_matrix[np.triu_indices(len(unique_labels), k=1)])
        norm_dist_matrix = dist_matrix / (mean_dist + 1e-10)
        
        all_distance_data[ds.name] = {
            "norm_matrix": norm_dist_matrix.tolist(),
            "labels": label_names
        }

    # Save to JSON
    with open(REPORT_DIR / "centroid_distances_768D.json", "w") as f:
        json.dump(all_distance_data, f, indent=4)
        
    print(f"✅ Phase 1 centroid distances saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

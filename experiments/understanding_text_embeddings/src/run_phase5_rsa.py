import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
P1_DIST_PATH = EXP_ROOT / "reports/phase1/centroid_distances_768D.json"
P4_DIST_PATH = EXP_ROOT / "reports/phase4/centroid_distances_20D.json"
REPORT_DIR = EXP_ROOT / "reports/phase5"
os.makedirs(REPORT_DIR, exist_ok=True)

def get_upper_tri(matrix):
    """Extracts the upper triangle values of a square matrix."""
    m = np.array(matrix)
    return m[np.triu_indices(m.shape[0], k=1)]

def main():
    print("🚀 Starting Phase 5: Representational Similarity Analysis (RSA)...")
    
    # 1. Load All RDMs
    with open(P1_DIST_PATH, "r") as f:
        p1_data = json.load(f)
    with open(P4_DIST_PATH, "r") as f:
        p4_data = json.load(f)
        
    # Standardize names and collect triangles
    all_systems = {}
    
    # Add 768D variants
    for name, data in p1_data.items():
        all_systems[f"{name} (768D)"] = get_upper_tri(data['norm_matrix'])
        
    # Add 20D variants
    for name, data in p4_data.items():
        all_systems[f"{name} (20D)"] = get_upper_tri(data['norm_matrix'])
        
    system_names = list(all_systems.keys())
    n_systems = len(system_names)
    
    # 2. Compute RSA Matrix (Spearman Correlation)
    rsa_matrix = np.zeros((n_systems, n_systems))
    
    for i in range(n_systems):
        for j in range(n_systems):
            vec_i = all_systems[system_names[i]]
            vec_j = all_systems[system_names[j]]
            corr, _ = spearmanr(vec_i, vec_j)
            rsa_matrix[i, j] = corr
            
    rsa_df = pd.DataFrame(rsa_matrix, index=system_names, columns=system_names)
    
    # 3. Plotting the RSA Matrix
    print("📈 Generating RSA Heatmap...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(rsa_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("RSA Matrix: Representational Similarity of Emotion Logic", fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "rsa_correlation_matrix.png", dpi=150)
    plt.close()
    
    # 4. Save results
    rsa_df.to_json(REPORT_DIR / "rsa_results.json")
    
    print(f"✅ RSA Analysis Complete. Results saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

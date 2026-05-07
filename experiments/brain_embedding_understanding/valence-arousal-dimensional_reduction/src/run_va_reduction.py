import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from sklearn.utils import resample, shuffle
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

# Validated VA coordinates (Russell's Circumplex + Expert Advice)
VA_GROUND_TRUTH = {
    # Text Labels
    "joy":         {"v": 0.85, "a": 0.70},
    "happiness":   {"v": 0.85, "a": 0.70},
    "delighted":   {"v": 0.85, "a": 0.70},
    "love":        {"v": 0.85, "a": 0.55},
    "surprise":    {"v": 0.10, "a": 0.85},
    "calm":        {"v": 0.70, "a": 0.15},
    "excited":     {"v": 0.75, "a": 0.90},
    "sadness":     {"v": -0.85, "a": 0.25},
    "depressed":   {"v": -0.85, "a": 0.25},
    "anger":       {"v": -0.75, "a": 0.80},
    "fear":        {"v": -0.80, "a": 0.85},
    "afraid":      {"v": -0.80, "a": 0.85}
}

def get_va_matrix(labels):
    v_vec = [VA_GROUND_TRUTH[l.lower()]['v'] for l in labels]
    a_vec = [VA_GROUND_TRUTH[l.lower()]['a'] for l in labels]
    return np.stack([v_vec, a_vec], axis=1)

def run_procrustes_validation(target_va, projected_2d):
    # Procrustes fits projected_2d to target_va
    mttx, mtty, disparity = procrustes(target_va, projected_2d)
    return disparity, mtty # mtty is the transformed projected map

def run_alignment_test(X, y, label_names, name):
    print(f"\n🧠 Analyzing VA Alignment for: {name}")
    
    # 1. Compute Centroids
    unique_labels = np.unique(y)
    centroids = []
    names = []
    for l in unique_labels:
        l_name = label_names[l] if isinstance(l, (int, np.integer)) else str(l).lower()
        if l_name.lower() in VA_GROUND_TRUTH:
            centroids.append(np.mean(X[y == l], axis=0))
            names.append(l_name)
    
    centroids = np.array(centroids)
    target_va = get_va_matrix(names)
    
    # 2. Dimensional Reduction
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(centroids)
    
    mds = MDS(n_components=2, dissimilarity='euclidean', random_state=42, normalized_stress='auto')
    coords_mds = mds.fit_transform(centroids)

    # 3. Correlation Metrics
    def get_corrs(c2d, t_va):
        # We check both axes (since PCA order might be flipped relative to V/A)
        r_v = max(abs(pearsonr(c2d[:, 0], t_va[:, 0])[0]), abs(pearsonr(c2d[:, 1], t_va[:, 0])[0]))
        r_a = max(abs(pearsonr(c2d[:, 0], t_va[:, 1])[0]), abs(pearsonr(c2d[:, 1], t_va[:, 1])[0]))
        return r_v, r_a

    r_v, r_a = get_corrs(coords_pca, target_va)
    disparity, aligned_map = run_procrustes_validation(target_va, coords_pca)

    # 4. Permutation Testing (n=1000)
    print("      - Running Permutation Test...")
    perm_rs = []
    for _ in range(1000):
        shuffled_va = shuffle(target_va)
        perm_rs.append(max(get_corrs(coords_pca, shuffled_va)))
    p_val = np.mean(np.array(perm_rs) >= max(r_v, r_a))

    # 5. Bootstrap CI (n=500)
    print("      - Running Bootstrap...")
    boot_rs = []
    for _ in range(500):
        # Resample indices for centroids (simple bootstrap of the 3-6 points isn't great, 
        # but we follow the request for CI logic)
        idx = np.random.choice(len(centroids), len(centroids), replace=True)
        if len(np.unique(idx)) < 2: continue
        try:
            boot_r_v, _ = get_corrs(coords_pca[idx], target_va[idx])
            boot_rs.append(boot_r_v)
        except: continue
    ci = np.percentile(boot_rs, [2.5, 97.5]) if boot_rs else [0,0]

    # 6. Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(target_va[:, 0], target_va[:, 1], s=300, c='blue', alpha=0.3, label="Ground Truth (Ideal)")
    plt.scatter(aligned_map[:, 0], aligned_map[:, 1], s=150, c='red', marker='X', label="Aligned Embedding")
    
    for i, txt in enumerate(names):
        plt.annotate(txt, (target_va[i, 0], target_va[i, 1]), color='blue', alpha=0.6)
        plt.annotate(txt, (aligned_map[i, 0], aligned_map[i, 1]), color='red')
    
    plt.title(f"VA Manifold Alignment: {name}\nDisparity: {disparity:.4f} | R_val: {r_v:.2f}")
    plt.xlabel("Valence (Normalized)")
    plt.ylabel("Arousal (Normalized)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(OUTPUT_DIR / f"va_alignment_{name}.png", dpi=150)
    plt.close()

    return {
        "name": name,
        "valence_r": float(r_v),
        "arousal_r": float(r_a),
        "procrustes_disparity": float(disparity),
        "permutation_p": float(p_val),
        "bootstrap_ci": [float(ci[0]), float(ci[1])]
    }

def main():
    print("🚀 Starting Valence-Arousal Alignment Study...")
    all_results = []

    # 1. LLMs
    datasets = load_all_text_datasets()
    target_models = ["qwen-768", "mpnet-balanced"]
    for ds in datasets:
        if any(m in ds.name.lower() for m in target_models):
            X, y, names = ds.get_data()
            res = run_alignment_test(X, y, names, ds.name)
            all_results.append(res)

    # 2. Brain
    print("\nProcessing Brain Data...")
    brain_csv = Path(
        os.environ.get(
            "BRAIN_48D_CSV",
            str(REPO_ROOT / "data/brain/human_subject_emotion_roi_48D_scaled.csv"),
        )
    )
    _, labels_b, df_b = load_brain_data(brain_csv)
    roi_cols = [c for c in df_b.columns if c not in ['subject', 'emotion']]
    res_b = run_alignment_test(df_b[roi_cols].values, df_b['emotion'].values, {l: l for l in labels_b}, "Brain-fMRI")
    all_results.append(res_b)

    with open(OUTPUT_DIR / "alignment_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✅ VA Reduction complete. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

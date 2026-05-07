import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from sklearn.utils import resample
from pathlib import Path

# Reuse existing loaders
import sys
REPO_ROOT = Path(__file__).resolve().parents[4]
TEXT_SRC = REPO_ROOT / "experiments/understanding_text_embeddings/src"
if str(TEXT_SRC) not in sys.path:
    sys.path.append(str(TEXT_SRC))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).resolve().parents[1]
BRAIN_11D_CSV = Path(
    os.environ.get(
        "BRAIN_11D_CSV",
        str(
            REPO_ROOT
            / "experiments/brain_embedding_understanding/adding_spatial_context/outputs/brain_11d_representation.csv"
        ),
    )
)
OUTPUT_DIR = EXP_ROOT / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_LABELS = ["fear", "happiness", "sadness"]
BRAIN_MAP = {
    "afraid": "fear",
    "calm": "happiness",
    "delighted": "happiness",
    "depressed": "sadness",
    "excited": "happiness"
}

def get_triu(matrix):
    return matrix[np.triu_indices(len(matrix), k=1)]

def compute_centroids(X, y, label_names):
    unique_labels = np.unique(y)
    centroids = {}
    for label in unique_labels:
        name = label_names[label] if isinstance(label, (int, np.integer)) else str(label).lower()
        if name == "joy": name = "happiness" # Standardize LLM
        
        pts = X[y == label]
        centroids[name] = np.mean(pts, axis=0)
    return centroids

def get_3class_vector(centroids, metric='cosine'):
    # Fear-Happiness, Fear-Sadness, Happiness-Sadness
    c_list = [centroids['fear'], centroids['happiness'], centroids['sadness']]
    rdm = squareform(pdist(np.array(c_list), metric=metric))
    return get_triu(rdm)

def plot_affect_map(centroids, title, filename):
    names = list(centroids.keys())
    vals = np.array(list(centroids.values()))
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vals)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(coords[:, 0], coords[:, 1], s=200, c='red', edgecolors='black')
    for i, name in enumerate(names):
        plt.annotate(name, (coords[i, 0], coords[i, 1]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()

def main():
    print("🚀 Starting 11D Centroid Analysis...")
    
    # 1. Load Brain 11D
    df_b = pd.read_csv(BRAIN_11D_CSV)
    df_b['emotion_raw'] = df_b['emotion'].str.lower()
    df_b['emotion_mapped'] = df_b['emotion_raw'].map(BRAIN_MAP)
    
    feat_cols = ["frontal", "temporal", "parietal", "occipital", "limbic", 
                 "salience", "dmn", "cen", "limbic_net", "visual", "neighbor_context"]
    
    # Compute mapped brain centroids
    brain_centroids = {}
    for target in TARGET_LABELS:
        pts = df_b[df_b.emotion_mapped == target][feat_cols].values
        brain_centroids[target] = np.mean(pts, axis=0)
    
    # 2. Load LLM Data
    llm_rdms = {}
    llm_centroids_all = {}
    datasets = load_all_text_datasets()
    for ds in datasets:
        if "qwen-768" in ds.name.lower() or "mpnet-balanced" in ds.name.lower():
            X, y, names = ds.get_data()
            c_dict = compute_centroids(X, y, names)
            llm_centroids_all[ds.name] = c_dict
            llm_rdms[ds.name] = get_3class_vector(c_dict)

    # 3. Numerical Comparison
    v_brain = get_3class_vector(brain_centroids)
    
    results = {"brain_vector": v_brain.tolist(), "models": {}}
    
    for name, v_llm in llm_rdms.items():
        r, _ = pearsonr(v_brain, v_llm)
        rho, _ = spearmanr(v_brain, v_llm)
        v_b_n = v_brain / np.linalg.norm(v_brain)
        v_l_n = v_llm / np.linalg.norm(v_llm)
        err = np.linalg.norm(v_b_n - v_l_n)
        
        results["models"][name] = {
            "pearson_r": float(r),
            "spearman_rho": float(rho),
            "geometric_error": float(err),
            "vector": v_llm.tolist()
        }
        
    # 4. Plots
    plot_affect_map(brain_centroids, "11D Brain Affect Map (PCA)", "centroid_map_brain_11d.png")
    for name, c_dict in llm_centroids_all.items():
        # Subset to the same 3 emotions for visual consistency
        subset = {k: c_dict[k] for k in TARGET_LABELS}
        plot_affect_map(subset, f"LLM Affect Map ({name})", f"centroid_map_{name}.png")

    # 5. Bootstrap Validation
    print("   -> Running Bootstrap...")
    boot_corrs = []
    for _ in range(500):
        boot_df = resample(df_b)
        b_c = {}
        valid = True
        for target in TARGET_LABELS:
            pts = boot_df[boot_df.emotion_mapped == target][feat_cols].values
            if len(pts) == 0: 
                valid = False; break
            b_c[target] = np.mean(pts, axis=0)
        
        if valid:
            v_boot = get_3class_vector(b_c)
            boot_corrs.append(pearsonr(v_boot, llm_rdms['Qwen-768'])[0])
    
    results["bootstrap_qwen_ci"] = np.percentile(boot_corrs, [2.5, 97.5]).tolist()

    with open(OUTPUT_DIR / "rdm_results_11d.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Done. Reports in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

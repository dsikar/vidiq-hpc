import numpy as np
import pandas as pd
import json
import os
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.utils import resample
from pathlib import Path

# Reuse existing loaders
import sys
sys.path.append("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings/src")
from loader_text import load_all_text_datasets
from loader_brain import load_brain_data

# --- CONFIG ---
EXP_ROOT = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/brain_embedding_understanding/checking_centroids")
OUTPUT_DIR = EXP_ROOT / "reports"
BRAIN_CSV = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/human_brain_emotion_exports/human_subject_emotion_roi_48D_scaled.csv"

EMOTION_MAP = {
    "afraid": "fear",
    "calm": "happiness",
    "delighted": "happiness",
    "depressed": "sadness",
    "excited": "happiness"
}
TARGET_LABELS = ["fear", "happiness", "sadness"]

def get_triu(matrix):
    return matrix[np.triu_indices(len(matrix), k=1)]

def get_3class_rdm_vector(X, y, label_names, metric='cosine'):
    """Extracts the 3 unique pairwise distances for fear, happiness, sadness."""
    mapping = {
        "afraid": "fear", "calm": "happiness", "delighted": "happiness", 
        "excited": "happiness", "depressed": "sadness",
        "fear": "fear", "happiness": "happiness", "joy": "happiness", "sadness": "sadness"
    }
    
    unique_labels = np.unique(y)
    mapped_centroids = {}
    
    for label in unique_labels:
        name = label_names[label] if isinstance(label, (int, np.integer)) else str(label).lower()
        target = mapping.get(name)
        if target:
            pts = X[y == label]
            if target not in mapped_centroids:
                mapped_centroids[target] = []
            mapped_centroids[target].append(np.mean(pts, axis=0))
            
    # Average centroids if multiple raw labels map to one target
    final_centroids = []
    for t in TARGET_LABELS:
        if t in mapped_centroids:
            final_centroids.append(np.mean(mapped_centroids[t], axis=0))
        else:
            return None # Missing one of the 3 classes
            
    if len(final_centroids) < 3:
        return None
        
    rdm = squareform(pdist(np.array(final_centroids), metric=metric))
    return get_triu(rdm)

def run_validation():
    print("🧪 Validating Relational Geometry Metrics...")
    
    # 1. Load LLM Reference (Qwen-768)
    datasets = load_all_text_datasets()
    qwen_ds = next(ds for ds in datasets if "qwen-768" in ds.name.lower())
    X_q, y_q, names_q = qwen_ds.get_data()
    v_qwen = get_3class_rdm_vector(X_q, y_q, names_q)

    # 2. Load Brain Data
    df_b_raw = pd.read_csv(BRAIN_CSV)
    df_b_raw['emotion'] = df_b_raw['emotion'].str.lower()
    roi_cols = [c for c in df_b_raw.columns if c not in ['subject', 'emotion']]
    subjects = df_b_raw['subject'].unique()
    
    # --- 1. NOISE CEILING ---
    print("   -> Computing Noise Ceiling (Subject Consistency)...")
    sub_rdms = []
    for s in subjects:
        sub_df = df_b_raw[df_b_raw.subject == s]
        # For a single subject, label_names is just the identity mapping
        lnames = {e: e for e in sub_df['emotion'].unique()}
        v_sub = get_3class_rdm_vector(sub_df[roi_cols].values, sub_df['emotion'].values, lnames)
        if v_sub is not None:
            sub_rdms.append(v_sub)
    
    sub_rdms = np.array(sub_rdms)
    if len(sub_rdms) > 1:
        mean_sub_rdm = np.mean(sub_rdms, axis=0)
        upper = np.mean([pearsonr(s, mean_sub_rdm)[0] for s in sub_rdms])
        lower_vals = []
        for i in range(len(sub_rdms)):
            others = np.mean(np.delete(sub_rdms, i, axis=0), axis=0)
            lower_vals.append(pearsonr(sub_rdms[i], others)[0])
        lower = np.mean(lower_vals)
    else:
        upper, lower = 0, 0

    # --- 2. BOOTSTRAP ---
    print("   -> Running Bootstrap (n=500)...")
    boot_corrs = []
    v_brain_obs = get_3class_rdm_vector(df_b_raw[roi_cols].values, df_b_raw['emotion'].values, {e: e for e in df_b_raw['emotion'].unique()})
    
    for _ in range(500):
        # Bootstrap at sample level
        boot_df = resample(df_b_raw)
        v_boot = get_3class_rdm_vector(boot_df[roi_cols].values, boot_df['emotion'].values, {e: e for e in boot_df['emotion'].unique()})
        if v_boot is not None:
            boot_corrs.append(pearsonr(v_boot, v_qwen)[0])
    
    ci = np.percentile(boot_corrs, [2.5, 97.5])

    # --- 3. METRIC SENSITIVITY ---
    print("   -> Checking Metric Sensitivity (L1 Distance)...")
    v_q_l1 = get_3class_rdm_vector(X_q, y_q, names_q, metric='cityblock')
    v_b_l1 = get_3class_rdm_vector(df_b_raw[roi_cols].values, df_b_raw['emotion'].values, {e: e for e in df_b_raw['emotion'].unique()}, metric='cityblock')
    r_l1 = pearsonr(v_q_l1, v_b_l1)[0]

    # --- RESULTS ---
    results = {
        "noise_ceiling": {"upper": float(upper), "lower": float(lower)},
        "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
        "manhattan_correlation": float(r_l1),
        "observed_pearson_r": float(pearsonr(v_brain_obs, v_qwen)[0])
    }

    print("\n" + "="*50)
    print("RELATIONAL VALIDATION SUMMARY")
    print("="*50)
    print(f"Noise Ceiling:      [{lower:.4f}, {upper:.4f}]")
    print(f"Bootstrap 95% CI:   [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Manhattan r:        {r_l1:.4f}")
    print(f"Observed Cosine r:  {results['observed_pearson_r']:.4f}")
    print("="*50)

    with open(OUTPUT_DIR / "relational_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_validation()

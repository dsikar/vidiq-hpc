import numpy as np
import pandas as pd
import json
import os
from scipy.stats import wasserstein_distance
from sklearn.utils import resample
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

EMOTION_MAP = {
    "afraid": "fear",
    "calm": "happiness",
    "delighted": "happiness",
    "depressed": "sadness",
    "excited": "happiness"
}

def get_density_array(X, y):
    """Returns a flat array of normalized distances to centroids."""
    all_dists = []
    for label in np.unique(y):
        pts = X[y == label]
        if len(pts) < 2: continue
        centroid = np.mean(pts, axis=0)
        d = np.linalg.norm(pts - centroid, axis=1)
        all_dists.extend((d / np.mean(d)).tolist())
    return np.array(all_dists)

def run_validation():
    print("🧪 Starting Statistical Validation of Density Metrics...")
    
    # 1. Load LLM Reference (Qwen-768 as the primary target)
    datasets = load_all_text_datasets()
    qwen_ds = next(ds for ds in datasets if "qwen-768" in ds.name.lower())
    X_q, y_q, _ = qwen_ds.get_data()
    qwen_density = get_density_array(X_q, y_q)

    # 2. Load Brain Data
    brain_csv = Path(
        os.environ.get(
            "BRAIN_48D_CSV",
            str(REPO_ROOT / "data/brain/human_subject_emotion_roi_48D_scaled.csv"),
        )
    )
    _, _, df_b = load_brain_data(brain_csv)
    
    print(f"      [DEBUG] Brain DF Columns: {df_b.columns.tolist()}")

    df_b["mapped"] = df_b["emotion"].map(EMOTION_MAP)
    df_b = df_b.dropna(subset=["mapped"])
    
    # Identify ROI columns (all numeric)
    roi_cols = df_b.select_dtypes(include=[np.number]).columns.tolist()
    if 'subject' in roi_cols: roi_cols.remove('subject')
    
    # --- OBSERVED METRIC ---
    brain_density_obs = get_density_array(df_b[roi_cols].values, df_b["mapped"].values)
    obs_wd = wasserstein_distance(qwen_density, brain_density_obs)
    
    # --- 1. NOISE CEILING (Brain vs Brain) ---
    print("   -> Calculating Noise Ceiling (Internal Consistency)...")
    # Split the raw samples into two random halves
    indices = np.arange(len(df_b))
    np.random.shuffle(indices)
    mid = len(indices) // 2
    
    b1 = df_b.iloc[indices[:mid]]
    b2 = df_b.iloc[indices[mid:]]
    
    d1 = get_density_array(b1[roi_cols].values, b1["mapped"].values)
    d2 = get_density_array(b2[roi_cols].values, b2["mapped"].values)
    noise_ceiling_wd = wasserstein_distance(d1, d2)

    # --- 2. BOOTSTRAP CI (Stability) ---
    print("   -> Running Bootstrap (n=500)...")
    boot_wds = []
    for _ in range(500):
        boot_df = resample(df_b)
        boot_density = get_density_array(boot_df[roi_cols].values, boot_df["mapped"].values)
        boot_wds.append(wasserstein_distance(qwen_density, boot_density))
    ci = np.percentile(boot_wds, [2.5, 97.5])

    # --- 3. PERMUTATION TEST (Significance) ---
    print("   -> Running Permutation Test (n=500)...")
    perm_wds = []
    for _ in range(500):
        perm_y = np.random.permutation(df_b["mapped"].values)
        perm_density = get_density_array(df_b[roi_cols].values, perm_y)
        perm_wds.append(wasserstein_distance(qwen_density, perm_density))
    p_val = np.mean(np.array(perm_wds) <= obs_wd)

    # --- RESULTS ---
    results = {
        "metric": "Wasserstein Distance (WD)",
        "observed_wd": float(obs_wd),
        "noise_ceiling_wd": float(noise_ceiling_wd),
        "95_ci": [float(ci[0]), float(ci[1])],
        "permutation_p_value": float(p_val),
        "interpretation": "Lower WD means higher similarity."
    }

    print("\n" + "="*50)
    print("DENSITY VALIDATION SUMMARY")
    print("="*50)
    print(f"Observed Brain-Qwen WD: {obs_wd:.4f}")
    print(f"Brain-Brain (Noise Ceiling): {noise_ceiling_wd:.4f}")
    print(f"95% Confidence Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"Permutation p-value:    {p_val:.4f}")
    print("="*50)

    with open(OUTPUT_DIR / "density_validation.json", "w") as f:
        json.dump(results, f, indent=4)
        
    return results

if __name__ == "__main__":
    run_validation()

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ks_2samp

# Reuse existing loaders
import sys
sys.path.append("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings/src")
from loader_text import load_all_text_datasets
from loader_brain import load_brain_data

# --- CONFIG ---
EXP_ROOT = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/brain_embedding_understanding/checking_density_geometry")
OUTPUT_DIR = EXP_ROOT / "reports"
SRC_DIR = EXP_ROOT / "src"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EMOTION_MAP = {
    "afraid": "fear",
    "calm": "happiness",
    "delighted": "happiness",
    "depressed": "sadness",
    "excited": "happiness"
}

def calculate_density(X, y, label_names):
    """Calculates density decay for a given dataset."""
    unique_labels = np.unique(y)
    results = {}
    all_norm_dists = []

    print(f"      [DEBUG] Input X shape: {X.shape}, unique labels: {unique_labels}")

    for label in unique_labels:
        # Get actual string name if numeric
        name = label_names[label] if isinstance(label, (int, np.integer)) else label
        
        # Normalize 'joy' to 'happiness' for cross-system consistency
        if name == "joy": name = "happiness"
        
        mask = (y == label)
        pts = X[mask]
        if len(pts) < 2: 
            continue
        
        centroid = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        
        mean_d = np.mean(dists)
        if mean_d == 0: continue
            
        norm_dists = dists / mean_d
        all_norm_dists.extend(norm_dists.tolist())
        
        counts, edges = np.histogram(norm_dists, bins=40, range=(0, 3), density=True)
        counts = np.nan_to_num(counts)
        
        results[name] = {
            "bins": ((edges[:-1] + edges[1:]) / 2).tolist(),
            "density": counts.tolist()
        }

    if not all_norm_dists:
        return {"_GLOBAL_": {"bins": [], "density": []}}

    # Global Hist
    counts, edges = np.histogram(all_norm_dists, bins=40, range=(0, 3), density=True)
    counts = np.nan_to_num(counts)
    results["_GLOBAL_"] = {
        "bins": ((edges[:-1] + edges[1:]) / 2).tolist(),
        "density": counts.tolist()
    }
    
    return results

def main():
    print("🚀 Starting Cross-System Density Analysis (Metadata Aligned)...")
    
    # 1. Load LLM Data
    datasets = load_all_text_datasets()
    llm_results = {}
    target_models = ["qwen-768", "mpnet-balanced", "bge-balanced"]
    
    for ds in datasets:
        if any(m in ds.name.lower() for m in target_models):
            X, y, names = ds.get_data()
            print(f"   Processing LLM: {ds.name}")
            llm_results[ds.name] = calculate_density(X, y, names)

    # 2. Load Brain Data
    print("   Processing Brain Data...")
    brain_csv = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/human_brain_emotion_exports/human_subject_emotion_roi_48D_scaled.csv"
    _, _, df_z = load_brain_data(brain_csv)
    
    df_z["mapped"] = df_z["emotion"].map(EMOTION_MAP)
    df_z = df_z.dropna(subset=["mapped"])
    
    roi_cols = df_z.columns[2:-1]
    X_brain = df_z[roi_cols].values
    y_brain = df_z["mapped"].values
    
    brain_results = calculate_density(X_brain, y_brain, {l: l for l in np.unique(y_brain)})

    # 3. Visualization: Global Comparison
    print("📈 Generating Visualizations...")
    plt.figure(figsize=(12, 7))
    for name, data in llm_results.items():
        plt.plot(data["_GLOBAL_"]["bins"], data["_GLOBAL_"]["density"], label=f"{name}", linewidth=2)
    plt.plot(brain_results["_GLOBAL_"]["bins"], brain_results["_GLOBAL_"]["density"], 
             label="Human Brain", linewidth=4, color='black', linestyle='--')
    plt.title("Global Density Decay Comparison")
    plt.xlabel("Norm. Distance")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(OUTPUT_DIR / "global_density_comparison.png", dpi=150)
    plt.close()

    # 4. Visualization: All 6 Emotions Comparison
    all_emotions = ["anger", "fear", "happiness", "love", "sadness", "surprise"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    axes = axes.flatten()
    
    for i, emo in enumerate(all_emotions):
        # Plot LLMs
        for name, data in llm_results.items():
            if emo in data:
                axes[i].plot(data[emo]["bins"], data[emo]["density"], label=name)
        
        # Plot Brain
        if emo in brain_results:
            axes[i].plot(brain_results[emo]["bins"], brain_results[emo]["density"], 
                        color='black', linewidth=3, linestyle='--', label='Brain')
        
        axes[i].set_title(f"Density Decay: {emo.capitalize()}")
        axes[i].grid(alpha=0.2)
        if i == 0: axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_emotion_density_comparison.png", dpi=150)
    plt.close()

    # 5. Save Metrics
    final_metrics = {
        "llm": llm_results,
        "brain": brain_results
    }
    with open(OUTPUT_DIR / "density_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\n✅ Analysis complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

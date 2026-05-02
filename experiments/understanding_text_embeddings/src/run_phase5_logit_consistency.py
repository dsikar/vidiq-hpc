import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from scipy.special import softmax

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase5"
VAL_CSV_PATH = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/data/Text_Datasets/20-emotions/Balanced_Data_split/validation.csv")
os.makedirs(REPORT_DIR, exist_ok=True)

def analyze_logit_consistency(name, X, y, logits, label_names, texts=None):
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    
    # 1. Compute Centroids
    centroids = np.array([np.mean(X[y == i], axis=0) for i in unique_labels])
    
    # 2. Compute distances to all centroids
    dists = cdist(X, centroids, metric='euclidean')
    own_dists = dists[np.arange(len(y)), y]
    
    # 3. Identify Overlapping Points
    # Definition: Closer to ANY wrong centroid than the true one
    overlap_points = []
    
    for i in range(len(X)):
        true_class = y[i]
        d_true = own_dists[i]
        
        # Find the closest WRONG centroid
        other_classes = [c for c in unique_labels if c != true_class]
        other_dists = dists[i, other_classes]
        min_other_idx = np.argmin(other_dists)
        d_other = other_dists[min_other_idx]
        closest_wrong_class = other_classes[min_other_idx]
        
        if d_other < d_true:
            entry = {
                "sample_idx": i,
                "true_class": int(true_class),
                "true_label": label_names[true_class],
                "closest_wrong_class": int(closest_wrong_class),
                "closest_wrong_label": label_names[closest_wrong_class],
                "d_true": float(d_true),
                "d_other": float(d_other),
                "dist_diff": float(d_other - d_true), # Negative if overlapping
                "logit_true": float(logits[i][true_class]),
                "logit_other": float(logits[i][closest_wrong_class]),
                "logit_diff": float(logits[i][true_class] - logits[i][closest_wrong_class])
            }
            if texts is not None:
                entry["text"] = texts[i]
            overlap_points.append(entry)
            
    # 4. Global Correlation (All Samples)
    # We look at: (d_other - d_true) vs (logit_true - logit_other)
    all_dist_diffs = []
    all_logit_diffs = []
    for i in range(len(X)):
        true_class = y[i]
        other_classes = [c for c in unique_labels if c != true_class]
        min_other_idx = np.argmin(dists[i, other_classes])
        closest_wrong_class = other_classes[min_other_idx]
        
        all_dist_diffs.append(dists[i, closest_wrong_class] - dists[i, true_class])
        all_logit_diffs.append(logits[i][true_class] - logits[i][closest_wrong_class])
        
    corr, p_val = pearsonr(all_dist_diffs, all_logit_diffs)
    
    # 5. Summarize
    df_overlap = pd.DataFrame(overlap_points)
    
    summary = {
        "model_name": name,
        "total_samples": len(X),
        "overlap_count": len(overlap_points),
        "overlap_pct": len(overlap_points) / len(X),
        "logit_agreement_rate": (df_overlap['logit_other'] > df_overlap['logit_true']).mean() if not df_overlap.empty else 0,
        "avg_logit_diff_overlap": df_overlap['logit_diff'].mean() if not df_overlap.empty else 0,
        "dist_logit_correlation": corr,
        "correlation_p_value": p_val
    }
    
    return summary, df_overlap, all_dist_diffs, all_logit_diffs

def plot_results(name, summary, df_overlap, all_dist_diffs, all_logit_diffs):
    # 1. Histogram of Logit Differences for Overlapping Points
    if not df_overlap.empty:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_overlap['logit_diff'], kde=True, color='red')
        plt.axvline(0, color='black', linestyle='--')
        plt.title(f"Logit Difference (True - Closer Wrong) for Overlapping Samples\n{name}")
        plt.xlabel("Logit Difference (Negative = Wrong Class Preferred)")
        plt.savefig(REPORT_DIR / f"logit_diff_hist_{name}.png")
        plt.close()

    # 2. Scatter Plot: Distance Diff vs Logit Diff (All samples)
    plt.figure(figsize=(10, 6))
    plt.scatter(all_dist_diffs, all_logit_diffs, alpha=0.3, s=10)
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title(f"Distance Gap vs Logit Gap\n{name} (Correlation: {summary['dist_logit_correlation']:.3f})")
    plt.xlabel("Geometric Margin (d_other - d_true)")
    plt.ylabel("Logit Margin (logit_true - logit_other)")
    plt.grid(True, alpha=0.2)
    plt.savefig(REPORT_DIR / f"dist_logit_scatter_{name}.png")
    plt.close()

def main():
    print("🚀 Starting Phase 5: Overlap–Logit Consistency Analysis...")
    datasets = load_all_text_datasets(split="val")
    
    # Load texts for verification
    val_texts = None
    if VAL_CSV_PATH.exists():
        val_df = pd.read_csv(VAL_CSV_PATH)
        val_texts = val_df['sentence'].tolist()
        print(f"  Loaded {len(val_texts)} validation sentences.")

    final_summaries = []
    
    for ds in datasets:
        logits = ds.get_logits()
        if logits is None:
            print(f"⚠️ Skipping {ds.name}: No logits found.")
            continue
            
        print(f"🔬 Analyzing {ds.name}...")
        X, y, label_names = ds.get_data()
        
        summary, df_overlap, all_dist_diffs, all_logit_diffs = analyze_logit_consistency(
            ds.name, X, y, logits, label_names, texts=val_texts
        )
        
        final_summaries.append(summary)
        
        # Save detailed overlap CSV
        df_overlap.to_csv(REPORT_DIR / f"overlap_details_{ds.name}.csv", index=False)
        
        # Plot
        plot_results(ds.name, summary, df_overlap, all_dist_diffs, all_logit_diffs)

    # Save all summaries
    with open(REPORT_DIR / "consistency_summary.json", "w") as f:
        json.dump(final_summaries, f, indent=4)
        
    print(f"\n✅ Phase 5 Complete. Results saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

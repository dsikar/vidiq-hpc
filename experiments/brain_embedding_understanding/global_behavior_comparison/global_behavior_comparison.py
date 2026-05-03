import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, ks_2samp

# --- PATHS ---
brain_dir = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/human_brain_emotion_exports"
llm_dir = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/embeddings_field/text/multiclass/pairwise"
output_dir = os.path.join(brain_dir, "global_behavior_comparison")
os.makedirs(output_dir, exist_ok=True)

# --- 1. LOAD BRAIN DATA ---
# Density Decay
brain_density_df = pd.read_csv(os.path.join(brain_dir, "human_density_decay_by_emotion_48D.csv"))
# Overlap vs Distance
brain_overlap_df = pd.read_csv(os.path.join(brain_dir, "human_overlap_decay_by_emotion_48D.csv"))
# Global summary for normalization
brain_summary = pd.read_csv(os.path.join(brain_dir, "human_brain_geometry_summary_48D.csv"))
mean_brain_radius = brain_summary['mean_point_to_own_centroid'].values[0]

# --- 2. LOAD LLM DATA (Aggregating from JSONs) ---
llm_density_curves = []
llm_overlap_curves = []
all_llm_radii = []

pair_folders = [d for d in os.listdir(llm_dir) if os.path.isdir(os.path.join(llm_dir, d)) and d != "mplconfig"]

for folder in pair_folders:
    m_path = os.path.join(llm_dir, folder, "metrics.json")
    if os.path.exists(m_path):
        with open(m_path, 'r') as f:
            data = json.load(f)
            # LLM metrics have 10 bins. We take the midpoint and density_per_unit
            for cluster in ['a_bins', 'b_bins']:
                mids = [b['midpoint'] for b in data[cluster]]
                densities = [b['density_per_unit'] for b in data[cluster]]
                overlaps = [b['overlap_ratio'] for b in data[cluster]]
                llm_density_curves.append((mids, densities))
                llm_overlap_curves.append((mids, overlaps))
                all_llm_radii.extend(mids)

mean_llm_radius = np.mean(all_llm_radii)

# --- 3. NORMALIZATION & INTERPOLATION ---
# We want to compare the systems on a "Normalized Radius" scale (R / Mean_Radius)
common_norm_r = np.linspace(0, 2.5, 100)

def get_avg_curve(curves, mean_r, common_r, is_density=True):
    interp_vals = []
    for r, v in curves:
        norm_r = np.array(r) / mean_r
        # If density, we normalize the Y values as well so the area under curve is comparable
        y = np.array(v)
        if is_density:
            y = y / np.max(y) if np.max(y) > 0 else y
        interp_vals.append(np.interp(common_r, norm_r, y))
    return np.mean(interp_vals, axis=0)

# Brain Curves
# For brain density decay: radius column is already there.
# We'll group by emotion and average
brain_emotions = brain_density_df['emotion'].unique()
brain_density_curves = []
for emo in brain_emotions:
    subset = brain_density_df[brain_density_df['emotion'] == emo]
    # We want density (change in proportion) rather than cumulative proportion
    # But the CSV has 'proportion_within_radius'. Let's use that as the signature.
    brain_density_curves.append((subset['radius'].values, subset['proportion_within_radius'].values))

avg_brain_density = get_avg_curve(brain_density_curves, mean_brain_radius, common_norm_r, is_density=False)
avg_llm_density = get_avg_curve(llm_density_curves, mean_llm_radius, common_norm_r, is_density=True)

# Overlap Curves
brain_overlap_curves = []
for emo in brain_overlap_df['true_emotion'].unique():
    subset = brain_overlap_df[brain_overlap_df['true_emotion'] == emo]
    brain_overlap_curves.append((subset['mean_distance'].values, subset['overlap_rate'].values))

avg_brain_overlap = get_avg_curve(brain_overlap_curves, mean_brain_radius, common_norm_r, is_density=False)
avg_llm_overlap = get_avg_curve(llm_overlap_curves, mean_llm_radius, common_norm_r, is_density=False)

# --- 4. PLOTTING ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Global Density Signature
axes[0].plot(common_norm_r, avg_brain_density, label='Human Brain', color='darkred', linewidth=3)
axes[0].plot(common_norm_r, avg_llm_density, label='Transformer LLM', color='darkblue', linewidth=3, linestyle='--')
axes[0].fill_between(common_norm_r, avg_brain_density, color='red', alpha=0.1)
axes[0].fill_between(common_norm_r, avg_llm_density, color='blue', alpha=0.1)
axes[0].set_title("Global Density Profile (All Emotions)", fontsize=14)
axes[0].set_xlabel("Normalized Distance (R / Mean Radius)")
axes[0].set_ylabel("Normalized Density / Proportion")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Ambiguity Gradient (Overlap Growth)
axes[1].plot(common_norm_r, avg_brain_overlap, label='Human Brain', color='darkred', linewidth=3)
axes[1].plot(common_norm_r, avg_llm_overlap, label='Transformer LLM', color='darkblue', linewidth=3, linestyle='--')
axes[1].set_title("Ambiguity Gradient (Overlap vs Distance)", fontsize=14)
axes[1].set_xlabel("Normalized Distance")
axes[1].set_ylabel("Overlap Ratio (Probability of Ambiguity)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "global_behavior_comparison.png"), dpi=200)

# --- 5. STATISTICAL VALIDATION ---
# Correlation of overlap gradients
corr_overlap, p_overlap = pearsonr(avg_brain_overlap, avg_llm_overlap)
# KS Test for density distributions
ks_stat, ks_p = ks_2samp(avg_brain_density, avg_llm_density)

with open(os.path.join(output_dir, "comparison_results.txt"), "w") as f:
    f.write("CROSS-SYSTEM EMOTION GEOMETRY: GLOBAL COMPARISON\n")
    f.write("==============================================\n\n")
    f.write(f"1. AMBIGUITY GRADIENT SIMILARITY\n")
    f.write(f"   Pearson Correlation: {corr_overlap:.4f}\n")
    f.write(f"   P-value: {p_overlap:.4e}\n")
    f.write("   Insight: High correlation indicates both systems represent 'certainty' similarly relative to the category center.\n\n")
    f.write(f"2. DENSITY DISTRIBUTION SIMILARITY\n")
    f.write(f"   KS Statistic: {ks_stat:.4f}\n")
    f.write(f"   KS P-value: {ks_p:.4f}\n")
    if ks_p > 0.05:
        f.write("   Insight: No significant difference in how points are packed around the emotional core.\n")
    else:
        f.write("   Insight: Systems differ in their precise packing density, though trends may be similar.\n")

print(f"Global comparison complete. Results in {output_dir}")

import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.spatial.distance import cdist

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase2"
VIS_DIR = REPORT_DIR / "visuals"
os.makedirs(VIS_DIR, exist_ok=True)

def calculate_overlap_metrics(X, y, label_names):
    unique_labels = np.unique(y)
    n_classes = len(unique_labels)
    
    # 1. Calculate Centroids
    centroids = np.array([np.mean(X[y == i], axis=0) for i in unique_labels])
    
    # 2. Calculate Distance Matrix (Samples x Centroids)
    # Using Euclidean distance as it's standard for "closest to" logic
    dists = cdist(X, centroids, metric='euclidean')
    
    # 3. Determine Overlaps
    # An overlap occurs if the closest centroid is not the true label
    closest_centroid = np.argmin(dists, axis=1)
    is_overlap = (closest_centroid != y)
    
    # 4. Normalized Radial Distance
    # Distance to OWN centroid / mean distance of own class
    own_dists = dists[np.arange(len(y)), y]
    norm_radial_dists = np.zeros_like(own_dists)
    for i in unique_labels:
        mask = (y == i)
        mean_r = np.mean(own_dists[mask])
        norm_radial_dists[mask] = own_dists[mask] / (mean_r + 1e-10)
        
    # 5. Binning for Density and Ambiguity Gradient
    bins = np.linspace(0, 2.5, 21) # 0 to 2.5 radii
    bin_mids = (bins[:-1] + bins[1:]) / 2
    
    density_curve = []
    ambiguity_gradient = []
    
    for i in range(len(bins)-1):
        bin_mask = (norm_radial_dists >= bins[i]) & (norm_radial_dists < bins[i+1])
        if np.sum(bin_mask) > 0:
            density_curve.append(np.sum(bin_mask) / len(y))
            ambiguity_gradient.append(np.mean(is_overlap[bin_mask]))
        else:
            density_curve.append(0.0)
            ambiguity_gradient.append(None) # No data in bin
            
    # 6. Pairwise Overlap Matrix
    overlap_matrix = np.zeros((n_classes, n_classes))
    for i in unique_labels:
        for j in unique_labels:
            if i == j: continue
            # Points belonging to i that are closer to centroid j
            mask_i = (y == i)
            # Find points where j is closer than i
            overlap_count = np.sum(dists[mask_i, j] < dists[mask_i, i])
            overlap_matrix[i, j] = overlap_count / np.sum(mask_i)
            
    return {
        "bin_mids": bin_mids.tolist(),
        "density": density_curve,
        "ambiguity": ambiguity_gradient,
        "overlap_matrix": overlap_matrix.tolist()
    }

def plot_variant_summary(name, metrics, label_names):
    # 1. Plot Overlap Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics['overlap_matrix'], annot=True, fmt=".1%", 
                xticklabels=label_names, yticklabels=label_names, cmap="YlOrRd")
    plt.title(f"Geometric Overlap Matrix: {name}\n(% of row closer to col centroid)")
    plt.tight_layout()
    plt.savefig(VIS_DIR / f"overlap_heatmap_{name}.png")
    plt.close()

def main():
    print("🚀 Starting Phase 2: Overlap & Radial Density Analysis...")
    datasets = load_all_text_datasets(split="val")
    
    all_results = {}
    
    # Setup global curves plot
    fig_global, axes_global = plt.subplots(2, 1, figsize=(12, 12))
    
    for ds in datasets:
        print(f"\n🔬 Analyzing Manifold: {ds.name}")
        X, y, label_names = ds.get_data()
        
        metrics = calculate_overlap_metrics(X, y, label_names)
        all_results[ds.name] = metrics
        
        # Plot individual heatmaps
        plot_variant_summary(ds.name, metrics, label_names)
        
        # Add to global curves
        axes_global[0].plot(metrics['bin_mids'], metrics['density'], label=ds.name, linewidth=2)
        axes_global[1].plot(metrics['bin_mids'], metrics['ambiguity'], label=ds.name, marker='o', markersize=4)

    # Finalize Global Curves
    axes_global[0].set_title("Radial Density Decay (Normalized Radius)", fontsize=14)
    axes_global[0].set_ylabel("Population Density")
    axes_global[0].grid(alpha=0.3)
    axes_global[0].legend()

    axes_global[1].set_title("Ambiguity Gradient (% Overlap vs. Distance)", fontsize=14)
    axes_global[1].set_ylabel("Overlap Probability")
    axes_global[1].set_xlabel("Distance from Centroid (Units of Mean Radius)")
    axes_global[1].grid(alpha=0.3)
    axes_global[1].legend()

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "global_density_ambiguity_curves.png", dpi=150)
    plt.close()

    # Save metrics
    with open(REPORT_DIR / "overlap_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n✅ Phase 2 Complete. Reports saved to {REPORT_DIR}")

if __name__ == "__main__":
    main()

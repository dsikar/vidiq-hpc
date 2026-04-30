import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Reuse loader
import sys
sys.path.append("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings/src")
from loader_brain import load_brain_data

# --- CONFIG ---
EXP_ROOT = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/brain_embedding_understanding/checking_density_geometry")
OUTPUT_DIR = EXP_ROOT / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    print("🧠 Analyzing Brain-Specific Density Geometry...")
    
    # 1. Load Data
    brain_csv = "/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/human_brain_emotion_exports/human_subject_emotion_roi_48D_scaled.csv"
    _, labels, df = load_brain_data(brain_csv)
    
    # Extract ROI columns (everything except subject and emotion)
    roi_cols = [c for c in df.columns if c not in ['subject', 'emotion']]
    X = df[roi_cols].values
    y = df['emotion'].values
    
    # 2. Calculate Distances per Emotion
    data_list = []
    unique_emotions = np.unique(y)
    
    plt.figure(figsize=(12, 7))
    
    for emo in unique_emotions:
        mask = (y == emo)
        pts = X[mask]
        
        centroid = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)
        
        # Normalization: Mean Radius of this emotion = 1.0
        norm_dists = dists / np.mean(dists)
        
        # Store for violin plot
        for d in norm_dists:
            data_list.append({"Emotion": emo, "Norm_Distance": d})
            
        # Plot Density Curve (KDE)
        sns.kdeplot(norm_dists, label=emo, linewidth=2)

    # 3. Finalize Density Plot
    plt.title("Neural Density Decay: Human Brain ROIs\n(Normalized Distance to Emotion Centroids)", fontsize=14)
    plt.xlabel("Distance from Centroid (Units of Mean Radius)")
    plt.ylabel("Density")
    plt.axvline(x=1.0, color='grey', linestyle=':', label='Mean Radius')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 3)
    plt.savefig(OUTPUT_DIR / "brain_only_density_decay.png", dpi=150)
    plt.close()

    # 4. Generate Spread Analysis (Violin Plot)
    plt.figure(figsize=(14, 6))
    df_dist = pd.DataFrame(data_list)
    sns.violinplot(data=df_dist, x="Emotion", y="Norm_Distance", palette="muted", inner="quartile")
    plt.title("Emotional Point Spread: Distance to Centroid per ROI Cluster", fontsize=14)
    plt.ylabel("Norm. Distance")
    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Mean Spread')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUTPUT_DIR / "brain_emotion_spread_violin.png", dpi=150)
    plt.close()

    print(f"\n✅ Brain analysis complete. Plots saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

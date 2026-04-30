import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIG ---
ROOT = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings")
METRICS_PATH = ROOT / "reports/phase3/retention_metrics_top200_directions.json"
REPORT_DIR = ROOT / "reports/phase3"

def find_chance_crossing(accs, threshold=0.17):
    for i, a in enumerate(accs):
        if a <= threshold:
            return i
    return len(accs) - 1

def plot_individual_convergences():
    print("📈 Generating Individual Chance Convergence Plots (SVD)...")
    
    with open(METRICS_PATH, "r") as f:
        all_data = json.load(f)
        
    chance_level = 1.0 / 6
    
    for name, data in all_data.items():
        accs = data["accuracies"] # Extract the accuracy list
        
        plt.figure(figsize=(10, 6))
        
        # 1. Find crossing point
        cross_idx = find_chance_crossing(accs)
        
        # 2. Plot Full Curve
        plt.plot(range(len(accs)), accs, color='darkorange', linewidth=2, label='Classification Accuracy')
        
        # 3. Mark Crossing
        plt.scatter(cross_idx, accs[cross_idx], color='red', s=100, zorder=5, label=f'Chance Crossing (Dim {cross_idx})')
        plt.annotate(f"Erased at Dim {cross_idx}", (cross_idx, accs[cross_idx]), xytext=(15, 15), 
                     textcoords='offset points', arrowprops=dict(arrowstyle="->", color='red'))
        
        plt.axhline(y=chance_level, color='grey', linestyle='--', alpha=0.5, label='Chance Baseline')
        
        plt.title(f"SVD Signal Erasure Profile: {name}", fontsize=14)
        plt.xlabel("Number of Top Directions Removed")
        plt.ylabel("Accuracy")
        plt.grid(alpha=0.2)
        plt.legend()
        plt.tight_layout()
        
        plt.savefig(REPORT_DIR / f"erasure_profile_{name}.png", dpi=150)
        plt.close()
        
    print(f"✅ Erasure profiles saved to {REPORT_DIR}")

if __name__ == "__main__":
    plot_individual_convergences()

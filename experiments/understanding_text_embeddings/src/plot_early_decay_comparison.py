import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# --- CONFIG ---
ROOT = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings")
METRICS_PATH = ROOT / "reports/phase3/importance_metrics.json"

def plot_early_decay():
    print("📈 Generating Combined Early Decay Plot (First 25 Dimensions)...")
    
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
        
    plt.figure(figsize=(14, 8))
    
    for name, data in metrics.items():
        if "Mid" in name: continue # Focus only on Final layers
        
        # Extend to 25 dimensions (index 26)
        accs = data["top_100_accuracy"][:26] 
        
        # Determine line style based on variant
        linestyle = '-'
        if "Base" in name:
            linestyle = '--'
            
        plt.plot(range(len(accs)), accs, label=name, marker='o', markersize=4, linewidth=2, linestyle=linestyle)

    plt.axhline(y=0.1666, color='grey', linestyle=':', label='Chance (16.7%)')
    plt.title("The 'Signal Cliff': Accuracy Decay over First 25 Dimensions Removed", fontsize=15)
    plt.xlabel("Number of Dominant Directions Removed", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.xticks(range(26))
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(ROOT / "reports/phase3/early_decay_comparison_top25.png", dpi=200)
    plt.close()
    print(f"✅ Combined early decay plot saved to reports/phase3/")

if __name__ == "__main__":
    plot_early_decay()

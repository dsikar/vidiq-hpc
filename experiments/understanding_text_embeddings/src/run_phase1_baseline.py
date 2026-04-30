import numpy as np
import json
import os
from pathlib import Path
from sklearn.metrics import silhouette_score
from tqdm import tqdm

# Import the custom loader
import sys
sys.path.append(str(Path(__file__).parent))
from loader_text import load_all_text_datasets

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase1"
os.makedirs(REPORT_DIR, exist_ok=True)

def run_baseline_clustering():
    print("🚀 Starting Phase 1: Baseline Topology (Silhouette only)...")
    
    # Load all 8 variants (Validation split)
    datasets = load_all_text_datasets(split="val")
    
    results = {}
    
    for ds in datasets:
        print(f"\n📊 Analyzing: {ds.name}")
        X, y, labels = ds.get_data()
        
        # 1. Clustering: Silhouette Score
        print("   -> Calculating Silhouette Score...")
        sil_score = silhouette_score(X, y)
        
        print(f"      Result: Silhouette={sil_score:.4f}")
        
        results[ds.name] = {
            "silhouette": float(sil_score)
        }

    # Save to JSON
    with open(REPORT_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Phase 1 Clustering Complete. Results saved to {REPORT_DIR}")

if __name__ == "__main__":
    run_baseline_clustering()

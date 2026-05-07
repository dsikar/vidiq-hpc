import numpy as np
import pandas as pd
import json
import os
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import procrustes
from pathlib import Path

# --- CONFIG ---
EXP_ROOT = Path(__file__).resolve().parents[1]
RDM_PATH = EXP_ROOT / "reports/rdm_matrices.json"
OUTPUT_DIR = EXP_ROOT / "reports"

def load_3class_rdm(system_data, target_labels, is_brain=False):
    # Extract the cosine distance values for the 3 target labels
    # We want a 1D vector of the 3 unique pairwise distances: 
    # (Fear-Happiness, Fear-Sadness, Happiness-Sadness)
    
    # Standardize keys to lowercase
    data = {k.lower(): {ik.lower(): iv for ik, iv in v.items()} for k, v in system_data['cosine'].items()}
    
    if is_brain:
        # Brain uses: afraid, calm, delighted, depressed, excited
        # We must map these back to our target 3
        # afraid -> fear
        # calm/delighted/excited -> happiness
        # depressed -> sadness
        
        f_h = data['afraid']['calm'] # representative of fear-happiness
        f_s = data['afraid']['depressed'] # representative of fear-sadness
        h_s = data['calm']['depressed'] # representative of happiness-sadness
        return np.array([f_h, f_s, h_s])
    else:
        # LLMs already use the target labels
        distances = [
            data['fear']['happiness'],
            data['fear']['sadness'],
            data['happiness']['sadness']
        ]
        return np.array(distances)

def run_comparison():
    print("📊 Comparing 3-Emotion Relational Numbers (Brain vs LLMs)...")
    
    with open(RDM_PATH, "r") as f:
        rdm_data = json.load(f)
    
    target_labels = ["fear", "happiness", "sadness"]
    
    # 1. Extract Vectors
    v_brain = load_3class_rdm(rdm_data['Brain-fMRI'], target_labels, is_brain=True)
    v_qwen  = load_3class_rdm(rdm_data['Qwen-768'], target_labels)
    v_mpnet = load_3class_rdm(rdm_data['MPNet-Balanced'], target_labels)
    
    # 2. Comparison Metrics
    def compare(v1, v2, name):
        # Pearson (Linear Similarity)
        r, p_r = pearsonr(v1, v2)
        # Spearman (Rank Similarity)
        rho, p_s = spearmanr(v1, v2)
        # Normalized Distance (Overall Error)
        # We normalize by the norm of the vectors to make them comparable
        v1_n = v1 / np.linalg.norm(v1)
        v2_n = v2 / np.linalg.norm(v2)
        error = np.linalg.norm(v1_n - v2_n)
        
        return {
            "pearson_r": float(r),
            "spearman_rho": float(rho),
            "normalized_error": float(error)
        }

    res_qwen = compare(v_brain, v_qwen, "Qwen")
    res_mpnet = compare(v_brain, v_mpnet, "MPNet")

    # 3. Format Results
    final_results = {
        "emotion_triplet": target_labels,
        "pairwise_order": ["Fear-Happiness", "Fear-Sadness", "Happiness-Sadness"],
        "raw_vectors": {
            "brain": v_brain.tolist(),
            "qwen": v_qwen.tolist(),
            "mpnet": v_mpnet.tolist()
        },
        "comparisons": {
            "brain_vs_qwen": res_qwen,
            "brain_vs_mpnet": res_mpnet
        }
    }

    print("\n" + "="*45)
    print("CROSS-SYSTEM NUMERICAL COMPARISON (with Brain)")
    print("="*45)
    print(f"Brain Vectors: {v_brain}")
    print("-" * 45)
    print(f"QWEN-768:")
    print(f"   Pearson r:  {res_qwen['pearson_r']:.4f}")
    print(f"   RSA (Rho):  {res_qwen['spearman_rho']:.4f}")
    print(f"   Error:      {res_qwen['normalized_error']:.4f}")
    print("-" * 45)
    print(f"MPNET:")
    print(f"   Pearson r:  {res_mpnet['pearson_r']:.4f}")
    print(f"   RSA (Rho):  {res_mpnet['spearman_rho']:.4f}")
    print(f"   Error:      {res_mpnet['normalized_error']:.4f}")
    print("="*45)

    with open(OUTPUT_DIR / "rdm_comparison_metrics.json", "w") as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    run_comparison()

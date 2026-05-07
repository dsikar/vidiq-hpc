import numpy as np
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
from scipy.integrate import simpson

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parents[4]
TEXT_EXPERIMENT_ROOT = REPO_ROOT / "experiments/understanding_text_embeddings"
METRICS_SOURCE = Path(
    os.environ.get(
        "RETENTION_METRICS_SOURCE",
        str(TEXT_EXPERIMENT_ROOT / "reports/phase3/retention_metrics_top200_directions.json"),
    )
)
EXP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_ROOT / "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def unpack_metrics(metrics):
    accuracies = metrics["accuracies"]
    dims = metrics.get("dims_removed", list(range(len(accuracies))))
    chance = metrics.get("chance_level", 1.0 / 6.0)
    return dims, accuracies, chance

def calculate_comparison_metrics(name, dims, accs, chance):
    # 1. AUC (Signal Volume)
    # Higher AUC = More Distributed/Robust
    auc = simpson(accs, x=dims) / (dims[-1] if dims[-1] > 0 else 1)
    
    # 2. Signal Half-Life (D50)
    # Number of dims removed before losing half the decodability
    baseline = accs[0]
    half_signal = baseline - (baseline - chance) / 2
    d50 = 0
    for d, a in zip(dims, accs):
        if a <= half_signal:
            d50 = d
            break
    else:
        d50 = dims[-1]

    # 3. Cliff Slope (Initial Compression)
    # Decay rate over first 10 dimensions
    limit = min(10, len(accs)-1)
    cliff = (accs[0] - accs[limit]) / limit if limit > 0 else 0
    
    return {
        "name": name,
        "auc_signal_volume": float(auc),
        "signal_half_life_d50": int(d50),
        "cliff_slope": float(cliff),
        "baseline_accuracy": float(baseline)
    }

def main():
    print("📈 Loading existing high-res metrics...")
    with open(METRICS_SOURCE, "r") as f:
        data = json.load(f)
    
    comparison_results = []
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 7))
    
    for name, metrics in data.items():
        dims, accs, chance = unpack_metrics(metrics)
        
        # Calculate comparison numbers
        comp = calculate_comparison_metrics(name, dims, accs, chance)
        comparison_results.append(comp)
        
        # Plot
        plt.plot(dims, accs, label=f"{name} (D50: {comp['signal_half_life_d50']})", linewidth=2.5)
        plt.axhline(y=chance, color='grey', linestyle='--', alpha=0.3)

    plt.title("Recursive Context Retention: Comparison of Manifold Decay", fontsize=14)
    plt.xlabel("Number of Dominant Dimensions Removed", fontsize=12)
    plt.ylabel("Classification Accuracy", fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "context_retention_comparison.png", dpi=200)
    plt.close()

    # --- FINAL REPORT GENERATION ---
    with open(OUTPUT_DIR / "retention_comparison_metrics.json", "w") as f:
        json.dump(comparison_results, f, indent=4)
        
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Retention Comparison: Brain vs LLM</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 40px; background: #f8f9fa; color: #333; }}
            .container {{ max-width: 1000px; margin: auto; }}
            .metric-table {{ width: 100%; border-collapse: collapse; background: white; margin: 20px 0; }}
            .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            .metric-table th {{ background-color: #2c3e50; color: white; }}
            .plot-card {{ text-align: center; background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }}
            .plot-card img {{ width: 100%; height: auto; }}
            .badge {{ padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
            .robust {{ background: #d4edda; color: #155724; }}
            .fragile {{ background: #f8d7da; color: #721c24; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Recursive Context Retention Comparison</h1>
            <p>This report quantifies how robustly emotional context is retained when the most important linear directions are surgically removed.</p>
            
            <div class="plot-card">
                <img src="context_retention_comparison.png">
            </div>

            <h2>1. Quantitative Comparison Metrics</h2>
            <table class="metric-table">
                <thead>
                    <tr>
                        <th>System</th>
                        <th>Signal Volume (AUC)</th>
                        <th>Half-Life (D50)</th>
                        <th>Cliff Slope</th>
                        <th>Retention Status</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for r in comparison_results:
        status = "Fragile (Compressed)" if r['cliff_slope'] > 0.03 else "Robust (Distributed)"
        badge_class = "fragile" if r['cliff_slope'] > 0.03 else "robust"
        
        html += f"""
                <tr>
                    <td><strong>{r['name']}</strong></td>
                    <td>{r['auc_signal_volume']:.4f}</td>
                    <td>{r['signal_half_life_d50']} dims</td>
                    <td>{r['cliff_slope']:.4f}</td>
                    <td><span class="badge {badge_class}">{status}</span></td>
                </tr>
        """
        
    html += """
                </tbody>
            </table>

            <h2>2. Interpretation of Metrics</h2>
            <ul>
                <li><strong>Signal Volume (AUC):</strong> Measures the "redundancy" of the signal. The higher the AUC, the more dimensions contain usable emotional context.</li>
                <li><strong>Signal Half-Life (D50):</strong> The "Shelf Life" of the context. A low D50 (like Qwen) means the system depends on a few specific dimensions. A high D50 (like Brain/MPNet) means information is distributed.</li>
                <li><strong>Cliff Slope:</strong> Measures the immediate impact of losing the most important feature. </li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(OUTPUT_DIR / "retention_comparison_report.html", "w") as f:
        f.write(html)
    
    print(f"\n✅ Analysis complete. Final report saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

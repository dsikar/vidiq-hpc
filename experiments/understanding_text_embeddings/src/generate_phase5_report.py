import json
import os
import pandas as pd
from pathlib import Path

# --- CONFIG ---
EXP_ROOT = Path(__file__).parent.parent
REPORT_DIR = EXP_ROOT / "reports/phase5"
OUTPUT_HTML = EXP_ROOT / "reports/phase5_summary.html"

def get_sample_table_html(model_name):
    csv_path = REPORT_DIR / f"overlap_details_{model_name}.csv"
    if not csv_path.exists():
        return "<p>No overlap data found.</p>"
    
    df = pd.read_csv(csv_path)
    if df.empty:
        return "<p>No overlapping samples identified for this model.</p>"
    
    # Pick 15 diverse examples (2-3 per emotion)
    samples = df.groupby('true_label').head(3).head(15)
    
    html = f"""
    <table class="summary-table">
        <thead>
            <tr>
                <th>Index</th>
                <th>True Label</th>
                <th>Closer (Wrong)</th>
                <th>D_True</th>
                <th>D_Wrong</th>
                <th>L_True</th>
                <th>L_Wrong</th>
                <th>Sentence</th>
            </tr>
        </thead>
        <tbody>
    """
    for _, r in samples.iterrows():
        text_val = r.get('text', 'N/A')
        html += f"""
            <tr>
                <td>{int(r['sample_idx'])}</td>
                <td>{r['true_label']}</td>
                <td>{r['closest_wrong_label']}</td>
                <td>{r['d_true']:.4f}</td>
                <td>{r['d_other']:.4f}</td>
                <td>{r['logit_true']:.4f}</td>
                <td>{r['logit_other']:.4f}</td>
                <td style="font-size: 0.85em; max-width: 400px;">{text_val}</td>
            </tr>
        """
    html += "</tbody></table>"
    return html

def generate_html(summaries):
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 5: Overlap–Logit Consistency</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 40px; background: #f4f7f6; color: #333; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: auto; }}
            h1 {{ border-bottom: 4px solid #8e44ad; padding-bottom: 10px; color: #2c3e50; }}
            h2 {{ margin-top: 50px; color: #8e44ad; border-left: 5px solid #8e44ad; padding-left: 15px; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            .summary-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: white; font-size: 0.9em; }}
            .summary-table th, .summary-table td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
            .summary-table th {{ background-color: #f1f3f5; }}
            .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; margin-top: 20px; }}
            .plot-card {{ background: white; padding: 15px; border-radius: 10px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
            .plot-card img {{ width: 100%; height: auto; border-radius: 6px; }}
            .metric {{ font-weight: bold; color: #8e44ad; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 5: Overlap–Logit Consistency (Geometry vs. Logic)</h1>
            
            <div class="card">
                <p>This phase investigates if the model's output probabilities (logits) follow the same Euclidean logic as the embedding space. We specifically look at samples that are geometrically closer to a <em>wrong</em> class centroid.</p>
            </div>

            <h2>1. Global Consistency Summary</h2>
            <div class="card">
                <table class="summary-table">
                    <thead>
                        <tr>
                            <th>Model Variant</th>
                            <th>Overlap Count</th>
                            <th>Overlap %</th>
                            <th>Logit Agreement Rate*</th>
                            <th>Dist-Logit Correlation</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for s in summaries:
        html += f"""
                <tr>
                    <td><strong>{s['model_name']}</strong></td>
                    <td>{s['overlap_count']}</td>
                    <td>{s['overlap_pct']:.2%}</td>
                    <td>{s['logit_agreement_rate']:.2%}</td>
                    <td>{s['dist_logit_correlation']:.3f}</td>
                </tr>
        """
        
    html += """
                    </tbody>
                </table>
                <p><small>*Percentage of overlapping samples where the logit of the closer (wrong) class is actually higher than the logit of the true class.</small></p>
            </div>

            <h2>2. Representative Overlapping Samples</h2>
            <p>Examples where the sample is geometrically closer to an incorrect centroid. Note how the logits (L_Wrong) typically follow the geometric bias.</p>
    """

    for s in summaries:
        name = s['model_name']
        html += f"""
            <div class="card">
                <h3>{name} - Sample Details</h3>
                {get_sample_table_html(name)}
            </div>
        """

    html += """
            <h2>3. The Geometry-Logit Relationship</h2>
            <div class="plot-grid">
    """
    
    for s in summaries:
        name = s['model_name']
        html += f"""
                <div class="plot-card">
                    <h3>{name}: Distance vs. Logit Gap</h3>
                    <img src="phase5/dist_logit_scatter_{name}.png" alt="Scatter {name}">
                    <p>Correlating geometric margin with logit margin.</p>
                </div>
                <div class="plot-card">
                    <h3>{name}: Logit Bias in Overlap Zone</h3>
                    <img src="phase5/logit_diff_hist_{name}.png" alt="Hist {name}">
                    <p>Distribution of (Logit_True - Logit_Wrong) for overlapping samples.</p>
                </div>
        """
        
    html += """
            </div>

            <h2>4. Key Findings</h2>
            <div class="card">
                <ul>
                    <li><strong>Geometric Determinism:</strong> High correlation between distance and logits confirms the model's decision boundary is tightly coupled to embedding geometry.</li>
                    <li><strong>Robustness:</strong> If the logit agreement rate is low despite overlap, it suggests the model's final classification layer compensates for geometric "crowding" through its learned weights.</li>
                    <li><strong>Error Prone Zones:</strong> Overlapping samples with high logit agreement (negative diff) represent the "blind spots" where Euclidean distance perfectly predicts model error.</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    return html

def main():
    with open(REPORT_DIR / "consistency_summary.json", "r") as f:
        summaries = json.load(f)
        
    html_content = generate_html(summaries)
    
    with open(OUTPUT_HTML, "w") as f:
        f.write(html_content)
        
    print(f"✅ Phase 5 Report generated at {OUTPUT_HTML}")

if __name__ == "__main__":
    main()

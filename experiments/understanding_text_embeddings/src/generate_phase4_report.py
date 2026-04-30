import json
from pathlib import Path
import numpy as np

def get_heatmap_color(val):
    """Returns a background color based on distance (Darker for distant, Lighter for close)"""
    # Normalized val expected to be around 0.5 to 2.0
    # Higher val = more distant = Darker Blue
    # Lower val = closer = Lighter/White
    factor = max(0, min(1, (val - 0.4) / 1.6))
    
    # For a Blue heatmap:
    # 0 distance (factor 0) -> rgb(255, 255, 255) (White)
    # Max distance (factor 1) -> rgb(30, 50, 150) (Dark Blue)
    r = int(255 - factor * 225)
    g = int(255 - factor * 205)
    b = 255 # Keep blue channel high
    
    return f"rgb({r}, {g}, {b})"

def generate_phase4_report():
    root = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings")
    metrics_path = root / "reports/phase4/subspace_metrics_20D.json"
    dist_path = root / "reports/phase4/centroid_distances_20D.json"
    
    if not metrics_path.exists() or not dist_path.exists():
        return

    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(dist_path, "r") as f:
        dist_data = json.load(f)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 4: Isolated Subspace Analysis</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 20px; background: #fdfdfd; color: #333; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: auto; }}
            h1 {{ border-bottom: 4px solid #27ae60; padding-bottom: 10px; color: #2c3e50; font-size: 2em; }}
            h2 {{ margin-top: 40px; color: #27ae60; border-left: 5px solid #27ae60; padding-left: 15px; font-size: 1.5em; }}
            h3 {{ margin-top: 30px; color: #444; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            
            .summary-table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; font-size: 0.85em; }}
            .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            .summary-table th {{ background-color: #f8f9fa; font-weight: bold; }}
            .row-label {{ text-align: left !important; font-weight: bold; background: #f8f9fa; width: 100px; }}
            .inc {{ color: #27ae60; font-weight: bold; }}
            
            .dist-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(550px, 1fr)); gap: 25px; margin-top: 20px; }}
            .comparison-grid {{ display: flex; flex-direction: column; gap: 40px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 4: Isolated Subspace Analysis (The "Top 20" Test)</h1>
            <p>Evaluating the efficiency and relational logic of the emotional core (Isolated 20D Subspace).</p>

            <h2>1. Clustering Quality & Signal Retention</h2>
            <div class="card">
                <img src="phase4/silhouette_comparison_20D.png" style="max-width:800px; display:block; margin:auto;" alt="Silhouette Comparison">
                <table class="summary-table" style="margin-top:20px; font-size: 1em;">
                    <thead>
                        <tr>
                            <th>Variant</th>
                            <th>768D Sil</th>
                            <th>20D Sil</th>
                            <th>Jump</th>
                            <th>20D Accuracy</th>
                        </tr>
                    </thead>
                    <tbody>
    """
    
    for name, data in metrics.items():
        sil_768 = data['baseline_768D'].get('silhouette', 0.001)
        sil_20D = data['isolated_20D']['silhouette_20D']
        acc_20D = data['isolated_20D']['accuracy_20D']
        improvement = (sil_20D / sil_768)
        
        html += f"""
                <tr>
                    <td class="row-label">{name}</td>
                    <td>{sil_768:.4f}</td>
                    <td class="inc">{sil_20D:.4f}</td>
                    <td>{improvement:.1f}x</td>
                    <td>{acc_20D:.2%}</td>
                </tr>
        """
        
    html += """
                    </tbody>
                </table>
            </div>

            <h2>2. Relational Heatmaps: Pairwise Centroid Distances (20D)</h2>
            <p><strong>Heatmap Logic (Inverted):</strong> <span style='color:#1e3296; font-weight:bold;'>Dark Blue</span> indicates large geometric distance (dissimilar). <span style='color:#eee;'>White</span> indicates close proximity (similar).</p>
            <div class="dist-grid">
    """

    for name, data in dist_data.items():
        labels = data['labels']
        matrix = np.array(data['norm_matrix'])
        
        html += f"""
                <div class="card">
                    <h3>{name}</h3>
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th></th>
                                {"".join([f"<th>{l[:3].capitalize()}</th>" for l in labels])}
                            </tr>
                        </thead>
                        <tbody>
        """
        for i, row_label in enumerate(labels):
            html += f"<tr><td class='row-label'>{row_label.capitalize()}</td>"
            for j, val in enumerate(matrix[i]):
                color = get_heatmap_color(val)
                # Diagonal is 0 distance (perfectly similar), so it should be white
                bg = "#fff" if i == j else color
                # If background is dark, make text white
                text_color = "white" if (val > 1.3) else "black"
                html += f"<td style='background-color: {bg}; color: {text_color};'>{val:.2f}</td>"
            html += "</tr>"
            
        html += """
                        </tbody>
                    </table>
                </div>
        """

    html += """
            </div>

            <h2>3. Visual Denoising Gallery</h2>
            <div class="comparison-grid">
    """

    for name in metrics.keys():
        html += f"""
                <div class="card">
                    <h3>{name}: Side-by-Side (768D vs 20D)</h3>
                    <img src="phase4/visuals/comparison_scatter_{name}.png" alt="Scatter Comparison">
                </div>
        """

    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    output_path = root / "reports/phase4_summary.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"✅ Phase 4 Dashboard finalized with Inverted Heatmaps at: {output_path}")

if __name__ == "__main__":
    generate_phase4_report()

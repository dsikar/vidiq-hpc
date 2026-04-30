import json
from pathlib import Path
import numpy as np

def get_heatmap_color(val):
    factor = max(0, min(1, (val - 0.4) / 1.6))
    r = int(255 - factor * 225)
    g = int(255 - factor * 205)
    b = 255
    return f"rgb({r}, {g}, {b})"

def update_html_with_visuals():
    root = Path("/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq-hpc/experiments/understanding_text_embeddings")
    metrics_path = root / "reports/phase1/baseline_metrics.json"
    dist_path = root / "reports/phase1/centroid_distances_768D.json"
    
    with open(metrics_path, "r") as f:
        results = json.load(f)
    
    dist_data = {}
    if dist_path.exists():
        with open(dist_path, "r") as f:
            dist_data = json.load(f)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phase 1: Baseline Emotion Geometry</title>
        <style>
            body {{ font-family: -apple-system, sans-serif; padding: 20px; background: #f8f9fa; color: #333; line-height: 1.6; }}
            .container {{ max-width: 1200px; margin: auto; }}
            h1 {{ border-bottom: 3px solid #333; padding-bottom: 10px; color: #2c3e50; }}
            h2 {{ margin-top: 40px; color: #2980b9; border-left: 5px solid #2980b9; padding-left: 15px; }}
            .card {{ background: white; padding: 20px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 30px; }}
            .summary-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background: white; font-size: 0.9em; }}
            .summary-table th, .summary-table td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            .summary-table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .row-label {{ text-align: left !important; font-weight: bold; background: #f2f2f2; width: 100px; }}
            .high {{ color: #27ae60; font-weight: bold; }}
            .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 20px; margin-top: 20px; }}
            .plot-card {{ background: white; padding: 15px; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); text-align: center; }}
            img {{ max-width: 100%; height: auto; border-radius: 8px; }}
            .dist-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 25px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Phase 1: Baseline Topology & Relational Logic</h1>
            <p>Evaluation of 8 embedding variants focusing on cluster purity and the raw semantic distances between emotion centroids in the full 768D space.</p>

            <h2>1. Quantitative Clustering Metrics</h2>
            <table class="summary-table">
                <thead>
                    <tr>
                        <th>Variant</th>
                        <th>Silhouette Score (768D)</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for name, m in results.items():
        html += f"""
                <tr>
                    <td class='row-label'>{name}</td>
                    <td class="{'high' if m['silhouette'] > 0.4 else ''}">{m['silhouette']:.4f}</td>
                </tr>
        """
        
    html += f"""
                </tbody>
            </table>

            <h2>2. Relational Heatmaps: Full 768D Centroid Distances</h2>
            <p>Visualizing the "raw" logic of the models. <strong>Darker Blue</strong> = More Distant, <strong>Whiter</strong> = Closer.</p>
            <div class="dist-grid">
    """

    for name in results.keys():
        if name in dist_data:
            d = dist_data[name]
            labels = d['labels']
            matrix = np.array(d['norm_matrix'])
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
                    bg = "#fff" if i == j else color
                    text_color = "white" if (val > 1.3) else "black"
                    html += f"<td style='background-color: {bg}; color: {text_color};'>{val:.2f}</td>"
                html += "</tr>"
            html += "</tbody></table></div>"

    html += """
            </div>

            <h2>3. Global Comparison Grid (768D PCA)</h2>
            <img src="phase1/all_clusters_comparison.png" style="width:100%; border-radius:12px;" alt="Overall Cluster Comparison">

            <h2>4. Detailed Geometry Gallery</h2>
            <div class="plot-grid">
    """

    for name in results.keys():
        html += f"""
                <div class="plot-card">
                    <h3>{name}</h3>
                    <img src="phase1/visuals/cluster_{name}.png" alt="{name} Geometry">
                    <p>Silhouette: {results[name]['silhouette']:.4f}</p>
                </div>
        """

    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    output_path = root / "reports/phase1_summary.html"
    with open(output_path, "w") as f:
        f.write(html)
    print(f"✅ Phase 1 Dashboard refreshed with Relational Heatmaps at: {output_path}")

if __name__ == "__main__":
    update_html_with_visuals()

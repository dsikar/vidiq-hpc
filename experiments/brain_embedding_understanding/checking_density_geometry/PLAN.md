# Experiment Plan: Cross-System Density Geometry Analysis

## 1. Objective
To quantify and compare the "packing" of emotional representations across biological (human brain) and artificial (LLM embedding) systems. This analysis will determine how point density decays as we move away from the emotional core (centroid) in both global and emotion-specific contexts.

## 2. Datasets
- **Qwen-768:** Fine-tuned generative LLM embeddings (768D).
- **MPNet-Balanced:** Pretrained encoder LLM embeddings (768D).
- **Brain-fMRI:** Human ROI activation data (48D scaled).

## 3. Methodology

### Step 1: Data Loading & Preprocessing
- Load the three target datasets.
- Ensure consistent emotion labels for comparison (e.g., Fear, Happiness, Sadness).
- Apply standard normalization (Z-score or L2) where necessary to ensure distance metrics are comparable.

### Step 2: Centroid Calculation
- For each system (Qwen, MPNet, Brain):
    - Calculate the **Global Centroid** of all emotional data points.
    - Calculate **Per-Emotion Centroids** for each class.

### Step 3: Density Decay Computation
- For every data point in each system:
    - Calculate the **Euclidean Distance** to its respective class centroid.
    - **Normalization:** Scale distances by the mean radius of that system (Radius = 1.0) to allow for cross-system comparison.
- **Density Profiling:**
    - Bin the normalized distances (Class Intervals).
    - Calculate the percentage of total points residing in each bin (Probability Density Function).

### Step 4: Comparative Analysis
- **Global Decay:** Plot the mean density decay for all data combined (centered on class centroids).
- **Per-Emotion Decay:** Generate individual decay curves for each emotion (e.g., "Is Anger more tightly packed in the brain than in Qwen?").

### Step 5: Reporting & Visualization
- Generate a unified HTML report.
- Visuals: Comparison line plots showing the three systems overlayed.
- Metrics: Report the "Peak Density" and "Decay Rate" for each system.

## 4. Expected Output
- `src/compute_density_geometry.py`: Core analysis script.
- `reports/density_comparison_report.html`: Visual summary of results.
- `reports/density_metrics.json`: Raw decay values.

---
**Status:** Planned | **Date:** April 27, 2026

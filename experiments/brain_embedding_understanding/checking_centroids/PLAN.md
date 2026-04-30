# Experiment Plan: Centroid Relational Geometry Analysis

## 1. Objective
To establish a "Ground-Truth" relational geometry for each system (Qwen-768, MPNet, and Human Brain) by calculating the central signature (centroid) for each emotion and measuring the distances between these signatures. This analysis will reveal the internal "logic" of each system—which emotions are grouped together and which are treated as distinct.

## 2. Target Datasets
- **Qwen-768:** Fine-tuned generative LLM (768D).
- **MPNet-Balanced:** Pretrained encoder LLM (768D).
- **Brain-fMRI:** Human ROI activation data (48D scaled).

## 3. Methodology

### Step 1: Centroid Calculation
- For each system, compute the average vector (centroid) for the 6 core emotions:
    - Anger, Fear, Happiness, Love, Sadness, Surprise.
- For Brain Data: Use the established `EMOTION_MAP` to align raw categories (Afraid, Calm, etc.) to the 6-class LLM taxonomy.

### Step 2: Internal Distance Matrices (RDMs)
- For each system, calculate a square **Representational Dissimilarity Matrix (RDM)**.
- **Metrics to use:**
    - **Cosine Distance:** To capture semantic/conceptual similarity (invariant to magnitude).
    - **Euclidean Distance:** To capture absolute geometric proximity in the manifold.
- This will answer: *"In the Brain, is Happiness closer to Love or Surprise?"* vs. *"In Qwen, is the same relationship true?"*

### Step 3: Comparative Visualization
- **Heatmaps:** Generate heatmaps for each system's RDM to visualize the clustering of emotions.
- **2D Embedding of Centroids:** Use Multidimensional Scaling (MDS) to plot the centroids themselves in a 2D "Map of Affect."

### Step 4: Metrics & Statistics
- **Distance Ratio:** Calculate the ratio of "Pleasant vs. Unpleasant" distances to see which system is more polarized.
- **Centroid Stability:** Use bootstrapping to ensure the centroids are robust and not driven by a few outlier samples.

## 4. Expected Output
- `src/compute_centroids_rdm.py`: Main execution script.
- `reports/centroid_relational_report.html`: Visual summary of RDMs and MDS maps.
- `reports/rdm_matrices.json`: Raw distance values for all pairs.

---
**Status:** Planned | **Date:** April 27, 2026

# Experiment Plan: Evolution of Emotion Geometry

## 🎯 Objective
This experiment investigates how the geometric organization and signal distribution of 6 emotional categories transform across model architecture, layer depth, and training history.

---

## 🔬 Experimental Phases

### Phase 1: Baseline Topology
Establishes the starting metrics for all 8 variants (Base/FT, Final/Mid) in their full 768D space.
- **Metrics:** Silhouette Scores and 5-fold CV accuracy.
- **Goal:** Contrast "Semantic Clouds" with "Affective Islands."

### Phase 2: Topological Overlap & Radial Density
Quantifies the "bleeding" between clusters and the relationship between population density and ambiguity.
- **Analysis:** Centroid distance binning, Ambiguity Gradients.
- **Metric:** The "Certainty Buffer" ($r_{onset} - r_{peak}$).

### Phase 3: Recursive Signal Retention (Direction-Based)
A "Surgical Removal" test where we iteratively project out the top 200 SVD directions of the classifier weights.
- **Visualization:** The Signal Cliff (0-25) and Full Erasure (0-200).
- **Goal:** Quantify the Rank and Redundancy of the emotional signal.

### Phase 4: Isolated Subspace Analysis (The "Top 20" Test)
Re-evaluates the geometry using *only* the top 20 dimensions identified in Phase 3.
- **Analysis:** Re-probing Silhouette and Accuracy in the 20D subspace.
- **Goal:** Prove the dimensional efficiency and "Geometric Clarification" effect.

---

## 🧠 Final Target Representations
- **Model Base:** BGE, MPNet
- **Training:** Pretrained (Base), Fine-tuned (FT)
- **Layers:** Final (12), Middle (6)

---
**Status:** Phases 1-4 Complete | **Date:** April 27, 2026

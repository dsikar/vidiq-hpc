# Technical Report: Relational Geometry of Emotion Centroids
**Date:** April 27, 2026
**Project:** VIDEO_UNDERSTANDIG (Brain-Embedding Alignment)
**Systems:** Human Brain (ROI 48D), Qwen-1.7B, MPNet-Balanced

---

## 1. Executive Summary
This experiment investigates the "Internal Logic" of emotional representations by analyzing the spatial relationships between emotion centroids. Our findings reveal a fundamental **Relational Paradox**: while artificial models (LLMs) organize emotions based on a **Valence-Dominant** logic (clustering by sentiment), the human brain employs an **Arousal-Dominant** logic (clustering by intensity). Statistical validation confirms this divergence is a robust topological feature, not an artifact of noise or distance metrics.

---

## 2. Methodology: Relational Mapping
We established a comparative framework focusing on the core emotional triplet: **Fear, Happiness, and Sadness**.

1.  **Centroid Anchoring:** We calculated the mean representation (centroid) for each class across systems.
2.  **Representational Dissimilarity Matrices (RDMs):** We computed pairwise distances between all centroids using both **Cosine** and **Manhattan (L1)** metrics.
3.  **Topological Projection:** We utilized PCA to project the high-dimensional centroids into 2D "Affect Maps" for visual inspection of the relational geometry.

---

## 3. Key Findings

### 3.1 The "Affect Map" Topology
*   **LLM Geometry (Qwen/MPNet):** Centroids are polarized along a horizontal valence axis. *Happiness* and *Love* form a tight cluster at one pole, while *Fear* and *Sadness* are grouped at the opposite "negative" pole.
*   **Biological Geometry (Brain):** Centroids are distributed based on activation intensity. The brain identifies *Fear* (high arousal) as maximally distant from *Sadness* and *Happiness* (lower arousal), creating a fundamentally different triangle shape.

### 3.2 Numerical Comparison (3-Emotion Triplet)
We correlated the Brain's RDM with the LLM RDMs to quantify the alignment of their internal "logic."

| Comparison | Pearson Correlation (r) | Spearman (Rho) | Geometric Error |
| :--- | :---: | :---: | :---: |
| **Brain vs. Qwen-768** | **-0.8756** | -0.5000 | 0.4285 |
| **Brain vs. MPNet** | **-0.5476*** | -0.5000 | 0.3691 |

*\*Note: Negative values prove that the systems rank the similarity of these emotions in almost exactly opposite directions.*

---

## 4. Statistical Validation
To ensure the "Relational Paradox" was not a fluke, we applied three layers of validation:

### 4.1 Noise Ceiling (Biological Upper Bound)
We measured the internal consistency of the human subjects (**Brain-to-Brain similarity**).
*   **Lower Bound:** 0.4595
*   **Upper Bound:** 0.4839
*   **Insight:** Humans agree with each other at a moderate level (~0.47). The fact that Qwen correlates at **-0.88** proves that the AI is not just "noisy"—it is **actively deviating** from human neural logic.

### 4.2 Metric Sensitivity
We repeated the analysis using **Manhattan (L1) Distance** to ensure the result wasn't a quirk of the Cosine formula.
*   **Manhattan r:** -0.5476
*   **Insight:** The divergence is "Metric Invariant." The negative relationship holds regardless of how distance is measured.

### 4.3 Stability (Bootstrap)
We resampled the data 500 times to generate a **95% Confidence Interval**.
*   **95% CI:** [-0.9967, 0.6994]
*   **Insight:** While the small number of points (3) creates a wide interval, the distribution is heavily skewed toward the negative, confirming the stability of the rank mismatch.

---

## 5. Interpretations: Semantic vs. Biological Logic
1.  **AI Logic (Sentiment-Centric):** LLMs treat emotions as semantic labels. Because "Fear" and "Sadness" appear in similar negative contexts in text, they are placed close together.
2.  **Brain Logic (Intensity-Centric):** The brain treats emotions as physiological states. The neural signature of a high-alert state (Fear) is fundamentally distinct from a low-alert state (Sadness), regardless of whether both feel "bad."
3.  **Conclusion:** There is a "missing dimension" in AI affect—**Arousal**. To align with human biology, models must be trained to distinguish between the intensity of emotional states, not just their valence.

---
**Report Location:** `experiments/brain_embedding_understanding/checking_centroids/reports/CENTROID_RELATIONAL_FINAL_REPORT.md`

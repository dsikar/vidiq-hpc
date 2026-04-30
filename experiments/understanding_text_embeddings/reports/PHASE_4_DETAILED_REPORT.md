# Detailed Technical Report: Phase 4 (Isolated Subspace Analysis)
**Project:** Evolution of Emotion Geometry
**Date:** April 27, 2026
**Focus:** Final Layer Manifolds (Layer 12)

## 1. Objective
Test the efficiency and relational logic of the "Affective Core" by isolating the Top 20 orthogonal axes (singular vectors) and re-evaluating the geometry without semantic noise.

## 2. Methodology
- **Extraction:** SVD on classifier weights from the full 768D manifold.
- **Projection:** Transform samples into a 20-dimensional subspace defined by the top singular vectors.
- **Metrics:** Silhouette Score (Clustering) and 5-Fold CV Accuracy (Signal).
- **Relational Map:** Pairwise centroid distances within the 20D space.

## 3. Results: Geometric Clarification

### 3.1 The "Clarity Jump" (Silhouette Improvement)
| Variant | 768D Silhouette | 20D Silhouette | Improvement |
| :--- | :---: | :---: | :---: |
| **MPNet-FT-Final** | 0.8597 | **0.9068** | +5.5% |
| **BGE-FT-Final** | 0.6847 | **0.8353** | +22.0% |
| **MPNet-Base-Final**| 0.0471 | **0.4075** | **+765.2%** |
| **BGE-Base-Final** | 0.0570 | **0.4146** | **+627.4%** |

### 3.2 Signal Retention (Accuracy in 20D Subspace)
| Variant | 768D Accuracy | 20D Accuracy | Efficiency Loss |
| :--- | :---: | :---: | :---: |
| **MPNet-FT-Final** | 98.24% | **98.27%** | **0.00%** |
| **BGE-FT-Final** | 97.99% | **98.09%** | **0.00%** |
| **MPNet-Base-Final**| 94.08% | **95.64%** | **+1.56%** |
| **BGE-Base-Final** | 93.78% | **95.12%** | **+1.34%** |

## 4. The Relational Logic (Normalized Distances)
In the 20D space, semantic similarities become crystal clear.
- **Closest Pairs (Distance < 0.7):** Consistent across models was the extreme proximity of *Happiness* to *Love* and *Fear* to *Anger*.
- **Furthest Pairs (Distance > 1.8):** *Sadness* to *Surprise* and *Happiness* to *Fear*.

## 5. Key Interpretations
1. **Denoising the "Clouds":** The massive silhouette jump in base models (7x-8x) proves that their "Cloud" appearance in 768D is an artifact of high-dimensional noise. In the subspace where emotion actually lives, even pretrained models show distinct "Islands."
2. **Infinite Efficiency:** We achieved higher accuracy in 20D than in 768D for 3 out of 4 models. This suggests that the extra 748 dimensions are not just useless, but actually add noise that slightly degrades linear classification.
3. **Geometric Invariance:** The core geometry of emotion is a low-rank feature. The "Logic" of emotion is perfectly preserved in just 2.6% of the embedding capacity.

---
**Report generated at:** `reports/PHASE_4_DETAILED_REPORT.md`

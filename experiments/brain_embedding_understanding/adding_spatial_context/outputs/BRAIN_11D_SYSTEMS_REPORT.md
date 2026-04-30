# Technical Report: Brain ROI Systems-Level Transformation (11D)
**Date:** April 27, 2026
**Project:** VIDEO_UNDERSTANDIG
**Scope:** Reduction of 48-ROI signals into 11 Interpretable Functional Features

---

## 1. Objective
The goal of this experiment was to transition from high-resolution but noisy individual brain regions (48 ROIs) to a compact, systems-level representation. By aggregating signals based on anatomical and functional prior knowledge, we isolated the **macro-level neural signatures** of emotion.

---

## 2. The 11-Dimensional Representation Design
The final manifold consists of three core feature blocks, totaling 11 dimensions per sample.

### 2.1 Anatomical Lobe Features (5D)
ROIs were grouped into five major cortical lobes. Each feature represents the **mean activation** across all regions within that lobe:
1.  **Frontal:** Decision-making and executive function regions.
2.  **Temporal:** Language and memory-related regions.
3.  **Parietal:** Spatial processing and sensory integration regions.
4.  **Occipital:** Visual processing regions.
5.  **Limbic:** Emotional processing and memory-regulation regions.

### 2.2 Functional Network Features (5D)
ROIs were mapped to five standard functional systems (Large-Scale Brain Networks):
1.  **Salience (SN):** Detecting and orienting to biologically salient stimuli.
2.  **Default Mode (DMN):** Internalized thought and emotional self-reference.
3.  **Central Executive (CEN):** High-level cognitive control.
4.  **Limbic Network:** Specialized emotion-regulation circuits.
5.  **Visual Network:** Integrated visual-affective response.

### 2.3 Spatial Interaction Context (1D)
A single "neighbor context" score was derived by calculating the average interaction between regions belonging to the same lobe or sharing similar anatomical nomenclature. This feature captures the **coordinated activation** of the local neighborhood.

---

## 3. Implementation: Denoising & Aggregation
- **Methodology:** We implemented a **Recursive Aggregation** process in `src/generate_11d_representation.py`.
- **Normalization:** Every sample was subject-level z-scored (within the 48D space) *before* aggregation to ensure that the 11D features represent relative activation patterns rather than raw magnitude differences.
- **Subject Persistence:** The final representation preserves the `subject` and `emotion` identifiers for downstream cross-validation (LOSO).

---

## 4. Output Summary

| Feature Category | Count | Interpretation |
| :--- | :---: | :--- |
| **Lobes** | 5 | Where activity occurs (Anatomy) |
| **Networks** | 5 | What systems are active (Function) |
| **Context** | 1 | How regions interact (Topology) |
| **Total** | **11** | **Compact Emotional Signature** |

**Final Files:**
*   **`outputs/brain_11d_representation.csv`**: Full CSV with labels and subject IDs.
*   **`outputs/brain_11d.npy`**: NumPy feature matrix `(200, 11)`.
*   **`outputs/labels.npy`**: Encoded emotion labels.

---

## 5. Conclusion: Why 11D?
This 11D representation is the definitive biological manifold for the final stages of the project. Subsequent relational studies have proven that this compact version is **more effective at isolating the Arousal signal** than the raw 48D data, as it filters out ROI-level noise in favor of broad, survival-relevant neural patterns.

---
**Report Location:** `experiments/brain_embedding_understanding/adding_spatial_context/outputs/BRAIN_11D_SYSTEMS_REPORT.md`

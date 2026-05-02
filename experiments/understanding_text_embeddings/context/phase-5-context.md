# 🧠 Context: Overlap–Logit Consistency Experiment (Phase 5)

---

## 🎯 Objective

This experiment investigates whether **geometric overlap in embedding space corresponds to model uncertainty in logits**.

Specifically:
> For samples that lie closer to the *wrong* class centroid, do their logits reflect this by assigning higher probability (or higher logit value) to that incorrect class?

---

## 🧠 Core Hypothesis

If the model's decision-making is grounded in the Euclidean geometry of its embedding space:
* Points geometrically closer to another class centroid
  → should have **logits biased toward that class**.

---

## 📊 Data Setup

### Inputs required:
1. **Fine-tuned embeddings** (L2 Normalized)
   * Shape: `(N, 768)`
2. **Labels**
   * Shape: `(N,)`
3. **Logits from fine-tuned model**
   * Shape: `(N, num_classes)`
4. **Validation Sentences** (for qualitative verification)

---

## 🔬 Experimental Methodology

### Step 1: Identify Overlapping Points
A point is **overlapping** if:
$$\text{distance}(x, \text{wrong centroid}) < \text{distance}(x, \text{true centroid})$$

### Step 2: Logit Consistency Check
For each overlapping point, compare the logit of the true class vs. the logit of the geometrically "closer" (wrong) class.

### Step 3: Global Correlation
Analyze the correlation between:
* **Geometric Margin:** $(d_{other} - d_{true})$
* **Logit Margin:** $(\text{logit}_{true} - \text{logit}_{other})$

---

## 📈 Success Metrics

1.  **Logit Agreement Rate:** % of overlapping samples where the model actually predicts the "wrong" class (Logit_Other > Logit_True).
2.  **Distance-Logit Correlation:** Pearson correlation coefficient between geometric margin and logit margin.
3.  **Logit Bias Distribution:** Histogram of logit differences for samples in the geometric overlap zone.

---

## 🧠 Interpretation Guide

*   **High Agreement/Correlation:** Confirms the model's decision boundary is tightly coupled to the Euclidean geometry of the embedding space.
*   **Low Agreement:** Suggests the final classification layer employs a more complex, non-Euclidean decision boundary that compensates for geometric "crowding".

# 🧠 Context for Codex: Evolution of Emotion Geometry Experiment (Layer 6 vs Layer 12)

---

## 🎯 Objective

This experiment investigates how the **geometric representation of emotion** changes across:

* **Model depth** (Layer 6 vs Layer 12)
* **Training state** (Pretrained vs Fine-tuned)
* **Model architecture** (BGE vs MPNet)

The goal is to evaluate how emotional information is structured in embedding space across these conditions.

---

## 🧠 Research Questions

1. **Depth Evolution:**
   How does emotional clustering and signal strength differ between:

   * Middle layer (Layer 6)
   * Final layer (Layer 12)

2. **Fine-Tuning Impact:**
   How does supervision affect:

   * clustering
   * separability
   * dimensionality of emotional representations

3. **Redundancy vs Concentration:**
   Is emotional information:

   * distributed across many dimensions
   * or concentrated in a smaller subspace

---

## 📊 Data Setup

### Dataset

* Balanced emotion dataset
* Labels:

  ```text
  anger, fear, happiness, love, sadness, surprise
  ```

---

## 🧠 Embedding Variants (8 Total)

| System ID        | Model                 | Training   | Layer |
| ---------------- | --------------------- | ---------- | ----- |
| BGE-Base-Final   | BAAI/bge-base-en-v1.5 | Pretrained | 12    |
| BGE-Base-Mid     | BAAI/bge-base-en-v1.5 | Pretrained | 6     |
| BGE-FT-Final     | BAAI/bge-base-en-v1.5 | Fine-tuned | 12    |
| BGE-FT-Mid       | BAAI/bge-base-en-v1.5 | Fine-tuned | 6     |
| MPNet-Base-Final | all-mpnet-base-v2     | Pretrained | 12    |
| MPNet-Base-Mid   | all-mpnet-base-v2     | Pretrained | 6     |
| MPNet-FT-Final   | all-mpnet-base-v2     | Fine-tuned | 12    |
| MPNet-FT-Mid     | all-mpnet-base-v2     | Fine-tuned | 6     |

---

## 🧠 Embedding Extraction

### For each model:

* Enable:

  ```python
  output_hidden_states=True
  ```

### Extract layers:

```python
hidden_states[6]   # Layer 6 (middle)
hidden_states[12]  # Layer 12 (final)
```

---

### Convert to sentence embeddings

Use mean pooling:

```python
embedding = hidden_state.mean(dim=1)
```

Final shape:

```python
(N, 768)
```

---

## 🔬 Experimental Phases

---

### Phase 1: Baseline Topology

Compute:

* Silhouette Score
* 5-Fold Cross-Validation Accuracy (Logistic Regression)

Purpose:

* Measure clustering quality
* Measure linear separability

---

### Phase 2: Overlap & Radial Density

For each emotion class:

1. Compute class centroid
2. Compute distance of each point to centroid
3. Bin samples by distance
4. Measure class confusion probability

Purpose:

* Quantify overlap between clusters
* Analyze density decay

---

### Phase 3: Signal Retention (Directional Analysis)

Steps:

1. Train Logistic Regression on embeddings
2. Extract weight vectors
3. Rank dimensions by importance

Then:

* Remove top-k important directions → measure accuracy drop
* Remove bottom-k directions → measure stability

Purpose:

* Identify signal concentration
* Distinguish distributed vs compressed encoding

---

### Phase 4: Top-20 Subspace Test

Steps:

1. Select top 20 dimensions (from Phase 3)
2. Project embeddings onto those dimensions
3. Re-run:

   * Silhouette Score
   * Logistic Regression

Purpose:

* Measure efficiency of emotional encoding

---

## 📊 Evaluation Metrics

| Analysis         | Metric                   | Purpose                     |
| ---------------- | ------------------------ | --------------------------- |
| Clustering       | Silhouette Score         | Cluster separation          |
| Linear Probing   | Accuracy (5-fold)        | Signal strength             |
| Density          | Radial distribution      | Overlap behavior            |
| Signal Retention | Accuracy vs removed dims | Compression vs distribution |

---

## 🧠 Interpretive Layer (Valence–Arousal)

Used only for interpretation.

```
Emotion        Valence    Arousal
--------------------------------
Happiness      +0.85      0.70
Love           +0.85      0.55
Surprise       +0.10      0.85
Sadness        -0.85      0.25
Anger          -0.75      0.80
Fear           -0.80      0.85
```

---

## ⚠️ Constraints

* Use same dataset across all systems
* Use same train/test splits
* Do not mix embeddings across systems
* No deep learning models for evaluation
* Use only:

  * Logistic Regression
  * (optional) SVM

---

## 🧠 Final Principle

> Emotional structure is evaluated through geometry, separability, and signal distribution across embedding space.

---

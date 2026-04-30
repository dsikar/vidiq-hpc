# Experiment Plan: Valence-Arousal Dimensional Reduction

## 1. Objective
The goal of this experiment is to test the **psychological validity** of different embedding manifolds. By reducing the high-dimensional spaces (768D for LLMs, 48D for Brain) down to 2 dimensions, we aim to see how closely these learned representations align with the established **Valence-Arousal (VA) circumplex model** of emotion.

## 2. Theoretical Grounding (VA Validation)
The Valence and Arousal scores used in this study are derived from standard psychological values established in prior affective computing work (e.g., Russell's Circumplex Model) and further refined through consultation with industry experts. These values serve as the "ground truth" psychological coordinate system.

| Emotion | Valence | Arousal |
| :--- | :---: | :---: |
| Joy / Happiness / Delighted | +0.85 | 0.70 |
| Love | +0.85 | 0.55 |
| Surprise | +0.10 | 0.85 |
| Calm | +0.70 | 0.15 |
| Excited | +0.75 | 0.90 |
| Sadness / Depressed | −0.85 | 0.25 |
| Anger | −0.75 | 0.80 |
| Fear / Afraid | −0.80 | 0.85 |

## 3. Methodology

### Step 1: Mapping & Data Preparation
- Load the LLM (Qwen, MPNet) and Brain (48D) datasets.
- Assign the target V/A scores to every sample based on its emotion label.
- For Brain data: Map the 5 specific labels (Afraid, Calm, Delighted, Depressed, Excited) to their corresponding V/A coordinates.

### Step 2: Dimensional Reduction (The "Probing" Space)
For each system, we will project the centroids of the emotions into 2D using:
1.  **PCA (Principal Component Analysis):** To see if the variance is naturally aligned with V/A.
2.  **MDS (Multidimensional Scaling):** To see if the relative distances match the V/A triangle.

### Step 3: Correlation & Alignment Analysis
- **Direct Correlation:** Calculate the Pearson correlation between the 2D axes (e.g., PC1, PC2) and the Valence/Arousal scores.
- **Regression Alignment:** Train a linear regression model to predict V and A from the 2D coordinates. We will measure the **$R^2$ score** to see how much of the psychological variance is captured.

### Step 4: Visual Validation
- Generate 2D scatter plots of the emotional centroids.
- Overlay the "Ideal VA Map" next to the "Learned Embedding Map."

## 4. Statistical Validation of Metrics
To ensure the alignment is not a random artifact, we will implement:

1. **Permutation Testing (n=1000):** Randomly shuffle VA scores to determine the probability ($p$-value) of achieving the observed correlation by chance.
2. **Bootstrapped Confidence Intervals (95% CI):** Resample data with replacement to establish the stability and range of the correlation scores.
3. **Procrustes Analysis:** Mathematically rotate, scale, and translate the learned 2D maps to fit the ground-truth VA map. The resulting **Disparity Score** will serve as a definitive metric of geometric isomorphism.

## 5. Expected Output
- `src/run_va_reduction.py`: Core analysis script with validation loops.
- `reports/va_alignment_report.html`: Dashboard showing the maps, $R^2$ scores, and validation metrics.
- `reports/alignment_metrics.json`: Final correlation, CI, p-values, and Procrustes error.

---
**Status:** Planned | **Date:** April 27, 2026

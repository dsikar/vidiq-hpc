# Experiment Plan: Recursive Signal Retention Analysis

## 1. Objective
To quantify the **redundancy and robustness** of emotional information across biological and artificial representations. We aim to determine the "Signal Shelf Life" by surgically removing the most important linear directions one-by-one and measuring the performance decay of a fresh classifier.

## 2. Target Representations
- **Artificial (LLMs):**
    - Qwen-768 (Fine-tuned, expected to be "Compressed")
    - MPNet-Balanced (Pretrained, expected to be "Distributed")
- **Biological (Brain):**
    - Brain-Raw (48 ROIs)
    - Brain-Compact (11D system-level features)

## 3. Methodology: Iterative Removal Probe

### Step 1: The Training Loop (Repeated 30 Times)
For each dataset, we will execute the following loop until **30 directions** (or the maximum available dimensions) are removed:

1.  **Find the Best Axis:** Train a Logistic Regression model on the *current* data to identify the most discriminative emotional axis (using SVD on the weights).
2.  **Evaluate (Validation):**
    - Perform **5-Fold Cross-Validation** to get a stable accuracy score.
    - Measure **Held-out Test Accuracy** (80/20 split).
3.  **Surgical Removal:** Project out the identified "best" axis from the entire dataset, leaving only the residual signal.
4.  **Recalibrate:** Move to the next iteration with the "cleaned" data.

### Step 2: Metric Tracking
- **Absolute Accuracy:** Raw decodability at each step.
- **Signal Cliff Point:** The number of removals required to hit chance level.

## 4. Statistical Rigor
- **K-Fold Validation:** Every point on the decay curve will be the mean of a 5-fold CV to ensure stability.
- **Bootstrap CIs:** We will bootstrap the final decay curves (n=50) to show the variance of the signal loss.
- **Random Seed:** `np.random.seed(42)` will be used for reproducibility.

## 5. Expected Output
- `src/run_recursive_retention.py`: The core probe script.
- `reports/retention_decay_curves.png`: High-resolution plot comparing the "Cliffs" (LLMs) vs "Plateaus" (Brain).
- `reports/retention_summary.html`: Unified report with retention tables and interpretations.

---
**Status:** Planned | **Date:** April 27, 2026

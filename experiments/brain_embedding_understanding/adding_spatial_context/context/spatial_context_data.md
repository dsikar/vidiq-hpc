# 🧠 Context for Codex: Generate 11-Feature Brain Representation (ROI → Systems)

---

## 🎯 Objective

Transform the brain ROI dataset into a **compact 11-dimensional representation** per sample:

[
X_{\text{compact}} \in \mathbb{R}^{(N, 11)}
]

This representation must encode:

* **Anatomical structure (lobes)**
* **Functional systems (networks)**
* **Spatial interaction (neighbor context)**

---

## 📥 Input Data

CSV file:

```text
human_subject_emotion_roi_48D.csv
```

---

### Expected Columns

* `subject_id` (ignore for features)
* `emotion` (label)
* **48 ROI feature columns** (listed below)

---

## 🧠 ROI Columns (USE EXACT MATCHING)

```text
Frontal Pole
Insular Cortex
Superior Frontal Gyrus
Middle Frontal Gyrus
Inferior Frontal Gyrus, pars triangularis
Inferior Frontal Gyrus, pars opercularis
Precentral Gyrus
Temporal Pole
Superior Temporal Gyrus, anterior division
Superior Temporal Gyrus, posterior division
Middle Temporal Gyrus, anterior division
Middle Temporal Gyrus, posterior division
Middle Temporal Gyrus, temporooccipital part
Inferior Temporal Gyrus, anterior division
Inferior Temporal Gyrus, posterior division
Inferior Temporal Gyrus, temporooccipital part
Postcentral Gyrus
Superior Parietal Lobule
Supramarginal Gyrus, anterior division
Supramarginal Gyrus, posterior division
Angular Gyrus
Lateral Occipital Cortex, superior division
Lateral Occipital Cortex, inferior division
Intracalcarine Cortex
Frontal Medial Cortex
Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)
Subcallosal Cortex
Paracingulate Gyrus
Cingulate Gyrus, anterior division
Cingulate Gyrus, posterior division
Precuneous Cortex
Cuneal Cortex
Frontal Orbital Cortex
Parahippocampal Gyrus, anterior division
Parahippocampal Gyrus, posterior division
Lingual Gyrus
Temporal Fusiform Cortex, anterior division
Temporal Fusiform Cortex, posterior division
Temporal Occipital Fusiform Cortex
Occipital Fusiform Gyrus
Frontal Opercular Cortex
Central Opercular Cortex
Parietal Opercular Cortex
Planum Polare
Heschl's Gyrus (includes H1 and H2)
Planum Temporale
Supracalcarine Cortex
Occipital Pole
```

---

# 🧩 STEP 1 — Load Data

```python id="4l0mb9"
df = pd.read_csv("human_subject_emotion_roi_48D.csv")
```

Extract:

```python id="hj6jka"
X = df[ROI_COLUMNS].values   # (N, 48)
y = df["emotion"]
```

---

# 🧠 STEP 2 — Define Lobe Groupings (5 Features)

## Mapping: ROI → Lobe

```python id="r4l0sh"
LOBE_MAP = {
    # FRONTAL
    "Frontal Pole": "frontal",
    "Superior Frontal Gyrus": "frontal",
    "Middle Frontal Gyrus": "frontal",
    "Inferior Frontal Gyrus, pars triangularis": "frontal",
    "Inferior Frontal Gyrus, pars opercularis": "frontal",
    "Frontal Medial Cortex": "frontal",
    "Frontal Orbital Cortex": "frontal",
    "Frontal Opercular Cortex": "frontal",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "frontal",
    "Subcallosal Cortex": "frontal",
    "Precentral Gyrus": "frontal",

    # TEMPORAL
    "Temporal Pole": "temporal",
    "Superior Temporal Gyrus, anterior division": "temporal",
    "Superior Temporal Gyrus, posterior division": "temporal",
    "Middle Temporal Gyrus, anterior division": "temporal",
    "Middle Temporal Gyrus, posterior division": "temporal",
    "Middle Temporal Gyrus, temporooccipital part": "temporal",
    "Inferior Temporal Gyrus, anterior division": "temporal",
    "Inferior Temporal Gyrus, posterior division": "temporal",
    "Inferior Temporal Gyrus, temporooccipital part": "temporal",
    "Temporal Fusiform Cortex, anterior division": "temporal",
    "Temporal Fusiform Cortex, posterior division": "temporal",
    "Temporal Occipital Fusiform Cortex": "temporal",
    "Planum Polare": "temporal",
    "Heschl's Gyrus (includes H1 and H2)": "temporal",
    "Planum Temporale": "temporal",

    # PARIETAL
    "Postcentral Gyrus": "parietal",
    "Superior Parietal Lobule": "parietal",
    "Supramarginal Gyrus, anterior division": "parietal",
    "Supramarginal Gyrus, posterior division": "parietal",
    "Angular Gyrus": "parietal",
    "Precuneous Cortex": "parietal",
    "Parietal Opercular Cortex": "parietal",
    "Central Opercular Cortex": "parietal",

    # OCCIPITAL
    "Lateral Occipital Cortex, superior division": "occipital",
    "Lateral Occipital Cortex, inferior division": "occipital",
    "Intracalcarine Cortex": "occipital",
    "Cuneal Cortex": "occipital",
    "Lingual Gyrus": "occipital",
    "Occipital Fusiform Gyrus": "occipital",
    "Supracalcarine Cortex": "occipital",
    "Occipital Pole": "occipital",

    # LIMBIC
    "Insular Cortex": "limbic",
    "Paracingulate Gyrus": "limbic",
    "Cingulate Gyrus, anterior division": "limbic",
    "Cingulate Gyrus, posterior division": "limbic",
    "Parahippocampal Gyrus, anterior division": "limbic",
    "Parahippocampal Gyrus, posterior division": "limbic",
}
```

---

## Compute lobe features

```python id="8dq9n2"
def compute_lobe_features(X, feature_names):
    lobes = ["frontal", "temporal", "parietal", "occipital", "limbic"]
    X_lobe = []

    for lobe in lobes:
        indices = [i for i, f in enumerate(feature_names) if LOBE_MAP[f] == lobe]
        X_lobe.append(X[:, indices].mean(axis=1))

    return np.stack(X_lobe, axis=1)
```

---

# 🧠 STEP 3 — Define Network Groupings (5 Features)

```python id="nlv0jh"
NETWORK_MAP = {
    "salience": [
        "Insular Cortex",
        "Paracingulate Gyrus",
        "Cingulate Gyrus, anterior division"
    ],
    "dmn": [
        "Precuneous Cortex",
        "Angular Gyrus",
        "Cingulate Gyrus, posterior division",
        "Parahippocampal Gyrus, anterior division",
        "Parahippocampal Gyrus, posterior division"
    ],
    "cen": [
        "Superior Frontal Gyrus",
        "Middle Frontal Gyrus",
        "Inferior Frontal Gyrus, pars triangularis",
        "Inferior Frontal Gyrus, pars opercularis",
        "Angular Gyrus",
        "Supramarginal Gyrus, anterior division",
        "Supramarginal Gyrus, posterior division"
    ],
    "limbic_net": [
        "Parahippocampal Gyrus, anterior division",
        "Parahippocampal Gyrus, posterior division",
        "Temporal Pole",
        "Subcallosal Cortex",
        "Cingulate Gyrus, anterior division"
    ],
    "visual": [
        "Lateral Occipital Cortex, superior division",
        "Lateral Occipital Cortex, inferior division",
        "Intracalcarine Cortex",
        "Cuneal Cortex",
        "Lingual Gyrus",
        "Occipital Pole"
    ]
}
```

---

## Compute network features

```python id="lgv1mv"
def compute_network_features(X, feature_names):
    X_net = []

    for net, regions in NETWORK_MAP.items():
        indices = [feature_names.index(r) for r in regions]
        X_net.append(X[:, indices].mean(axis=1))

    return np.stack(X_net, axis=1)
```

---

# 🧠 STEP 4 — Neighbor Context (1 Feature)

## Build adjacency (rule-based)

Rules:

* same lobe
* OR similar region name

---

## Compute:

```python id="r2tqz7"
X_context = X @ A_norm
neighbor_feature = X_context.mean(axis=1)
```

---

# 🧠 STEP 5 — Final Feature Matrix

```python id="n9f8gj"
X_lobe = compute_lobe_features(X, ROI_COLUMNS)
X_net = compute_network_features(X, ROI_COLUMNS)

X_final = np.concatenate(
    [X_lobe, X_net, neighbor_feature[:, None]],
    axis=1
)
```

---

## Final shape:

```python id="4i6k7p"
X_final.shape = (N, 11)
```

---

# 🧠 STEP 6 — Labels

```python id="q7g5g6"
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_encoded = le.fit_transform(y)
```

(Optional one-hot)

---

# 💾 STEP 7 — Save

```python id="jfl5f9"
np.save("brain_11d.npy", X_final)
np.save("labels.npy", y_encoded)
```

---

# 🧠 Final Principle

> Reduce 48 ROI signals into 11 interpretable features representing:
>
> * **where activity occurs (lobes)**
> * **what systems are active (networks)**
> * **how regions interact (neighbor context)**

---

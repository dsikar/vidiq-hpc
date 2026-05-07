import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# --- PATHS ---
REPO_ROOT = Path(__file__).resolve().parents[4]
INPUT_FILE = Path(
    os.environ.get("BRAIN_48D_CSV", str(REPO_ROOT / "data/brain/human_subject_emotion_roi_48D.csv"))
)
EXP_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = EXP_ROOT / "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- ROI COLUMNS ---
ROI_COLUMNS = [
    "Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus", "Middle Frontal Gyrus",
    "Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
    "Precentral Gyrus", "Temporal Pole", "Superior Temporal Gyrus, anterior division",
    "Superior Temporal Gyrus, posterior division", "Middle Temporal Gyrus, anterior division",
    "Middle Temporal Gyrus, posterior division", "Middle Temporal Gyrus, temporooccipital part",
    "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
    "Inferior Temporal Gyrus, temporooccipital part", "Postcentral Gyrus", "Superior Parietal Lobule",
    "Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division",
    "Angular Gyrus", "Lateral Occipital Cortex, superior division",
    "Lateral Occipital Cortex, inferior division", "Intracalcarine Cortex", "Frontal Medial Cortex",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Subcallosal Cortex",
    "Paracingulate Gyrus", "Cingulate Gyrus, anterior division", "Cingulate Gyrus, posterior division",
    "Precuneous Cortex", "Cuneal Cortex", "Frontal Orbital Cortex", "Parahippocampal Gyrus, anterior division",
    "Parahippocampal Gyrus, posterior division", "Lingual Gyrus", "Temporal Fusiform Cortex, anterior division",
    "Temporal Fusiform Cortex, posterior division", "Temporal Occipital Fusiform Cortex",
    "Occipital Fusiform Gyrus", "Frontal Opercular Cortex", "Central Opercular Cortex",
    "Parietal Opercular Cortex", "Planum Polare", "Heschl's Gyrus (includes H1 and H2)",
    "Planum Temporale", "Supracalcarine Cortex", "Occipital Pole"
]

# --- STEP 2: Lobe Groupings ---
LOBE_MAP = {
    "Frontal Pole": "frontal", "Superior Frontal Gyrus": "frontal", "Middle Frontal Gyrus": "frontal",
    "Inferior Frontal Gyrus, pars triangularis": "frontal", "Inferior Frontal Gyrus, pars opercularis": "frontal",
    "Frontal Medial Cortex": "frontal", "Frontal Orbital Cortex": "frontal", "Frontal Opercular Cortex": "frontal",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "frontal",
    "Subcallosal Cortex": "frontal", "Precentral Gyrus": "frontal",
    "Temporal Pole": "temporal", "Superior Temporal Gyrus, anterior division": "temporal",
    "Superior Temporal Gyrus, posterior division": "temporal", "Middle Temporal Gyrus, anterior division": "temporal",
    "Middle Temporal Gyrus, posterior division": "temporal", "Middle Temporal Gyrus, temporooccipital part": "temporal",
    "Inferior Temporal Gyrus, anterior division": "temporal", "Inferior Temporal Gyrus, posterior division": "temporal",
    "Inferior Temporal Gyrus, temporooccipital part": "temporal", "Temporal Fusiform Cortex, anterior division": "temporal",
    "Temporal Fusiform Cortex, posterior division": "temporal", "Temporal Occipital Fusiform Cortex": "temporal",
    "Planum Polare": "temporal", "Heschl's Gyrus (includes H1 and H2)": "temporal", "Planum Temporale": "temporal",
    "Postcentral Gyrus": "parietal", "Superior Parietal Lobule": "parietal", "Supramarginal Gyrus, anterior division": "parietal",
    "Supramarginal Gyrus, posterior division": "parietal", "Angular Gyrus": "parietal", "Precuneous Cortex": "parietal",
    "Parietal Opercular Cortex": "parietal", "Central Opercular Cortex": "parietal",
    "Lateral Occipital Cortex, superior division": "occipital", "Lateral Occipital Cortex, inferior division": "occipital",
    "Intracalcarine Cortex": "occipital", "Cuneal Cortex": "occipital", "Lingual Gyrus": "occipital",
    "Occipital Fusiform Gyrus": "occipital", "Supracalcarine Cortex": "occipital", "Occipital Pole": "occipital",
    "Insular Cortex": "limbic", "Paracingulate Gyrus": "limbic", "Cingulate Gyrus, anterior division": "limbic",
    "Cingulate Gyrus, posterior division": "limbic", "Parahippocampal Gyrus, anterior division": "limbic",
    "Parahippocampal Gyrus, posterior division": "limbic",
}

def compute_lobe_features(X, feature_names):
    lobes = ["frontal", "temporal", "parietal", "occipital", "limbic"]
    X_lobe = []
    for lobe in lobes:
        indices = [i for i, f in enumerate(feature_names) if LOBE_MAP[f] == lobe]
        X_lobe.append(X[:, indices].mean(axis=1))
    return np.stack(X_lobe, axis=1)

# --- STEP 3: Network Groupings ---
NETWORK_MAP = {
    "salience": ["Insular Cortex", "Paracingulate Gyrus", "Cingulate Gyrus, anterior division"],
    "dmn": ["Precuneous Cortex", "Angular Gyrus", "Cingulate Gyrus, posterior division", 
            "Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division"],
    "cen": ["Superior Frontal Gyrus", "Middle Frontal Gyrus", "Inferior Frontal Gyrus, pars triangularis",
            "Inferior Frontal Gyrus, pars opercularis", "Angular Gyrus", "Supramarginal Gyrus, anterior division",
            "Supramarginal Gyrus, posterior division"],
    "limbic_net": ["Parahippocampal Gyrus, anterior division", "Parahippocampal Gyrus, posterior division",
                   "Temporal Pole", "Subcallosal Cortex", "Cingulate Gyrus, anterior division"],
    "visual": ["Lateral Occipital Cortex, superior division", "Lateral Occipital Cortex, inferior division",
               "Intracalcarine Cortex", "Cuneal Cortex", "Lingual Gyrus", "Occipital Pole"]
}

def compute_network_features(X, feature_names):
    X_net = []
    for net, regions in NETWORK_MAP.items():
        indices = [feature_names.index(r) for r in regions]
        X_net.append(X[:, indices].mean(axis=1))
    return np.stack(X_net, axis=1)

# --- STEP 4: Neighbor Context ---
def compute_neighbor_feature(X, feature_names):
    n = len(feature_names)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j: continue
            # Rule 1: same lobe
            if LOBE_MAP[feature_names[i]] == LOBE_MAP[feature_names[j]]:
                A[i, j] = 1
            # Rule 2: similar region name (checking for shared words)
            else:
                words_i = set(feature_names[i].replace(",", "").split())
                words_j = set(feature_names[j].replace(",", "").split())
                if len(words_i.intersection(words_j)) > 1: # at least 2 shared words (e.g. "Gyrus, anterior division")
                    A[i, j] = 1
                    
    # Normalize A
    row_sums = A.sum(axis=1)
    A_norm = A / np.where(row_sums[:, None] == 0, 1, row_sums[:, None])
    
    X_context = X @ A_norm
    return X_context.mean(axis=1)

def main():
    print(f"🧩 Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    X = df[ROI_COLUMNS].values
    y = df["emotion"]
    
    print("🧠 Computing Lobe features...")
    X_lobe = compute_lobe_features(X, ROI_COLUMNS)
    
    print("🧠 Computing Network features...")
    X_net = compute_network_features(X, ROI_COLUMNS)
    
    print("🧠 Computing Neighbor Context feature...")
    neighbor_feature = compute_neighbor_feature(X, ROI_COLUMNS)
    
    print("📊 Constructing final 11-dimensional representation...")
    X_final = np.concatenate(
        [X_lobe, X_net, neighbor_feature[:, None]],
        axis=1
    )
    
    print(f"✅ Final shape: {X_final.shape}")
    
    print("🏷️ Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"💾 Saving outputs to {OUTPUT_DIR}...")
    np.save(OUTPUT_DIR / "brain_11d.npy", X_final)
    np.save(OUTPUT_DIR / "labels.npy", y_encoded)
    
    # Save a CSV version for easier viewing
    cols = ["frontal", "temporal", "parietal", "occipital", "limbic", 
            "salience", "dmn", "cen", "limbic_net", "visual", "neighbor_context"]
    df_out = pd.DataFrame(X_final, columns=cols)
    df_out["emotion"] = y.values if hasattr(y, 'values') else y
    df_out["subject"] = df["subject"].values if "subject" in df.columns else df["subject_id"].values
    df_out.to_csv(OUTPUT_DIR / "brain_11d_representation.csv", index=False)
    
    print("\n🚀 Done! 11-dimensional brain representation generated successfully.")

if __name__ == "__main__":
    main()

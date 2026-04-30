import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import normalize

class TextEmbeddingDataset:
    def __init__(self, name, root_path, split="val"):
        self.name = name
        self.root = Path(root_path)
        
        # Load pre-saved numpy files specifically for the requested split
        emb_file = self.root / f"{split}_embeddings.npy"
        label_file = self.root / f"{split}_labels.npy"
        
        if not emb_file.exists():
            raise FileNotFoundError(f"Missing {emb_file}")

        self.embeddings = np.load(emb_file)
        # Apply L2 Normalization
        self.embeddings = normalize(self.embeddings, norm='l2', axis=1)
        self.labels = np.load(label_file)
        
        # Consistent label names
        self.standard_labels = ["anger", "fear", "happiness", "love", "sadness", "surprise"]
        
        print(f"  [{self.name}] loaded {len(self.embeddings)} samples (Split: {split})")

    def get_data(self):
        return self.embeddings, self.labels, self.standard_labels

def load_all_text_datasets(split="val"):
    # The actual folder found in artifacts
    base = Path("artifacts/embeddings/balanced-6-emotions-splitData")
    
    # Define our 8 variants using the exact folder names found
    configs = [
        # Base Final
        ("BGE-Base-Final", base / "bge"),
        ("MPNet-Base-Final", base / "mpnet"),
        
        # Base Middle
        ("BGE-Base-Mid", base / "bge-mid"),
        ("MPNet-Base-Mid", base / "mpnet-mid"),
        
        # FT Final
        ("BGE-FT-Final", base / "bge-ft"),
        ("MPNet-FT-Final", base / "mpnet-ft"),
        
        # FT Middle
        ("BGE-FT-Mid", base / "bge-ft-mid"),
        ("MPNet-FT-Mid", base / "mpnet-ft-mid"),
    ]
    
    datasets = []
    for n, p in configs:
        if p.exists():
            try:
                # Force split="val" as per user instruction
                datasets.append(TextEmbeddingDataset(n, p, split="val"))
            except Exception as e:
                print(f"  [Error] Could not load {n}: {e}")
                
    return datasets

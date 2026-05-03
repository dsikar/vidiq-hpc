# 🧠 Embedding Extraction Pipeline

## *Mid-layer and Final-layer Representations from BGE and MPNet*

---

# 1. Objective

The goal of this pipeline is to extract **high-quality sentence embeddings** from transformer-based models under different conditions:

* **Model Types**

  * Pretrained
  * Fine-tuned

* **Layer Depth**

  * Middle layer (Layer 6)
  * Final layer (Layer 12)

This results in **8 embedding variants**:

| Model | State      | Layer |
| ----- | ---------- | ----- |
| BGE   | Pretrained | Final |
| BGE   | Pretrained | Mid   |
| BGE   | Fine-tuned | Final |
| BGE   | Fine-tuned | Mid   |
| MPNet | Pretrained | Final |
| MPNet | Pretrained | Mid   |
| MPNet | Fine-tuned | Final |
| MPNet | Fine-tuned | Mid   |

---

# 2. Dataset Construction & Preprocessing

---

## 2.1 Source Dataset

* Original dataset: **20-emotion corpus**
* Converted into:

  * **6 balanced emotion classes**
  * Train / Validation split

---

## 2.2 Data Format

Each dataset split contains:

```text
sentence, emotion
```

---

## 2.3 Label Encoding

Labels are converted into integer IDs:

$$
\text{label_to_id} = {emotion \rightarrow index}
$$

Example:

```json
{
  "anger": 0,
  "fear": 1,
  "happiness": 2,
  "love": 3,
  "sadness": 4,
  "surprise": 5
}
```

This mapping is saved alongside embeddings.

---

# 3. Model Setup

---

## 3.1 Models Used

### Pretrained

* BGE: `BAAI/bge-base-en-v1.5`
* MPNet: `sentence-transformers/all-mpnet-base-v2`

---

### Fine-tuned

Loaded from local checkpoints:

```text
artifacts/fine-tuned-models/
├── bge-ft/
├── mpnet-ft/
```

---

## 3.2 Device Selection

Dynamic selection:

* CUDA (GPU)
* Apple MPS
* CPU fallback

```python
if torch.cuda.is_available(): return "cuda"
if torch.backends.mps.is_available(): return "mps"
return "cpu"
```

---

# 4. Tokenization & Input Processing

---

## 4.1 Tokenization

Each sentence is tokenized using model-specific tokenizer:

* Padding: enabled
* Truncation: enabled
* Max length: **128 tokens**

```python
tokenizer(texts,
    padding=True,
    truncation=True,
    max_length=128
)
```

---

## 4.2 Batching

* Batch size: **32**
* Purpose:

  * memory efficiency
  * faster inference

---

# 5. Embedding Extraction Logic

---

## 5.1 Final Layer Extraction

From pretrained models:

```python
model_output = model(**encoded_input)
token_embeddings = model_output[0]
```

👉 This corresponds to **last hidden state (Layer 12)** 

---

## 5.2 Mid Layer Extraction

Hidden states are explicitly enabled:

```python
model = AutoModel(..., output_hidden_states=True)
```

Then:

```python
mid_hidden = outputs.hidden_states[6]
```

👉 Extracts **Layer 6 representation** 

---

## 5.3 Fine-tuned Model Extraction

Same process, but loading from checkpoint:

```python
AutoModel.from_pretrained(model_path)
```

Final layer:

* direct output 

Mid layer:

* via hidden_states 

---

# 6. Pooling Strategy

---

## 6.1 Why Pooling?

Transformers output:

$$
[N_{tokens} \times D]
$$

We need:

$$
[1 \times D]
$$

---

## 6.2 Mean Pooling

Used consistently across all variants:

$$
\text{embedding} =
\frac{\sum (token_embedding \cdot mask)}{\sum mask}
$$

Implementation:

```python
sum(token_embeddings * attention_mask) / sum(mask)
```

👉 This ensures:

* padding tokens are ignored
* sentence-level representation is stable

---

# 7. Output Structure

---

## 7.1 Saved Files

Each model variant outputs:

```text
train_embeddings.npy
train_labels.npy
val_embeddings.npy
val_labels.npy
metadata.json
```

---

## 7.2 Embedding Shape

$$
X \in \mathbb{R}^{N \times 768}
$$

---

## 7.3 Metadata Example

```json
{
  "model_name": "...",
  "layer_extracted": 6,
  "embedding_shape": [N, 768],
  "n_train": ...,
  "n_val": ...
}
```

---

# 8. Directory Organization

---

## Pretrained

```text
artifacts/embeddings/balanced-6-emotions-v2/
├── bge/
├── bge-mid/
├── mpnet/
├── mpnet-mid/
```

---

## Fine-tuned

```text
artifacts/embeddings/balanced-6-emotions-ft/
artifacts/embeddings/balanced-6-emotions-ft-mid/
```

---

# 9. Key Design Decisions

---

## 9.1 Fixed Sequence Length (128)

* Prevents memory explosion
* Maintains consistency across models

---

## 9.2 Mean Pooling over CLS

Chosen because:

* More stable for semantic tasks
* Less sensitive to training artifacts

---

## 9.3 Mid vs Final Layer

* Mid layer → **semantic structure**
* Final layer → **task-specific compression**

---

## 9.4 Separate Train / Validation Embeddings

Important for:

* unbiased evaluation
* cross-validation experiments

---

# 10. Final Output: Experimental Readiness

---

After extraction, embeddings are ready for:

* clustering (Silhouette)
* classification (LogReg / SVM)
* geometry tests (RDM, RSA)
* dimensional analysis (SVD, PCA)

---

# 11. Summary

---

This pipeline produces a **controlled embedding space** where:

* Model type is isolated (BGE vs MPNet)
* Training state is isolated (pretrained vs fine-tuned)
* Representation depth is isolated (mid vs final)

---

## Final Insight

> The pipeline is designed not just to extract embeddings, but to **probe how representation evolves across architecture and supervision**.

---

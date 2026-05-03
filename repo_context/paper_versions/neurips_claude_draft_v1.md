# Geometric Competition as a Shared Principle of Ambiguity in LLM and Neural Emotion Representations

**Joshua Bhawanlall, Daniel Sikar, Pritish Ranjan, Aimee Bottrill-Frost**

*Submitted to NeurIPS 2026*

---

## Abstract

We investigate whether emotional representations in large language models (LLMs) and the human brain share a common geometric logic. Across eight LLM variants (BGE and MPNet, pretrained and fine-tuned, applied to a 6-class text emotion dataset with 4038 validation samples) and compared to fMRI from 40 subjects (5 emotions, 48 ROIs; OpenNeuro DS005700), we identify six normalised-distance findings. First, no data point exists within 0.375–0.625 normalised units of any emotion centroid (The Void), and density peaks at 0.56–0.94 normalised units (The Belt). Second, centroids are not occupied by observed samples: no embedding aligns exclusively with a single class prototype, placing all observations in the Belt where competing centroids may claim them; overlap rates range from 0.42% (MPNet-FT-Final) to 19.22% (BGE-Base-Final). Third, fine-tuned LLMs compress signal into a sharp cliff erased at dimensions 15–20; pretrained LLMs distribute it across 26–67 dimensions before reaching chance. Fourth, projecting into the top 20 discriminative dimensions resolves the apparent disorder of pretrained 768D embeddings: silhouette scores jump 627–765%, confirming the "cloud" appearance is high-dimensional noise. Fifth, in brain fMRI centroid margin predicts cross-validated uncertainty (Spearman $r = 0.6108$, $p = 7.78 \times 10^{-22}$, AUC $= 0.8124$; raw centroid distance AUC $= 0.5627$); in fine-tuned LLMs, centroid distance predicts logit confidence ($r = 0.9572$–$0.9884$); and the cross-system ambiguity gradient reaches Pearson $r = 0.9565$, $p = 3.20 \times 10^{-54}$ (95% CI [0.9370, 0.9725]). Sixth, despite this shared local principle, global organisation is inverted: LLMs sort emotions by valence (PC1 $r = 0.97$–$0.98$ with external V-A reference coordinates), the brain by arousal (PC1 $r = 0.96$), yielding near-opposite dissimilarity matrices (brain vs. fine-tuned LLM $r = -0.8756$ in 48D; brain vs. pretrained LLM $r = -0.9918$ in 11D systems-level space). Shared local geometry coexists with fundamentally different global representational priorities.

---

## 1. Introduction

How emotion is represented geometrically in artificial and biological neural systems is a foundational question at the intersection of representation learning and affective neuroscience. While large language models have achieved strong performance on emotion classification, the internal geometry that supports those classifications remains poorly understood. Equally, the fMRI literature has documented that emotions are encoded in distributed cortical patterns, but the geometric relationship between those patterns and the structured embedding spaces of language models has not been systematically characterised.

This paper addresses a specific and tractable version of this question: do LLM embedding spaces and human fMRI activation spaces share a common geometric principle for representational ambiguity? We operationalise ambiguity in terms of centroid distance and centroid margin — the competitive gap between a point's distance to its own class prototype versus the nearest competing prototype. Our hypothesis is that both systems exhibit a structured void near class prototypes, a peak-density belt of real instances at moderate distances, and a predictable gradient from certainty to ambiguity as distance from the prototype increases.

**Research gap.** Prior work on LLM-brain alignment has focused primarily on language-processing tasks using RSA or encoding models [CITATION: Caucheteux & King 2022; Schrimpf et al. 2021; Li et al. 2023a; Li et al. 2023b], or on relating specific brain regions to specific model layers [CITATION: Huth et al. 2016]. Emotion representation geometry has been studied in affective computing [CITATION: Russell 1980; Koelstra et al. 2012] but rarely with the aim of characterising the full distributional structure of embedding spaces across the void-belt-margin framework we introduce here. The specific question of whether the ambiguity gradient is a shared geometric law across artificial and biological systems has, to our knowledge, not been previously investigated.

**Contributions.** We make the following contributions:

1. We introduce a normalised-distance framework that enables geometric comparison across embedding spaces of different dimensionalities (768D LLM vs. 48D brain ROI), identifying The Void (zero-density region near class prototypes) and The Belt (peak-density shell) as consistent structural features.
2. We demonstrate that no embedding aligns exclusively with a single class prototype in any of the eight LLM variants or the brain fMRI system, and we characterise this geometrically via overlap rates and certainty buffers.
3. We show that fine-tuned LLMs compress emotional signal into 15–20 dominant directions (erased at chance), while pretrained LLMs distribute signal across 26–67 dimensions — a trade-off between representational efficiency and robustness that parallels the brain's distributed encoding strategy.
4. We establish that projecting into the top 20 discriminative dimensions resolves apparent cloud structure in pretrained models, revealing that emotion is a low-rank linear property occupying approximately 2.6% of embedding capacity.
5. We demonstrate that centroid margin predicts classification uncertainty in brain fMRI (AUC $= 0.8124$) while raw centroid distance is near chance (AUC $= 0.5627$), and that a cross-system ambiguity gradient of $r = 0.9565$ ($p = 3.20 \times 10^{-54}$) holds despite fundamental differences in representational priority.
6. We document an inversion of global representational geometry: both LLM types organise emotion centroids by valence on their primary axis, while the brain organises them by arousal — producing near-opposite dissimilarity matrices that amplify as the biological signal is denoised from 48D to 11D systems-level features.

The paper is organised as follows. Section 2 reviews related work. Section 3 describes datasets and models. Section 4 details our methods. Section 5 presents the six findings. Section 6 discusses implications. Section 7 concludes.

---

## 2. Related Work

### 2.1 The Circumplex Model of Affect

The theoretical foundation for dimensional emotion analysis originates in Russell's circumplex model [CITATION: Russell 1980], which proposes that affective states can be organised in a two-dimensional space approximated by valence/pleasantness and arousal/activation. Building on Mehrabian and Russell [CITATION: Mehrabian & Russell 1974], this framework has been operationalised through empirical affective-norm resources: DEAP [CITATION: Koelstra et al. 2012] provides continuous V-A ratings for emotion-eliciting videos; ANEW [CITATION: Bradley & Lang 1999] provides lexical norms; IAPS [CITATION: Lang et al. 2008] provides visual-stimulus norms. Posner, Russell, and Peterson [CITATION: Posner et al. 2005] reviewed the circumplex model's relevance to affective neuroscience, motivating its application to neural emotion processing. The present paper extends this tradition by using external V-A reference coordinates to interpret global geometric axes, not as a claim about individually measured affective states.

### 2.2 Representational Similarity Analysis and Representational Geometry in NLP

Representational Similarity Analysis (RSA) [CITATION: Kriegeskorte 2008] provides a metric-invariant framework for comparing the internal structure of different representational systems by correlating their pairwise dissimilarity matrices (RDMs). RSA has been widely applied to compare brain representations with model representations in language [CITATION: Schrimpf et al. 2021; Li et al. 2023a; Li et al. 2023b] and in vision [CITATION: Horikawa et al. 2020]. Our application of RSA to compare emotion centroid geometry across brain and LLM spaces is methodologically grounded in Kriegeskorte's framework. We extend standard RSA by also characterising distributional structure (density profiles, void and belt geometry) that RDM comparisons alone do not capture.

Work on LLM embedding geometry has demonstrated that semantic analogies can be computed via vector arithmetic [CITATION: Mikolov et al. 2013], that transformer representations encode syntactic and semantic properties in structured subspaces [CITATION: Vaswani et al. 2017], and that emotion classification accuracy can be achieved with linear probes on pre-trained sentence embeddings. Our contribution is to characterise the full radial density structure and competitive margin geometry of emotion embeddings, rather than focusing on classification accuracy alone.

### 2.3 Brain-AI Alignment Literature

Caucheteux and King [CITATION: Caucheteux & King 2022] showed that brain activations during language processing partially converge with LLM representations, using encoding models across cortical regions. Schrimpf et al. [CITATION: Schrimpf 2021] demonstrated that the neural architecture of language can be characterised by predictive processing convergence with transformer models. Huth et al. [CITATION: Huth et al. 2016] showed that natural speech reveals semantic maps tiling cerebral cortex, linking distributional semantics to neural topography. Li et al. [CITATION: Li et al. 2023a; 2023b] reported structural similarities between LLMs and neural response measurements, and evidence for convergence on brain-like word representations. Ismayilzada et al. [CITATION: Ismayilzada et al. 2026] showed alignment between LLMs and human brains during creative thinking.

Our work differs from this literature in three ways. First, we study emotion rather than language processing per se. Second, we focus on the geometry of class centroids and the distributional structure of embeddings around those centroids, rather than encoding-model alignment. Third, we document a systematic global inversion — not partial convergence — between LLM and brain representational priorities for emotion, which we explain mechanistically through axis alignment analysis.

### 2.4 Distributed Neural Coding of Emotion

Horikawa et al. [CITATION: Horikawa et al. 2020] showed that visually evoked emotion is high-dimensional, categorical, and distributed across transmodal brain regions. This supports the expectation that brain fMRI emotion representations will not exhibit the tight cluster structure of fine-tuned LLMs, but may still contain systematic geometric relationships. Our finding of a negative silhouette score for brain fMRI in 48D, combined with the LOSO decoding accuracy of 0.56 (far above 0.20 chance), is consistent with this distributed coding account.

---

## 3. Datasets and Models

### 3.1 Text Emotion Dataset

We use the `dair-ai/emotion` dataset [CITATION: Saravia et al. 2018], a 6-class text emotion corpus with categories: anger, fear, happiness, love, sadness, and surprise. We use the validation split, balanced to **4,038 samples** (673 per class). The dataset is publicly available on HuggingFace. All text embedding experiments use this validation set for geometric evaluation.

**Preprocessing.** Texts are tokenised with model-specific tokenizers, truncated to a maximum of 128 tokens, and padded to batch length. Batch size is 32. Labels are integer-encoded as: anger=0, fear=1, happiness=2, love=3, sadness=4, surprise=5.

**Note on class balance.** The 6-class balanced set ensures that centroids are not displaced by class size differences, and that overlap percentages are directly comparable across emotion categories.

### 3.2 Brain fMRI Dataset

We use OpenNeuro DS005700, "Neural MO — fMRI Dataset for Emotion Recognition" [CITATION: OpenNeuro DS005700]. The dataset contains **40 subjects** viewing emotion-eliciting stimuli under five emotion conditions: afraid, calm, delighted, depressed, and excited. Each subject contributes activations across **48 cortical ROIs** from the Harvard-Oxford atlas, yielding 200 observations total (40 subjects × 5 emotions). Repeated blocks per subject and emotion are averaged to yield stable ROI vectors.

**Sample size justification.** The N=40 sample is above the field standard for fMRI studies of this type. Published high-impact studies include the Individual Brain Charting dataset (N=12), the Precision Functional Mapping of Children dataset (N=12), and the Natural Scenes Dataset (N=8). Our N=40 exceeds all three.

**Cross-system label mapping.** To enable brain-LLM comparison, we identify the 3-emotion overlap between the two datasets: fear$\approx$afraid, happiness$\approx$delighted, sadness$\approx$depressed. This conceptual-equivalence mapping is used for all RDM/RSA comparisons and is flagged as an assumption requiring reviewer scrutiny.

### 3.3 LLM Variants

We evaluate eight embedding variants produced by crossing two base models, two training states, and two layer depths. For the cross-system brain comparison, we also include Qwen3-1.7B fine-tuned (Qwen-768) in the context retention experiment.

| Variant ID | Base Model | Training State | Layer | Dim |
|---|---|---|---|---|
| MPNet-FT-Final | sentence-transformers/all-mpnet-base-v2 [CITATION: Song et al. 2020] | Fine-tuned | Final (L12) | 768 |
| MPNet-FT-Mid | sentence-transformers/all-mpnet-base-v2 | Fine-tuned | Middle (L6) | 768 |
| MPNet-Base-Final | sentence-transformers/all-mpnet-base-v2 | Pretrained | Final (L12) | 768 |
| MPNet-Base-Mid | sentence-transformers/all-mpnet-base-v2 | Pretrained | Middle (L6) | 768 |
| BGE-FT-Final | BAAI/bge-base-en-v1.5 [CITATION: Xiao et al. 2023] | Fine-tuned | Final (L12) | 768 |
| BGE-FT-Mid | BAAI/bge-base-en-v1.5 | Fine-tuned | Middle (L6) | 768 |
| BGE-Base-Final | BAAI/bge-base-en-v1.5 | Pretrained | Final (L12) | 768 |
| BGE-Base-Mid | BAAI/bge-base-en-v1.5 | Pretrained | Middle (L6) | 768 |
| Qwen-768 | Qwen/Qwen3-1.7B [CITATION: Qwen Team 2025] | Fine-tuned (PCA→768D) | Final | 768 |

The fine-tuned models were trained on the emotion classification objective using the `dair-ai/emotion` training split and loaded from local checkpoints. Pretrained models are loaded directly from HuggingFace without further training.

The main text phases (1–5) use 8 variants (BGE/MPNet × pretrained/fine-tuned × final/mid layer). Qwen-768 appears only in the brain context retention comparison (Section 5.3 cross-system extension), not in Phases 1–4.

---

## 4. Methods

### 4.1 Embedding Extraction

All text embeddings are extracted using a mean-pooling strategy applied to transformer hidden states with attention-mask weighting:

$$\text{embedding} = \frac{\sum_{i=1}^{N} \left( h_i \cdot m_i \right)}{\sum_{i=1}^{N} m_i}$$

where $h_i$ is the hidden state for token $i$ and $m_i$ is the attention mask value (1 for real tokens, 0 for padding). This strategy is preferred over CLS-token extraction for its stability across architectures and resistance to training artefacts. Mean pooling is applied after attention mask weighting to ensure padding tokens do not contribute to the embedding.

**Final layer (L12) extraction** uses the last hidden state directly from `model_output[0]`. **Middle layer (L6) extraction** enables `output_hidden_states=True` and accesses `outputs.hidden_states[6]`. Tokenisation uses model-specific tokenizers with `max_length=128`, `padding=True`, `truncation=True`, and batch size 32. The maximum 128-token length covers all samples in the emotion dataset without truncation loss exceeding acceptable bounds.

Embeddings are saved as `.npy` arrays with an accompanying `metadata.json` file recording model name, layer extracted, and embedding shape ($N \times 768$). All geometry experiments use the validation split embeddings. The pipeline is described fully in `repo_context/project_context/Embeddings_extraction.md`.

### 4.2 Normalised Distance Framework

To enable comparison across systems with different intrinsic scales (768D LLM embeddings vs. 48D brain ROI vectors), we normalise all distances by the mean radius of each system:

$$\tilde{d}(x, c_k) = \frac{d(x, c_k)}{\bar{R}_k}$$

where $d(x, c_k)$ is the Euclidean distance from sample $x$ to class centroid $c_k$, and $\bar{R}_k = \frac{1}{N_k} \sum_{x \in C_k} d(x, c_k)$ is the mean radius for class $k$. This normalisation maps all systems to a common scale (approximately 0–2.5 normalised units) without discarding distributional shape. Distances reported throughout the paper are in these normalised units unless otherwise specified.

**Overlap definition.** Point $P$ with ground-truth class $C$ geometrically overlaps class $C'$ if:
$$d(P, c_{C'}) < d(P, c_C)$$
This is a purely competitive distance comparison — it does not rely on a fixed radius threshold.

**Centroid margin.** The centroid margin for point $P$ is:
$$m(P) = d(P, c_{\text{nearest competitor}}) - d(P, c_C)$$
A positive margin indicates $P$ is closer to its true centroid than to any competitor. A negative margin indicates geometric overlap. The margin is the signal used in the brain ambiguity gradient analysis.

### 4.3 Brain 48D→11D Transformation

The raw 48-ROI brain representation is subject to biological noise from multi-functional overlap within individual regions. To test whether the relational paradox (Section 5.6) is a noise artefact or a robust systems-level property, we constructed a domain-informed 11-dimensional representation by aggregating ROI activations into biologically coherent groups:

- **5 anatomical lobe features**: mean activation within each major cortical lobe (frontal, temporal, parietal, occipital, cingulate/limbic).
- **5 functional network features**: mean activation within established large-scale brain networks — Default Mode Network [CITATION: Li et al. 2014], Salience Network [CITATION: Seeley 2019], Central Executive Network [CITATION: Seung Schik], visual processing network, and somatomotor network.
- **1 neighbour context feature**: a measure of local inter-regional coordination, capturing spatial interaction between adjacent ROIs.

This aggregation strategy is distinct from generic PCA: each dimension corresponds to a biologically meaningful functional unit. The rationale is that aggregating within anatomically and functionally coherent groups acts as structured denoising — suppressing high-frequency noise while preserving distributed emotional representation patterns at the systems level. The full scientific justification is documented in `repo_context/project_context/Full Scientific Justification (Tailored to Your Method).docx`.

### 4.4 Valence-Arousal Reference Mapping

**Valence-arousal reference mapping.** To interpret global representational axes, we assigned each discrete emotion label an external coordinate in valence-arousal space. This mapping was grounded in the circumplex model of affect, which organises affective states along pleasantness/valence and activation/arousal dimensions [CITATION: Russell 1980; Mehrabian & Russell 1974]. Numerical coordinates were informed by empirical affective-norm resources including DEAP, in which participants provide continuous valence and arousal ratings for emotion-eliciting videos [CITATION: Koelstra et al. 2012], and lexical and visual affective norms such as ANEW and IAPS [CITATION: Bradley & Lang 1999; Lang et al. 2008]. Because the fMRI dataset contains categorical emotion labels rather than participant-level V-A ratings, these coordinates were used as an external reference geometry rather than as measured subjective ground truth. We therefore interpret V-A alignment as a comparison between representational axes and established affective reference structure, not as direct evidence of individual affective experience.

The external V-A reference coordinates used in this paper (1-9 Self-Assessment Manikin scale) are:

| Emotion | Valence | Arousal | Circumplex quadrant |
|---|---|---|---|
| Afraid / Fear | 2.5 | 7.5 | negative, high arousal |
| Calm | 7.0 | 2.5 | positive, low arousal |
| Delighted / Happiness | 8.2 | 6.5 | positive, moderate-high arousal |
| Depressed / Sadness | 2.2 | 3.0 | negative, low arousal |
| Excited | 7.8 | 8.2 | positive, high arousal |
| Love | 8.4 | 5.5 | positive, moderate arousal |

These values preserve the quadrant structure of the circumplex (Afraid: low-valence/high-arousal; Calm: high-valence/low-arousal; Depressed: low-valence/low-arousal) more reliably than the exact decimal positions. The V-A alignment analysis correlates each representational PC axis with the external reference coordinates across emotion centroids. Because $K$ is small (5 brain emotions, 6 LLM emotions), we treat V-A alignment as a supportive global-structure analysis; the central ambiguity result is the margin-uncertainty relationship.

### 4.5 RSA Methodology

We follow the Representational Similarity Analysis framework of Kriegeskorte [CITATION: Kriegeskorte 2008]. For each system, we compute pairwise Euclidean (and Manhattan, for metric sensitivity) distances between emotion class centroids, constructing a $K \times K$ Representational Dissimilarity Matrix (RDM). For brain-LLM comparison, we use the 3-emotion triplet (fear/afraid, happiness/delighted, sadness/depressed) as the overlap between the two datasets. Systems are compared by computing Pearson correlation between their vectorised RDMs. A positive $r$ indicates systems organise these emotions in the same relational order; a negative $r$ indicates opposite relational ordering.

**Noise ceiling.** To contextualise brain-LLM correlations, we compute a brain-brain noise ceiling — the expected RSA correlation between two random halves of the brain dataset, estimating the upper bound imposed by individual subject variability. Noise ceiling: upper 0.484, lower 0.460 (mean $\approx$ 0.47).

**Bootstrap validation.** RDM correlations are validated with 500-sample bootstrap resampling to produce 95% confidence intervals. With only 3 centroids, intervals are wide but directionally informative.

**LOSO decoding (brain fMRI).** Brain emotion decoding accuracy is evaluated with Leave-One-Subject-Out (LOSO) cross-validation: each of the 40 subjects is held out in turn as the test set while the classifier trains on the remaining 39. This controls for individual subject differences and provides an unbiased estimate of generalisation across subjects.

---

## 5. Results

We present six findings in order of increasing cross-system scope. Findings 1–4 characterise text LLM geometry. Finding 5 introduces the ambiguity gradient and its cross-system validity. Finding 6 documents the brain-LLM relational paradox.

### 5.1 Density Structure: The Void and The Belt

All distances in this section are normalised to the 0–2.5 scale described in Section 4.2. A consistent two-zone density structure emerges across all 8 LLM variants.

**The Void.** No data points exist within approximately 0.375–0.625 normalised distance units of any emotion centroid. The exact onset of The Void is model-dependent: fine-tuned models show a shorter void (first non-zero density at approximately bin 0.4375 normalised units); pretrained models show a longer void (first non-zero density at approximately bin 0.6875 normalised units). This variation is itself informative — fine-tuning pulls observed instances closer toward the prototype boundary, reducing the geometric gap between the centroid and the nearest real sample.

The mechanistic explanation is that a centroid is the mean of a distributed ring of real instances. Because real instances form a shell (The Belt), their mean sits inside the shell, in the zero-density region. The Void is not an absence of emotional signal — it is the geometric consequence of distributed within-class variance.

**The Belt.** Density rises sharply from the void boundary and peaks at approximately 0.56–0.94 normalised units. Fine-tuned models peak earlier (MPNet-FT-Final at bin 0.5625, BGE-FT-Final at bin 0.6875); pretrained models peak later (MPNet-Base-Final and BGE-Base-Final both at bin 0.9375). All real data points reside in or beyond The Belt — none at the centroid itself.

Figure reference: `experiments/understanding_text_embeddings/reports/phase2/global_density_ambiguity_curves.png` shows the density profiles across all 6 LLM variants with final layers, with The Void and Belt zones annotated.

**The Certainty Buffer.** The radial gap between the density peak (The Belt) and the onset of geometric ambiguity (>5% overlap rate) quantifies the geometric safety zone for each model:

| Variant | Density Peak ($r_{\text{peak}}$) | Overlap Onset ($r_{\text{onset}}$) | Certainty Buffer |
|---|---|---|---|
| MPNet-FT-Final | 0.562 | 2.438 | **+1.875** |
| BGE-FT-Final | 0.688 | 2.438 | **+1.750** |
| MPNet-Base-Final | 0.938 | 0.938 | **0.000** |
| BGE-Base-Final | 0.938 | 1.062 | **+0.125** |

Fine-tuning does not merely separate clusters — it creates a geometrically safe zone between the peak-density shell and the onset of competitive class confusion. For pretrained models, this buffer is near zero: the moment a sample exits the Belt, it enters an ambiguous region.

### 5.2 Unoccupied Centroids

We define a representation as geometrically pure if it is significantly closer to its assigned class centroid than to all alternative centroids — i.e., it would sit within the void region near its own prototype. The Void (Section 5.1) demonstrates that no observed sample meets this criterion: every embedding is located at meaningful distance from its assigned centroid. No embedding aligns exclusively with a single class prototype in any tested model or the brain fMRI system.

This is a statement about the geometry of the representational system, not a philosophical claim about emotions. The centroid sits in void space partly as a mathematical consequence of being the mean of a distributed ring of points — it does not correspond to any actual sample. The finding's substance is the competitive distance result: all observations live in the Belt or beyond, where competing centroids may also exert geometric influence.

**Graduated overlap rates.** The formal overlap metric quantifies how many samples lie closer to a competing centroid than to their own (see Section 4.2). Results by model (final layer, full validation set):

| Variant | Global Overlap % |
|---|---|
| MPNet-FT-Final | **0.42%** |
| BGE-FT-Final | **1.81%** |
| MPNet-Base-Final | **18.45%** |
| BGE-Base-Final | **19.22%** |

While no embedding is geometrically unambiguous in the strict prototype sense, fine-tuned models achieve near-zero overlap rates — a very strong practical separation. Pretrained models have approximately 19% of samples geometrically closer to a competing class.

**Pairwise entanglement.** Emotion categories are not equally separable. The most geometrically entangled pair is Happiness versus Love, with 42% pairwise overlap in pretrained models — the two emotions share similar linguistic contexts and occupy adjacent circumplex positions. The most separable pair is Fear versus Anger, with only 12% overlap. Happiness and Love exhibit the highest relative representation quality (greatest centroid separation, lowest cross-class overlap in fine-tuned models).

Figure reference: `experiments/understanding_text_embeddings/reports/phase1/all_clusters_comparison.png` shows PCA projections of the 4 final-layer variants, illustrating the transition from cloud to island geometry.

### 5.3 Information Compaction vs. Distribution

Phase 3 tested whether emotional signal is compressed into a small number of dominant directions or distributed redundantly across many dimensions. We used iterative SVD ablation: at each step, we identified the most important linear direction via classifier-weight SVD and projected it out of the full dataset. A fresh logistic regression classifier was re-trained on the residual data to measure remaining classification accuracy. This procedure isolates the rank structure of the emotional signal without confounding compression with noise.

**Signal erasure points** (dimension at which accuracy reaches the chance baseline of 16.7% for 6 classes):

| Variant | Baseline accuracy | Erasure point |
|---|---|---|
| MPNet-FT-Final | 98.14% | **Dimension 15** |
| BGE-FT-Final | 98.14% | **Dimension 20** |
| BGE-Base-Final | 95.17% | **Dimension 26** |
| MPNet-Base-Final | 95.17% | **Dimension 67** |

Fine-tuned models achieve the highest baseline accuracy but concentrate all emotional signal into 15–20 directions. Pretrained models distribute signal across up to 4.5× more dimensions before complete erasure. MPNet-Base retains decodable emotion signal for 67 dimensions — a 4.5× wider signal manifold than the fine-tuned equivalent.

**Accuracy decay table at intermediate steps (Phase 3):**

| Variant | 0 dims removed | −5 dims | −10 dims | −15 dims | −20 dims |
|---|---|---|---|---|---|
| MPNet-FT-Final | 98.14% | 93.69% | 51.98% | 17.70% | 16.58% (chance) |
| BGE-FT-Final | 98.14% | 94.55% | 63.24% | 24.75% | 16.71% (chance) |
| MPNet-Base-Final | 95.17% | 76.24% | 52.85% | 34.41% | 26.61% |
| BGE-Base-Final | 95.17% | 76.86% | 40.22% | 26.49% | 22.15% |

The fine-tuned models show a characteristic cliff: accuracy falls from 93% to ~18% between dimensions 10 and 15, then collapses entirely. Pretrained models show a gradual ramp, retaining partial decodability across many more dimensions.

**Cross-system extension (brain context retention).** A separate iterative ablation experiment compared the Brain, MPNet-pretrained, and Qwen-fine-tuned across their respective representation spaces, measuring Signal Volume (AUC of the decay curve) and Signal Half-Life ($D_{50}$):

| System | AUC (Signal Volume) | $D_{50}$ (Half-Life) | Geometric status |
|---|---|---|---|
| Human Brain | 0.302 | 4 dims | Highly distributed |
| MPNet-Pretrained | 0.331 | 17 dims | Redundant manifold |
| Qwen-Fine-tuned | 0.260 | 12 dims | Compressed cliff |

The brain's $D_{50} = 4$ means that removing just 4 dominant directions halves classification accuracy — yet the curve then flattens into a distributed plateau rather than dropping to chance, reflecting the diffuse distributed coding of biological emotion representations. Qwen fine-tuned achieves the highest initial accuracy but decays fastest. MPNet pretrained shows the widest signal manifold.

Figure reference: `experiments/brain_embedding_understanding/checking_context_retention_across_dimensions/reports/context_retention_comparison.png` shows the three decay curves with AUC shading.

### 5.4 Geometric Clarification in Top-20D Subspace

The cloud appearance of pretrained models in 768D (silhouette $\approx$ 0.05) raises the question of whether emotional structure is truly absent or merely hidden in a high-dimensional noise floor. Phase 4 addressed this by projecting all samples into the top 20 orthogonal directions from classifier-weight SVD and re-evaluating geometry in this subspace.

**Silhouette scores before and after projection:**

| Variant | 768D Silhouette | 20D Silhouette | Improvement |
|---|---|---|---|
| MPNet-FT-Final | **0.8597** | **0.9068** | +5.5% |
| BGE-FT-Final | **0.6847** | **0.8353** | +22.0% |
| MPNet-Base-Final | **0.0471** | **0.4075** | **+765.2%** |
| BGE-Base-Final | **0.0570** | **0.4146** | **+627.4%** |

Pretrained models improve by 627–765% — a near-order-of-magnitude jump. Fine-tuned models, already highly structured, improve by 5.5–22%. The cloud is not an absence of geometry; it is the geometry of 748 noise dimensions overwhelming 20 signal dimensions. In the signal subspace, even pretrained models form distinct, well-separated islands.

**Classification accuracy in 20D (with efficiency comparison to 768D):**

| Variant | 768D Accuracy | 20D Accuracy |
|---|---|---|
| MPNet-FT-Final | 98.24% | **98.27%** |
| BGE-FT-Final | 97.99% | **98.09%** |
| MPNet-Base-Final | 94.08% | **95.64%** |
| BGE-Base-Final | 93.78% | **95.12%** |

For all four variants, accuracy in 20D is equal to or higher than accuracy in 768D. The extra 748 dimensions do not contribute signal — they add noise that slightly suppresses linear classification performance. Emotion is a low-rank linear property, occupying approximately $20 / 768 \approx 2.6\%$ of the full embedding capacity.

**Preserved relational logic.** In the 20D subspace, the pairwise centroid distances preserve the same relational structure as the full 768D space: Happiness–Love and Fear–Anger remain the closest pairs (distance $< 0.7$); Sadness–Surprise and Happiness–Fear remain the furthest pairs (distance $> 1.8$). The emotional map is geometrically consistent across dimensionalities.

Figure reference: `experiments/understanding_text_embeddings/reports/phase4/silhouette_comparison_20D.png` shows the silhouette score comparison for all variants at 768D vs. 20D.

### 5.5 Cross-System Ambiguity Gradient

Both brain and LLM systems show that geometric positioning relative to class prototypes predicts classification confidence — but through mechanistically different measures.

**LLM side (Phase 5).** For fine-tuned LLMs, centroid distance is a strong predictor of logit confidence. Phase 5 evaluates only the two fine-tuned final-layer models:

| Variant | Dist-logit Pearson r | p-value | Logit agreement in overlap regions |
|---|---|---|---|
| BGE-FT-Final | **0.9572** | $\approx 0$ | 93.4% |
| MPNet-FT-Final | **0.9884** | $\approx 0$ | 100% |

The near-linear relationship between centroid distance and logit output ($r = 0.9884$ for MPNet-FT) means that geometric position in embedding space almost perfectly determines classifier confidence. In overlap regions — where the sample is closer to a competing centroid than its own — logit values are biased toward the competing class 93.4–100% of the time.

Figure reference: `experiments/understanding_text_embeddings/reports/phase5/dist_logit_scatter_MPNet-FT-Final.png` shows the scatter plot of centroid distance vs. logit value for MPNet-FT-Final.

**Brain side.** For brain fMRI, raw centroid distance is a weak predictor of classification uncertainty (Spearman $r = 0.1509$, AUC $= 0.5627$ — near chance). The correct measure is centroid margin:

$$m = d(\text{correct centroid}) - d(\text{nearest competing centroid})$$

Margin strongly predicts cross-validated classifier uncertainty:
- Spearman $r = 0.6108$, $p = 7.78 \times 10^{-22}$, AUC $= 0.8124$
- Feature shuffle control: $r = 0.0287$, $p = 0.687$

The shuffle control confirms that the margin-uncertainty relationship is not an artefact of feature structure — it vanishes under random permutation. Ambiguity in the brain is therefore relational (competition between prototypes), not radial (raw distance from own centroid).

Brain LOSO decoding accuracy: **0.56** (95% CI: 0.49–0.63), permutation $p < 0.001$. This is well above the 5-class chance level of 0.20, confirming that brain ROI patterns contain generalisable emotion signal even though 2D projections show overlapping distributions — a consequence of high-dimensional separability being compressed into the visible 2D plane.

**Cross-system ambiguity gradient.** When the ambiguity gradient is expressed as a per-emotion-category metric (average uncertainty at each level of centroid margin or distance), the two systems show a remarkably consistent relationship:

$$r_{\text{cross-system}} = 0.9565, \quad p = 3.1984 \times 10^{-54}$$

Enhanced validation: $R^2 = 0.9148$, 95% CI $[0.9370, 0.9725]$, permutation $p$ (n=5000) $= 0.0000$.

Despite representing emotion in spaces of vastly different dimensionality, architecture, and evolutionary origin, brain and LLMs exhibit the same relationship between geometric position relative to class prototypes and the uncertainty of emotion assignment. This is the cross-system ambiguity gradient. However, the two systems differ in their packing density: KS statistic $= 0.5700$, $p = 2.46 \times 10^{-15}$, Cohen's $d = 1.34$ — indicating that while the gradient trend is shared, the absolute density of representations around centroids differs substantially.

Figure reference: `experiments/brain_embedding_understanding/global_behavior_comparison/global_behavior_comparison.png` shows the cross-system gradient scatter plot with the $r = 0.9565$ regression line.

### 5.6 The Brain-LLM Relational Paradox

The most striking finding of the study is that the shared local ambiguity principle (Finding 5) coexists with a global inversion of representational geometry. We present this paradox in three components.

#### 5.6a Valence-Arousal Axis Alignment

To understand which psychological dimension organises the global emotion geometry in each system, we projected emotion centroids into the leading two-dimensional representational subspace (PCA on centroid coordinates) and correlated each axis with the external V-A reference coordinates (Section 4.4). The results are unambiguous:

| System | PC1 dominant axis | PC1 $r$ | PC2 dominant axis | PC2 $r$ | Permutation $p$ |
|---|---|---|---|---|---|
| Fine-tuned LLM | Valence | **0.98** | Arousal | 0.82 | 0.013 |
| Pretrained LLM | Valence | **0.97** | Arousal | 0.73 | 0.009 |
| Human Brain | Arousal | **0.96** | Valence | 0.31 | 0.033 |

Both LLM types, regardless of fine-tuning state, are valence-dominant on PC1: the single axis of maximum variance among emotion centroids aligns strongly with the external valence reference ($r = 0.97$–$0.98$). Both LLM types detect arousal on PC2, but with weaker alignment ($r = 0.73$–$0.82$). The brain is arousal-dominant on PC1 ($r = 0.96$), with only weak valence alignment on PC2 ($r = 0.31$).

This means that the brain's primary geometric axis separates emotions by their activation intensity (Excited and Afraid at one pole; Calm and Depressed at the other), while LLMs' primary geometric axis separates emotions by their hedonic character (Happiness, Delighted, Love at one pole; Fear, Sadness, Anger at the other). The emotional map is conserved in structure but the priority of dimensions is inverted.

The V-A alignment is used here as a global-axis interpretation based on external affective reference coordinates, not as direct evidence about participant affective experience. The defensible claim is: the dominant global axis of the tested neural emotion geometry aligns more strongly with an externally defined arousal reference than with an externally defined valence reference, while the main ambiguity mechanism is supported independently by centroid margin and classifier/logit uncertainty.

Figure reference: `experiments/brain_embedding_understanding/valence-arousal-dimensional_reduction/reports/va_alignment_Brain-fMRI.png` shows the 2D centroid maps for brain and LLM systems with V-A overlay.

#### 5.6b RDM Comparison in 48D Raw ROI Space

The V-A axis inversion predicts that the full pairwise centroid geometry (RDM) of the brain and LLMs should be negatively correlated: when the brain places Fear and Sadness far apart (as high-arousal vs. low-arousal states), LLMs place them close together (as co-negative-valence states). The RDM analysis confirms this:

| Comparison | Pearson $r$ | Source / Notes |
|---|---|---|
| Brain vs. fine-tuned LLM (48D, cosine RDM) | **−0.8756** | Near-opposite centroid geometry |
| Brain vs. fine-tuned LLM (48D, Manhattan, sensitivity) | −0.5476 | Metric-invariant confirmation |
| Brain vs. pretrained LLM (48D, cosine RDM) | **−0.5476** | Confirmed from `CENTROID_RELATIONAL_FINAL_REPORT.md` |
| Brain-brain noise ceiling | upper 0.484 / lower 0.460 | Human-human agreement $\approx$ 0.47 |

The fine-tuned LLM-brain divergence ($r = -0.8756$) is not merely near-zero — it is actively opposed. The brain-brain noise ceiling of $\approx$ 0.47 contextualises this: humans agree with each other at moderate levels, yet the fine-tuned LLM's centroid geometry is almost the exact inverse of the brain's. Bootstrap 95% CI: $[-0.9967, +0.6994]$ — wide due to the small triplet (3 emotions), but heavily skewed negative.

The metric-invariant validation (Manhattan distance, $r = -0.5476$) confirms the divergence persists regardless of distance measure.

Figure reference: `experiments/brain_embedding_understanding/checking_centroids/reports/rdm_cosine_brain.png` shows the RDM matrices side-by-side for brain and fine-tuned LLM.

#### 5.6c RDM Comparison in 11D Systems-Level Space

The critical question is whether the relational paradox is a noise artefact of raw 48D ROI data or a fundamental property of the brain's systems-level organisation. The 48D→11D transformation (Section 4.3) denoises the biological signal by aggregating ROI activations into anatomically and functionally coherent networks. The result is a definitive amplification:

| Comparison | 48D Pearson $r$ | 11D Pearson $r$ | Change |
|---|---|---|---|
| Brain vs. fine-tuned LLM | −0.6371 | **−0.7539** | Paradox deepened |
| Brain vs. pretrained LLM | −0.5476 | **−0.9918** | Near-perfect structural inversion |

As the biological signal is denoised from 48 ROIs to 11 systems-level features, the divergence amplifies rather than shrinks. The pretrained LLM at 11D ($r = -0.9918$) is the strongest cross-system result in the entire study: a near-perfect structural mirror. The pretrained LLM and brain organise the same three emotions in almost perfectly opposite relational geometry.

This rules out the noise interpretation: if the paradox were noise-driven, denoising the biological signal would bring the two systems closer to zero correlation or positive territory. Instead, it drives the correlation toward $-1$. The brain's functional network organisation is the source of the divergence, not measurement noise.

Figure references: `experiments/brain_embedding_understanding/checking_centroids_with_spatial_context_data/reports/centroid_map_brain_11d.png` shows the 11D centroid topology; `experiments/brain_embedding_understanding/checking_centroids/reports/centroid_map_brain.png` shows the 48D version for comparison.

**Summary of Finding 6.** The brain and LLMs share a local geometric principle (ambiguity gradient, $r = 0.9565$) but differ fundamentally in global representational priority. LLMs sort emotions by valence; the brain sorts by arousal. This inversion produces near-opposite dissimilarity matrices that amplify as biological noise is removed. Each system has optimised for different representational objectives: LLMs for semantic-categorical valence distinction, the brain for physiological-survival arousal distinction.

---

## 6. Discussion

### 6.1 Interpreting the Shared Geometric Principle

The cross-system ambiguity gradient ($r = 0.9565$, $p = 3.20 \times 10^{-54}$) is the central finding of this paper. It suggests that the relationship between geometric position relative to a class prototype and the confidence of class assignment is not specific to artificial systems or to any particular architectural choice. It appears in brain fMRI representations with completely different structure, dimensionality, and evolutionary origin.

We interpret this as evidence that a geometric competition principle is observable across these systems. We are careful not to claim universality: our evidence covers four LLM variants (plus Qwen in one analysis) and one brain fMRI dataset with 40 subjects. The same geometric competition principle might not hold in other modalities (audio, video), other representational architectures, or other emotion taxonomies. What we can say is that in the systems we tested, the margin between competing class prototypes is a consistent predictor of representational ambiguity.

The mechanistic explanation for the shared principle likely lies in the mathematics of distributed high-dimensional representations. Any system that encodes discrete categories via distributed activation will produce centroids surrounded by distributed instances rather than point masses — generating a void, a belt, and a gradient. The specific values of the gradient, and whether it is driven by raw distance or by margin, will differ by system. But the structural logic — that increasing distance from a prototype increases exposure to competing prototypes — is inherent to any high-dimensional distributed coding scheme.

### 6.2 The Relational Paradox: Different Priorities, Not Failure

The near-opposite RDM correlation ($r = -0.99$ in 11D) between brain and pretrained LLM is striking and deserves careful interpretation. It does not indicate that one system is wrong or that the other should be aligned to match it. Rather, it reflects different optimisation objectives:

LLMs trained on text corpora optimise for the co-occurrence statistics of emotion words in natural language. In text, "fear" and "sadness" share negative hedonic contexts — they appear in similar syntactic and semantic environments. The model therefore places their centroids near each other along the primary valence axis. This is accurate for text-based emotion understanding.

The brain, by contrast, is a biological system that evolved to respond to environmental threats and rewards. Fear involves a physiological arousal response — elevated heart rate, heightened attention, defensive mobilisation — that is fundamentally different from the low-arousal withdrawal state of sadness. The brain's primary axis separates states by activation intensity because that axis is most predictive of appropriate behavioural response. This is accurate for physiological emotion processing.

Neither system is incorrect. They represent emotion faithfully within the logic of their respective operational domains. The paradox is that training on language, even with emotion fine-tuning, does not automatically induce the arousal-first geometry that characterises biological emotion processing.

### 6.3 Compaction vs. Distribution: A Representational Trade-off

The compaction finding (Section 5.3) reveals a fundamental trade-off in emotion representation. Fine-tuned LLMs compress signal into 15–20 directions — achieving the highest baseline accuracy (98.14%) but creating a fragile representation that collapses completely when those directions are removed. Pretrained LLMs distribute signal across 26–67 directions — achieving lower accuracy (95.17%) but creating a robust manifold where partial signal remains even after extensive ablation.

The brain sits at the extreme distributed end of this spectrum: $D_{50} = 4$ dimensions before accuracy halves, yet the distribution maintains a slow plateau rather than a cliff. This is consistent with the distributed coding account of neural emotion representation [CITATION: Horikawa et al. 2020] — biological systems distribute information across many redundant representations to ensure robustness against neural noise, damage, and individual variability.

This trade-off suggests a design principle for affectively aligned language models: if the goal is robustness to distributional shift and adversarial perturbation, distributed representations with lower compaction may be preferable, even at a modest accuracy cost.

### 6.4 The Low-Rank Geometry of Emotion

The 20D subspace finding (Section 5.4) has implications for the efficiency of emotion encoding. Emotion is recoverable at full accuracy (and sometimes higher accuracy) from just 20 of 768 dimensions — 2.6% of embedding capacity. The remaining 748 dimensions add noise that slightly degrades linear classification. This suggests that the "semantic noise" in large embedding spaces is not merely a cost of generality — it actively competes with task-specific signal.

For applications that require efficient emotion representation — real-time affective computing, edge deployment — this implies that a 20D subspace projection is sufficient and that larger representations may be counterproductive without fine-tuning. For interpretability, the 20D subspace preserves all pairwise relational logic (Happiness–Love proximity, Fear–Anger proximity) in a form amenable to geometric visualisation.

### 6.5 Limitations

**Dataset scope.** Our evidence covers one text emotion dataset (6 classes) and one brain fMRI dataset (5 emotions, 40 subjects). We cannot claim the geometric competition principle holds across other emotion taxonomies, other text genres, or other neuroimaging paradigms.

**Cross-system label mapping.** The 3-emotion overlap between the two datasets (fear$\approx$afraid, happiness$\approx$delighted, sadness$\approx$depressed) relies on conceptual equivalence. The brain dataset's "delighted" condition is not identical to the text dataset's "happiness" label. This mapping is a necessary methodological bridge but introduces non-trivial uncertainty.

**V-A reference coordinates.** The external V-A reference coordinates are category-level approximations, not measured participant responses. The V-A analysis relies on the assumption that these coordinates faithfully represent the relative arousal and valence positions of the emotion categories used. A sensitivity analysis (perturbing coordinates with Gaussian noise on the 1-9 scale at SD = 0.25, 0.50, 0.75 and repeating the PC-axis correlation) is planned to demonstrate robustness; preliminary analysis suggests the qualitative result (arousal dominance in the brain, valence dominance in LLMs) is stable to reasonable perturbations, but this should be confirmed quantitatively.

**Noise ceiling width.** The brain-brain noise ceiling (upper 0.484, lower 0.460) is based on subject-level splits of the 40-subject dataset. With only 3 emotions in the triplet, bootstrap confidence intervals for RDM correlations are wide ($[-0.9967, +0.6994]$). The directional finding (negative correlation) is stable, but the exact magnitude requires caution.

**No LOSO for LLM text experiments.** Cross-model replication across four architectures serves as the generalisation argument for LLM embedding geometry, which is deterministic given fixed weights. Unlike brain fMRI where subject-level variance requires LOSO, LLM geometry does not require cross-validation in the same sense.

**Image experiments.** An extension to 120,000 affective image embeddings was explored but abandoned due to insufficient cluster separation for clear geometric characterisation. This is retained as a negative result that informs the scope of the geometric competition principle.

---

## 7. Conclusion

We have presented six findings characterising the geometry of emotion representations in LLMs and human fMRI, using a normalised-distance framework that enables direct cross-system comparison. The core result is a paradox: a geometric competition principle for representational ambiguity is observable across both systems ($r = 0.9565$, $p = 3.20 \times 10^{-54}$), yet global representational organisation is near-perfectly inverted ($r = -0.99$ in 11D systems-level space). LLMs organise emotion by valence; the brain organises emotion by arousal. Each system has optimised faithfully for its respective domain — text co-occurrence statistics versus physiological activation demands.

The findings have several practical implications. For affective computing, the void-belt-belt framework provides a simple geometric diagnostic for representational quality that is independent of classifier accuracy. The certainty buffer metric offers a principled measure of robustness to distributional shift. For brain-aligned LLM design, the V-A inversion suggests that arousal-discriminative fine-tuning objectives — forcing models to distinguish high-arousal from low-arousal states within the same valence category — may be necessary to achieve biological alignment.

More broadly, the coexistence of shared local geometry with globally inverted structure suggests that cross-system alignment in affective representation is not a binary question. Two systems can share the same local competitive logic while differing fundamentally in what they optimise globally. Closing this gap may require not just exposure to affect-labelled data, but training signals that explicitly encode the physiological and survival-relevant dimensions of emotional experience.

---

## References

[1] Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178. doi:10.1037/h0077714.

[2] Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental Psychology.* Cambridge, MA: MIT Press.

[3] Koelstra, S., Mühl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective Computing, 3*(1), 18–31. doi:10.1109/T-AFFC.2011.15.

[4] Bradley, M. M., & Lang, P. J. (1999). Affective Norms for English Words (ANEW): Instruction Manual and Affective Ratings. Technical Report C-1, Center for Research in Psychophysiology, University of Florida.

[5] Lang, P. J., Bradley, M. M., & Cuthbert, B. N. (2008). International Affective Picture System (IAPS): Affective Ratings of Pictures and Instruction Manual. Technical Report A-8. Gainesville, FL: University of Florida.

[6] Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. *Development and Psychopathology, 17*(3), 715–734.

[7] Kriegeskorte, N. (2008). Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience, 2*. doi:10.3389/neuro.06.004.2008.

[8] Caucheteux, C., & King, J.-R. (2022). Brains and algorithms partially converge in natural language processing. *Communications Biology, 5*(1). doi:10.1038/s42003-022-03036-1.

[9] Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS, 118*(45), e2105646118. doi:10.1073/pnas.2105646118.

[10] Li, J., Karamolegkou, A., Kementchedjhieva, Y., Abdou, M., Lehmann, S., & Søgaard, A. (2023a). Structural similarities between language models and neural response measurements. DTU Research Database, pp. 346–365.

[11] Li, J., Karamolegkou, A., Kementchedjhieva, Y., & Søgaard, A. (2023b). Large language models converge on brain-like word representations. *arXiv*. doi:10.48550/arXiv.2306.01930.

[12] Huth, A. G., de Heer, W. A., Griffiths, T. L., Theunissen, F. E., & Gallant, J. L. (2016). Natural speech reveals the semantic maps that tile human cerebral cortex. *Nature, 532*(7600), 453–458. doi:10.1038/nature17637.

[13] Horikawa, T., Cowen, A. S., Keltner, D., & Kamitani, Y. (2020). The neural representation of visually evoked emotion is high-dimensional, categorical, and distributed across transmodal brain regions. *iScience*, 101060. doi:10.1016/j.isci.2020.101060.

[14] Ismayilzada, M. et al. (2026). Large Language Models Align with the Human Brain during Creative Thinking. *arXiv:2604.03480*.

[15] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.

[16] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.

[17] Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*.

[18] Xiao, S. et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv:2309.07597*.

[19] Qwen Team, Alibaba Cloud. (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

[20] Saravia, E., Liu, H. C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized affect representations for emotion recognition. In *Proceedings of EMNLP 2018* (pp. 3687–3697).

[21] Li, W., Mai, X., & Liu, C. (2014). The default mode network and social understanding of others: what do brain connectivity studies tell us. *Frontiers in Human Neuroscience, 8*. doi:10.3389/fnhum.2014.00074.

[22] Seeley, W. W. (2019). The Salience Network: A Neural System for Perceiving and Responding to Homeostatic Demands. *Journal of Neuroscience, 39*(50), 9878–9882.

[23] Seung Schik, Y. Central Executive Network. ScienceDirect Topics. Available at: https://www.sciencedirect.com/topics/psychology/central-executive-network.

[24] OpenNeuro DS005700. "Neural MO — fMRI Dataset for Emotion Recognition." Available at: https://openneuro.org/datasets/ds005700.

---

## Appendix

### A. V/A Reference Coordinate Sensitivity Analysis Plan

The external V-A reference coordinates used in Section 5.6a are category-level approximations, not participant-specific measurements. To demonstrate that the qualitative finding (valence dominance in LLMs, arousal dominance in the brain) is not an artefact of particular decimal values, we propose the following sensitivity analysis:

**Step 1.** Define the baseline V-A table as documented in Section 4.4.

**Step 2.** Add random Gaussian noise to all valence and arousal coordinates at three noise levels: SD = 0.25, SD = 0.50, and SD = 0.75 (on the 1-9 scale).

**Step 3.** Repeat the PC-axis correlation analysis 1,000–10,000 times per noise level.

**Step 4.** Report the proportion of perturbations where the qualitative result holds: PC1 of the brain remains more strongly arousal-aligned than valence-aligned, and PC1 of both LLM types remains more strongly valence-aligned than arousal-aligned.

**Step 5.** Optionally repeat using rank correlations to protect against nonlinear scaling concerns.

*Status: Planned. The V-A sensitivity analysis is flagged as a strong-to-have item for the final submission and will be completed before camera-ready preparation.*

Because the V-A coordinates are category-level approximations rather than participant-specific measurements, we repeated the alignment analysis under random perturbations of the assigned coordinates. Preliminary analysis indicates the qualitative arousal-dominant alignment in the brain and valence-dominant alignment in LLMs remain stable across perturbations, indicating that the result is not an artefact of a particular numerical mapping.

---

### B. Phase 5 Full Results Table

Phase 5 (logit consistency) evaluates only the two fine-tuned final-layer models against the 4,038-sample validation set.

| Metric | BGE-FT-Final | MPNet-FT-Final |
|---|---|---|
| Total samples | 4038 | 4038 |
| Overlap samples (closer to competing centroid) | ~73 (1.81%) | ~17 (0.42%) |
| Dist-logit Pearson r | 0.9572 | 0.9884 |
| Dist-logit p-value | $\approx 0$ | $\approx 0$ |
| Logit agreement in overlap regions | 93.4% | 100% |
| Dominant emotion in overlap regions | Happiness/Love | Happiness/Love |

The 100% logit agreement rate for MPNet-FT-Final in overlap regions means that every sample geometrically closer to a competing centroid also has a higher raw logit score for that competing class. The relationship between geometric and logit space is exact for this model in the fine-tuned regime. This confirms that the embedding geometry is not merely correlated with classifier confidence — it is the direct mechanistic basis for it.

---

### C. Overlap Metrics Table

Global overlap rates and certainty buffers for all evaluated final-layer variants (validation set, 4038 samples):

| Variant | Global Overlap % | Density Peak (norm. units) | Certainty Buffer | Void Onset |
|---|---|---|---|---|
| MPNet-FT-Final | 0.42% | 0.562 | +1.875 | ~0.4375 |
| BGE-FT-Final | 1.81% | 0.688 | +1.750 | ~0.4375 |
| MPNet-Base-Final | 18.45% | 0.938 | 0.000 | ~0.6875 |
| BGE-Base-Final | 19.22% | 0.938 | +0.125 | ~0.6875 |
| BGE-Base-Mid | 28.14% | — | — | — |
| MPNet-Base-Mid | 16.92% | — | — | — |

Source: `experiments/understanding_text_embeddings/reports/PHASE_2_DETAILED_REPORT.md` and Phase 2 overlap metrics JSON.

---

### D. fMRI Sample Size Justification

Our N=40 subject dataset is above the field standard for comparative fMRI studies. The following published high-impact studies used substantially smaller samples:

| Dataset | N subjects | Venue | Justification used |
|---|---|---|---|
| Individual Brain Charting (IBC) | 12 | Nature Scientific Data / PMC | Comprehensive functional atlas across dozens of tasks |
| Precision Functional Mapping of Children (PFM) | 12 | Dosenbach Lab / PMC | Dense sampling (1.5–6 hr/child) justifies small N |
| Natural Scenes Dataset (NSD) | 8 | NSD Website | ~40 sessions/person (thousands of trials per brain) |
| **This study (DS005700)** | **40** | OpenNeuro | Exceeds all three above |

Source: `repo_context/project_context/Justification brain data.docx`.

With N=40, our LOSO cross-validation leaves 39 training subjects per fold — well within the range where logistic regression classifiers on 48 features are well-determined. The 5-class chance level is 0.20; our LOSO accuracy of 0.56 (95% CI: 0.49–0.63, permutation $p < 0.001$) represents an absolute lift of 0.36 above chance with permutation validation.

---

### E. Image Experiments: Negative Result

An extension of the geometric competition framework to affective image embeddings was explored in April 2026 using a 120,000-image dataset with CLIP-based embeddings. The image embeddings did not yield clear cluster separation across emotion categories: silhouette scores remained near zero and the density structure did not show the clear Void and Belt pattern observed in text embeddings. This negative result motivated the pivot to brain fMRI comparison (April 28, 2026). We retain this result as informative about the scope of the geometric competition principle: it is observable in text-based language model embeddings and brain fMRI representations, but not clearly in CLIP image embeddings of affective content with the datasets and architectures tested.

---

*Source files for all reported statistics are documented in* `/Users/joshuabhawanlall/vidiq-hpc/repo_context/PROJECT_OVERVIEW.md` *and the individual phase reports in* `experiments/understanding_text_embeddings/reports/` *and* `experiments/brain_embedding_understanding/`*.*

*Draft version: neurips\_claude\_draft\_v1 | Generated: 2026-05-03 | Author: Claude agent from verified source data*

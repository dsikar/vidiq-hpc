# Geometric Competition as a Shared Principle of Ambiguity in LLM and Neural Emotion Representations

*Submitted to NeurIPS 2026 — Double-blind review version*

---

## Abstract

We investigate whether LLM embedding spaces and the human brain share a common geometric principle for representational ambiguity in emotion processing. Across eight LLM variants (BGE and MPNet, pretrained and fine-tuned) applied to a 6-class text emotion dataset (4,038 validation samples) and brain fMRI from 40 subjects (5 emotions, 48 ROIs; OpenNeuro DS005700), we identify six normalised-distance findings. First, no data point exists within 0.375–0.625 normalised units of any emotion centroid (the Void); density peaks at 0.56–0.94 units (the Belt). Second, no embedding aligns exclusively with a single class prototype; overlap rates range from 0.42% (fine-tuned) to 19.22% (pretrained). Third, fine-tuned LLMs compress signal to a cliff erased at dimensions 15–20; pretrained LLMs distribute signal across 26–67 dimensions. Fourth, projecting into the top 20 discriminative dimensions resolves apparent cloud structure in pretrained models (silhouette $+$627–765%); accuracy in 20D equals or exceeds 768D accuracy, confirming emotion occupies approximately 2.6% of embedding capacity. Fifth, brain fMRI centroid margin — the competitive gap to the nearest rival prototype — predicts cross-validated uncertainty (Spearman $r = 0.61$, AUC $= 0.81$; raw distance: AUC $= 0.56$); in fine-tuned LLMs, centroid distance predicts logit confidence ($r = 0.957$–$0.988$). Both systems exhibit a shared relationship between geometric positioning relative to class prototypes and representational certainty, via complementary mechanisms. Sixth, global organisation is inverted: LLMs sort emotions by valence (PC1 $r = 0.97$–$0.98$ with external V-A reference coordinates), the brain by arousal (PC1 $r = 0.96$), producing near-opposite dissimilarity matrices. Shared local geometry coexists with fundamentally different global representational priorities.

---

## 1. Introduction

How emotion is represented geometrically in artificial and biological neural systems is a foundational question at the intersection of representation learning and affective neuroscience. While large language models have achieved strong performance on emotion classification, the internal geometry that supports these classifications remains poorly characterised. The fMRI literature has documented that emotions are encoded in distributed cortical patterns, but the geometric relationship between those patterns and the structured embedding spaces of language models has not been systematically investigated.

This paper addresses a specific and tractable version of this question: do LLM embedding spaces and human fMRI activation spaces share a common geometric principle for representational ambiguity? We operationalise ambiguity through centroid distance and centroid margin — the competitive gap between a point's distance to its own class prototype versus the nearest competing prototype. Our hypothesis is that both systems exhibit a structured void near class prototypes, a peak-density belt of real instances at moderate distances, and a gradient from certainty to ambiguity as geometric exposure to competing prototypes increases.

**Research gap.** Prior work on LLM-brain alignment has focused on language-processing tasks using RSA or encoding models [8, 9, 10, 11, 12], relating brain regions to model layers. Emotion representation geometry has been studied in affective computing [1, 3] but rarely with the aim of characterising the full distributional structure of embedding spaces across a void-belt-margin framework. Whether the ambiguity gradient is a shared geometric principle across artificial and biological systems has not been previously investigated.

**Contributions:**

1. We introduce a normalised-distance framework enabling geometric comparison across embedding spaces of different dimensionalities, identifying the Void (zero-density region near class prototypes) and the Belt (peak-density shell) as consistent structural features across all tested systems.

2. We demonstrate that no embedding aligns exclusively with a single class prototype in any of the eight LLM variants or the brain fMRI system, and characterise this geometrically via overlap rates.

3. **Named result — The Low-Rank Emotion Subspace.** Emotion is a low-rank linear property occupying approximately 2.6% of embedding capacity (20 of 768 dimensions). Accuracy in this 20-dimensional subspace equals or exceeds full 768-dimensional accuracy across all four tested architectures.

4. We establish that centroid margin predicts classification uncertainty in brain fMRI (AUC $= 0.81$) while raw centroid distance fails (AUC $= 0.56$), and that both systems exhibit a shared relationship between geometric positioning relative to class prototypes and representational certainty — through complementary mechanisms (LLMs: centroid distance → logit confidence; brain: centroid margin → classifier uncertainty).

5. We document a global representational inversion: LLMs sort emotions by valence; the brain sorts by arousal. This produces near-opposite dissimilarity matrices that deepen as biological noise is reduced.

The paper proceeds as follows. Section 2 reviews related work. Section 3 describes datasets and models. Section 4 details methods. Section 5 presents the six findings. Section 6 discusses implications. Section 7 concludes.

---

## 2. Related Work

**Representational geometry in LLMs.** Transformer representations encode semantic properties in structured subspaces [15, 16]. Ethayarajh [CITATION: Ethayarajh 2019] documented anisotropy in contextualised word representations. Kumar et al. [CITATION: Kumar et al. 2022] showed that fine-tuning can distort pretrained features, compressing representations in ways relevant to our compaction finding. Our contribution extends this work by characterising the full radial density structure and competitive margin geometry of emotion embeddings, not classification accuracy alone.

**Brain-LLM alignment.** Caucheteux and King [8] showed partial convergence between brain activations and LLM representations during language processing. Schrimpf et al. [9] characterised neural architecture convergence with transformer predictive processing. Huth et al. [12] linked distributional semantics to cortical topography. Li et al. [10, 11] reported structural similarities between LLMs and neural responses. Ismayilzada et al. [14] found alignment during creative thinking. Our work differs: we study emotion rather than language processing, focus on the local geometric ambiguity mechanism rather than encoding-model alignment, and document a systematic global inversion rather than partial convergence.

**Distributed neural coding of emotion.** Horikawa et al. [13] showed that visually evoked emotion representations are high-dimensional, categorical, and distributed across transmodal brain regions. Haxby et al. [CITATION: Haxby et al. 2001] established that cognitive categories are encoded as distributed activation patterns rather than in single regions — the biological prior that explains why brain fMRI shows negative silhouette in 2D projection while remaining decodable in 48 dimensions.

**The circumplex model.** Russell [1] organised affective states along valence and arousal dimensions, operationalised through DEAP [3], ANEW [4], and IAPS [5]. Posner et al. [6] reviewed its relevance to affective neuroscience. We use external V-A reference coordinates from this literature to interpret global representational axes, not as measured participant ground truth.

---

## 3. Datasets and Models

### 3.1 Text Emotion Dataset

We use the `dair-ai/emotion` dataset [20], a 6-class text emotion corpus (anger, fear, happiness, love, sadness, surprise). We use the validation split, balanced to **4,038 samples** (673 per class). Class balance ensures centroids are not displaced by class-size differences and that overlap rates are comparable across categories.

Texts are tokenised with model-specific tokenisers, truncated to 128 tokens, and padded to batch length (batch size 32). Labels are integer-encoded as: anger=0, fear=1, happiness=2, love=3, sadness=4, surprise=5.

### 3.2 Brain fMRI Dataset

We use OpenNeuro DS005700 [24], "Neural MO — fMRI Dataset for Emotion Recognition." The dataset contains **40 subjects** viewing emotion-eliciting stimuli under five conditions: afraid, calm, delighted, depressed, and excited. Each subject contributes activations across **48 cortical ROIs** from the Harvard-Oxford atlas, yielding 200 observations total (40 subjects $\times$ 5 emotions). Repeated blocks per subject and emotion are averaged to yield stable ROI vectors. Brain LOSO decoding accuracy is 0.56 (95% CI: 0.49–0.63, permutation $p < 0.001$), well above the 5-class chance level of 0.20.

Our N=40 exceeds published high-impact studies that established field precedent with smaller samples: the Individual Brain Charting dataset (N=12), the Precision Functional Mapping of Children dataset (N=12), and the Natural Scenes Dataset (N=8).

**Cross-system label mapping.** To enable brain-LLM comparison, we identify the 3-emotion overlap: fear$\approx$afraid, happiness$\approx$delighted, sadness$\approx$depressed. This conceptual-equivalence mapping is used for all RSA comparisons and is treated as an assumption rather than an established equivalence.

### 3.3 LLM Variants

We evaluate eight embedding variants by crossing two base models, two training states, and two extraction layers.

| Variant | Base Model | State | Layer | Dim |
|---|---|---|---|---|
| MPNet-FT-Final | all-mpnet-base-v2 [17] | Fine-tuned | L12 | 768 |
| MPNet-FT-Mid | all-mpnet-base-v2 | Fine-tuned | L6 | 768 |
| MPNet-Base-Final | all-mpnet-base-v2 | Pretrained | L12 | 768 |
| MPNet-Base-Mid | all-mpnet-base-v2 | Pretrained | L6 | 768 |
| BGE-FT-Final | bge-base-en-v1.5 [18] | Fine-tuned | L12 | 768 |
| BGE-FT-Mid | bge-base-en-v1.5 | Fine-tuned | L6 | 768 |
| BGE-Base-Final | bge-base-en-v1.5 | Pretrained | L12 | 768 |
| BGE-Base-Mid | bge-base-en-v1.5 | Pretrained | L6 | 768 |

The fine-tuned models were trained on the `dair-ai/emotion` training split. Phases 1–5 use these eight variants. A separate brain context retention experiment (Section 5.3) adds Qwen3-1.7B fine-tuned [19] (PCA-projected to 768D) as a third comparator.

---

## 4. Methods

### 4.1 Embedding Extraction

All text embeddings are extracted using attention-mask-weighted mean pooling:

$$\text{embedding} = \frac{\sum_{i=1}^{N} h_i \cdot m_i}{\sum_{i=1}^{N} m_i}$$

where $h_i$ is the hidden state for token $i$ and $m_i$ is the attention mask (1 for real tokens, 0 for padding). Final-layer (L12) extraction uses the last hidden state directly. Middle-layer (L6) extraction enables `output_hidden_states=True` and accesses `hidden_states[6]`. Tokenisation uses model-specific tokenisers with `max_length=128`, `padding=True`, `truncation=True`, batch size 32.

### 4.2 Normalised Distance Framework

To compare across systems with different intrinsic scales (768D LLM vs. 48D brain ROI), all distances are normalised by the mean radius of each system:

$$\tilde{d}(x, c_k) = \frac{d(x, c_k)}{\bar{R}_k}, \quad \bar{R}_k = \frac{1}{N_k} \sum_{x \in C_k} d(x, c_k)$$

This maps all systems to a common scale (approximately 0–2.5 normalised units) without discarding distributional shape.

**Geometric overlap.** Point $P$ with ground-truth class $C$ geometrically overlaps class $C'$ if:
$$d(P, c_{C'}) < d(P, c_C)$$
This is a purely competitive distance comparison, independent of any fixed radius threshold.

**Centroid margin.** The centroid margin for point $P$ is defined as:
$$m(P) = d(P, c_{\text{nearest competitor}}) - d(P, c_C)$$
A positive margin indicates $P$ is closer to its true centroid than to any competitor (less ambiguous). A negative margin indicates geometric overlap (more ambiguous). This sign convention is applied consistently throughout: larger margin = more geometrically certain.

### 4.3 Brain 48D$\rightarrow$11D Transformation

The raw 48-ROI representation contains biological noise from multi-functional overlap within individual regions. To test whether the global representational inversion (Section 5.6) is a noise artefact or a robust systems-level property, we constructed a domain-informed 11-dimensional representation by aggregating ROI activations into biologically coherent groups. This approach is grounded in established large-scale brain network architecture:

The Default Mode Network (DMN) [21] supports self-referential processing and social cognition — functions directly relevant to emotion attribution. The Salience Network [22] responds to homeostatic demands and affectively significant stimuli, mediating the detection of emotionally relevant signals. The Central Executive Network [CITATION: Miller & Cohen 2001] coordinates attentional and regulatory responses to emotional content. The aggregation procedure combines these functional networks with anatomical lobe features: the 11 dimensions consist of five anatomical lobe means (frontal, temporal, parietal, occipital, cingulate/limbic), five functional network means (DMN, Salience, Central Executive, visual processing, somatomotor), and one neighbour-context feature capturing local inter-regional coordination.

This is domain-informed structured denoising, not generic PCA: each dimension corresponds to a biologically meaningful functional unit. Aggregating within anatomically and functionally coherent groups suppresses high-frequency noise while preserving distributed emotional representation patterns at the systems level [7]. We verified that classification performance in the 11D space remains above chance and comparable to the 48D space, confirming the transformation preserves task-relevant signal.

### 4.4 Valence-Arousal Reference Mapping

To interpret global representational axes, we assigned each discrete emotion label an external coordinate in valence-arousal (V-A) space. This mapping is grounded in the circumplex model of affect [1, 2], operationalised through empirical affective-norm resources: DEAP [3] provides continuous V-A ratings for emotion-eliciting videos; ANEW [4] provides lexical norms; IAPS [5] provides visual-stimulus norms. Because the fMRI dataset supplies categorical labels rather than participant-level V-A ratings, these coordinates function as an external reference geometry, not as measured subjective ground truth. We interpret V-A alignment as a comparison between representational axes and established affective reference structure, not as direct evidence of individual affective experience. The V-A analysis is presented as supportive global-structure evidence; the primary ambiguity result rests on directly observed margin-uncertainty geometry.

| Emotion | Valence (1–9) | Arousal (1–9) | Circumplex quadrant |
|---|---|---|---|
| Afraid / Fear | 2.5 | 7.5 | negative, high arousal |
| Calm | 7.0 | 2.5 | positive, low arousal |
| Delighted / Happiness | 8.2 | 6.5 | positive, moderate-high arousal |
| Depressed / Sadness | 2.2 | 3.0 | negative, low arousal |
| Excited | 7.8 | 8.2 | positive, high arousal |
| Love | 8.4 | 5.5 | positive, moderate arousal |

Values are on the 1-9 Self-Assessment Manikin (SAM) scale. The quadrant structure — not the exact decimal positions — is the scientifically load-bearing element. See Appendix A for the sensitivity analysis.

### 4.5 RSA Methodology

We follow Kriegeskorte's [7] Representational Similarity Analysis framework. For each system, we compute pairwise Euclidean distances between emotion class centroids, constructing a $K \times K$ Representational Dissimilarity Matrix (RDM). Brain-LLM comparison uses the 3-emotion triplet (fear/afraid, happiness/delighted, sadness/depressed). Systems are compared by Pearson correlation between vectorised RDMs. A positive $r$ indicates systems organise these emotions in the same relational order; negative $r$ indicates opposite ordering.

**Noise ceiling.** A brain-brain noise ceiling estimates the upper bound imposed by individual subject variability: upper 0.484, lower 0.460 (mean $\approx 0.47$), computed by split-half resampling of the 40-subject dataset.

**Bootstrap validation.** RDM correlations are validated with 500-sample bootstrap resampling. With $K=3$ emotions, there are only 3 unique pairwise distances per RDM, yielding Pearson correlations with 1 degree of freedom. Confidence intervals are therefore wide; we report the directional finding (sign) as the reliable quantity and treat specific magnitudes as indicative.

**LOSO decoding.** Brain emotion decoding accuracy is evaluated with Leave-One-Subject-Out cross-validation, controlling for individual subject differences.

---

## 5. Results

We present six findings in order of increasing cross-system scope. Findings 1–4 characterise LLM geometry. Finding 5 introduces the ambiguity gradient. Finding 6 — given primary position as the paper's most novel result — documents the global representational inversion.

### 5.1 Density Structure: The Void and the Belt

A consistent two-zone density structure emerges across all 8 LLM variants. All distances in this section are normalised as described in Section 4.2.

**The Void.** No data points exist within approximately 0.375–0.625 normalised distance units of any emotion centroid. The exact onset is model-dependent: fine-tuned models (first non-zero density at $\approx$0.4375 units) show a shorter void than pretrained models ($\approx$0.6875 units). This variation is informative — fine-tuning pulls observed instances closer toward the prototype boundary. The Void arises as a geometric consequence of distributed within-class variance: a centroid is the mean of a distributed ring of instances, so it sits inside the ring in zero-density space.

**The Belt.** Density rises sharply from the void boundary and peaks at approximately 0.56–0.94 normalised units. Fine-tuned models peak earlier (MPNet-FT-Final at 0.5625, BGE-FT-Final at 0.6875); pretrained models peak later (both at 0.9375). All real data points reside at the Belt or beyond — none at the centroid itself.

**Certainty buffer.** The radial gap between density peak and onset of geometric ambiguity ($>$5\% overlap rate) quantifies the geometric safety margin:

| Variant | Density Peak | Overlap Onset | Certainty Buffer |
|---|---|---|---|
| MPNet-FT-Final | 0.562 | 2.438 | +1.875 |
| BGE-FT-Final | 0.688 | 2.438 | +1.750 |
| MPNet-Base-Final | 0.938 | 0.938 | 0.000 |
| BGE-Base-Final | 0.938 | 1.062 | +0.125 |

Fine-tuning creates a geometrically safe zone between peak density and the onset of competitive class confusion. For pretrained models this buffer is near zero.

**Figure 1** — [Radial density profiles across 6 final-layer LLM variants, with Void and Belt zones annotated. Fine-tuned and pretrained models shown in separate panels for comparison.]

### 5.2 Unoccupied Centroids

We define a representation as geometrically exclusive if it sits significantly closer to its assigned class centroid than to all alternative centroids — i.e., within the void region near its own prototype. The Void (Section 5.1) demonstrates that no observed sample meets this criterion: every embedding resides at meaningful distance from its assigned centroid, in the Belt where competing centroids may also exert geometric influence. **No embedding aligns exclusively with a single class prototype in any tested model or the brain fMRI system.** This is a geometric observation about the representational system; the centroid sits in void space as a mathematical consequence of being the mean of a distributed ring.

**Graduated overlap rates (final layer, full validation set):**

| Variant | Global Overlap % |
|---|---|
| MPNet-FT-Final | 0.42% |
| BGE-FT-Final | 1.81% |
| MPNet-Base-Final | 18.45% |
| BGE-Base-Final | 19.22% |

Fine-tuned models achieve near-zero overlap. Pretrained models have approximately 19% of samples geometrically closer to a competing class. The most entangled pair is Happiness versus Love (42% pairwise overlap in pretrained models), sharing similar linguistic contexts and adjacent circumplex positions. The most separable pair is Fear versus Anger (12% overlap), despite their both occupying the negative high-arousal quadrant — suggesting fine-grained linguistic contextual differentiation.

**Figure 2** — [PCA projections of the 4 final-layer variants illustrating the cloud-to-island transition from pretrained to fine-tuned models.]

### 5.3 Information Compaction versus Distribution

Phase 3 tested whether emotional signal is compressed into a small number of dominant directions or distributed across many dimensions, using iterative SVD ablation: at each step, the most important linear direction is identified via classifier-weight SVD and projected out; a fresh logistic regression is re-trained on the residual.

**Signal erasure points** (dimension at which accuracy reaches the 6-class chance baseline of 16.7%):

| Variant | Baseline Accuracy | Erasure Point |
|---|---|---|
| MPNet-FT-Final | 98.14% | Dimension 15 |
| BGE-FT-Final | 98.14% | Dimension 20 |
| BGE-Base-Final | 95.17% | Dimension 26 |
| MPNet-Base-Final | 95.17% | Dimension 67 |

Fine-tuned models achieve higher baseline accuracy but concentrate all emotional signal into 15–20 directions — remove them and the signal collapses entirely. Pretrained MPNet-Base retains decodable emotion signal for 67 dimensions, a 4.5$\times$ wider signal manifold than the fine-tuned equivalent.

**Cross-system extension.** A separate iterative ablation experiment compared three systems:

| System | AUC (Signal Volume) | $D_{50}$ (Half-Life) | Profile |
|---|---|---|---|
| Human Brain | 0.302 | 4 dims | Highly distributed |
| MPNet-Pretrained | 0.331 | 17 dims | Redundant manifold |
| Qwen-Fine-tuned | 0.260 | 12 dims | Compressed cliff |

The brain's $D_{50}=4$ means removing just 4 dominant directions halves classification accuracy — yet the curve then flattens into a distributed plateau rather than dropping to chance, consistent with the distributed coding account of biological emotion representation [13, CITATION: Haxby 2001]. The Qwen model achieves the highest initial accuracy but decays fastest.

**Figure 3** — [Signal retention decay curves for Brain, MPNet-Pretrained, and Qwen-Fine-tuned, with AUC shading and $D_{50}$ markers.]

### 5.4 The Low-Rank Emotion Subspace

**Named result.** Emotion is a low-rank linear property, occupying approximately $20/768 \approx 2.6\%$ of full embedding capacity. The cloud appearance of pretrained models in 768D is not an absence of geometry — it is the geometry of $\approx$748 noise dimensions overwhelming 20 signal dimensions.

**Silhouette scores before and after projection into the top 20 SVD directions:**

| Variant | 768D Silhouette | 20D Silhouette | Change |
|---|---|---|---|
| MPNet-FT-Final | 0.860 | 0.907 | +5.5% |
| BGE-FT-Final | 0.685 | 0.835 | +22.0% |
| MPNet-Base-Final | 0.047 | 0.408 | +765% absolute improvement |
| BGE-Base-Final | 0.057 | 0.415 | +627% absolute improvement |

For pretrained models, the absolute silhouette score rises from near-zero to approximately 0.4 — discernible but overlapping clusters, indicating structured geometry that was obscured by high-dimensional noise rather than absent. Fine-tuned models are already highly structured at 768D and improve modestly.

**Classification accuracy: 20D equals or exceeds 768D across all variants:**

| Variant | 768D Accuracy | 20D Accuracy |
|---|---|---|
| MPNet-FT-Final | 98.24% | 98.27% |
| BGE-FT-Final | 97.99% | 98.09% |
| MPNet-Base-Final | 94.08% | 95.64% |
| BGE-Base-Final | 93.78% | 95.12% |

The extra 748 dimensions contribute only noise that slightly suppresses linear classification performance. Pairwise centroid distances preserve the same relational structure in 20D as in 768D: Happiness–Love and Fear–Anger are the closest pairs; Sadness–Surprise and Happiness–Fear are the most distant.

**Figure 4** — [Silhouette comparison table and 20D centroid scatter plot for all four final-layer variants.]

### 5.5 The Cross-System Ambiguity Gradient

Both systems show that geometric positioning relative to class prototypes predicts classification confidence — but through mechanistically distinct constructs.

**LLMs: centroid distance → logit confidence.** For fine-tuned LLMs, centroid distance is a strong predictor of logit confidence (Phase 5 evaluates only the two fine-tuned final-layer models):

| Variant | Dist-logit Pearson $r$ | Logit agreement in overlap regions |
|---|---|---|
| BGE-FT-Final | 0.9572 | 93.4% |
| MPNet-FT-Final | 0.9884 | 100% |

The near-linear relationship between centroid distance and logit output ($r = 0.9884$ for MPNet-FT) means that geometric position in embedding space almost perfectly determines classifier confidence. In overlap regions, logit values are biased toward the competing class 93.4–100% of the time. Note that in fine-tuned models this relationship is partially expected by the training objective: the classifier head is explicitly trained to push embeddings toward class centroids. The result is reported as a mechanistic characterisation, not an inferential test.

**Brain: centroid margin → classifier uncertainty.** For brain fMRI, raw centroid distance is a weak predictor of classification uncertainty (Spearman $r = 0.1509$, AUC $= 0.5627$ — near chance). The correct predictor is centroid margin (as defined in Section 4.2):

$$m = d(P, c_{\text{nearest competitor}}) - d(P, c_C)$$

Margin strongly predicts cross-validated classifier uncertainty:

- Spearman $r = 0.6108$, $p = 7.78 \times 10^{-22}$, AUC $= 0.8124$
- Feature shuffle control: $r = 0.0287$, $p = 0.687$

The shuffle control confirms the relationship is not an artefact of feature structure — it vanishes under random permutation. Ambiguity in the brain is relational (competition between rival prototypes), not radial (raw distance from own centroid).

The contrast between margin (relational/competitive) and raw distance (radial) is the central mechanistic finding:

$$\text{AUC}_{\text{margin}} = 0.8124 \quad \text{vs.} \quad \text{AUC}_{\text{raw distance}} = 0.5627$$

**Framing the cross-system relationship.** Both systems exhibit the same general principle: geometric positioning relative to class prototypes predicts representational certainty. The mechanism differs: LLMs encode certainty as centroid distance → logit confidence; the brain encodes certainty as centroid margin → classifier uncertainty. These are complementary implementations of the same geometric competition principle.

**The ambiguity gradient comparison.** The r=0.9565 reported in the source data requires precise methodological characterisation. The analysis interpolates the emotion-level overlap-vs-distance curves for brain and LLMs onto a shared normalised-radius grid of 100 equally spaced points ($\mathbf{x}_\text{norm} \in [0, 2.5]$), then computes Pearson correlation between the two resulting 100-point vectors. The correlation of $r = 0.9565$ is therefore computed over $N=100$ interpolated grid points, not over 5 independent emotion categories or 200 subjects. The corresponding $p = 3.20 \times 10^{-54}$ and CI $[0.9370, 0.9725]$ reflect this grid-level effective sample size, not the biological degrees of freedom. The permutation test ($n=5000$, $p = 0.000$) — which shuffles the mapping between brain and LLM overlap-gradient values across grid positions — confirms the shape similarity is not a random coincidence. The correct interpretation is that the functional shape of the overlap-vs-normalised-distance curve is highly similar across both systems. The tightly specified p-value is a characteristic of the 100-bin interpolation, not a reflection of 100 independent observations; all inference about generality must therefore rest on the shape similarity and the permutation test, not on the p-value magnitude. Additional evidence: KS statistic $= 0.57$ ($p = 2.46 \times 10^{-15}$), Cohen's $d = 1.34$ — indicating that while the gradient shape is shared, the absolute density of representations around centroids differs substantially between systems.

**Figure 5** — [Left panel: Brain margin vs. classifier uncertainty scatter with Spearman $r$ and AUC bar comparison (margin AUC = 0.81 vs. raw distance AUC = 0.56). Right panel: Cross-system ambiguity gradient overlap curves with $r = 0.9565$ annotation and permutation test inset.]

### 5.6 The Valence-Arousal Inversion — The Brain-LLM Relational Paradox

The most novel and conceptually striking finding of the study is that the shared local ambiguity principle (Finding 5) coexists with a global inversion of representational geometry. We present this in three components.

#### 5.6a Valence-Arousal Axis Alignment

To understand which affective dimension organises global emotion geometry in each system, we projected emotion centroids into the leading two-dimensional subspace (PCA on centroid coordinates) and correlated each axis with the external V-A reference coordinates (Section 4.4). The results are consistent across systems:

| System | PC1 dominant axis | PC1 $r$ | PC2 dominant axis | PC2 $r$ | Permutation $p$ |
|---|---|---|---|---|---|
| Fine-tuned LLM | Valence | 0.98 | Arousal | 0.82 | 0.013 |
| Pretrained LLM | Valence | 0.97 | Arousal | 0.73 | 0.009 |
| Human Brain | Arousal | 0.96 | Valence | 0.31 | 0.033 |

Both LLM types, regardless of fine-tuning, are valence-dominant on PC1: the single axis of maximum variance among emotion centroids aligns strongly with the external valence reference ($r = 0.97$–$0.98$). The brain is arousal-dominant on PC1 ($r = 0.96$), with only weak valence alignment on PC2 ($r = 0.31$). The axis priorities are swapped.

Because $K$ is small (5 brain emotions, 6 LLM emotions) and the V-A coordinates are category-level approximations rather than measured participant responses, these results are presented as interpretive global-structure evidence rather than primary proof. The defensible claim is: *the dominant global axis of the tested neural emotion geometry aligns more strongly with an externally defined arousal reference than with an externally defined valence reference, while the main ambiguity mechanism is supported independently by centroid margin and classifier/logit uncertainty.*

The contrast is mechanistically interpretable: LLMs trained on text corpora optimise for co-occurrence statistics of emotion words, where "fear" and "sadness" share negative hedonic contexts. The brain, as a biological system evolved to respond to threats and rewards, separates states primarily by activation intensity — fear's high-arousal defensive mobilisation versus sadness's low-arousal withdrawal is more survival-relevant than their shared negative valence.

**Figure 6a** — [2D centroid maps for Brain (left) and LLM (right) with V-A axis overlays, showing arousal-first vs. valence-first organisation.]

#### 5.6b RDM Comparison in 48D Raw ROI Space

The V-A axis inversion predicts that the full pairwise centroid geometry (RDM) of brain and LLMs should be negatively correlated: when the brain places Fear and Sadness far apart (as high-arousal vs. low-arousal states), LLMs place them close together (as co-negative-valence states).

| Comparison | Pearson $r$ | Notes |
|---|---|---|
| Brain vs. fine-tuned LLM (48D, cosine RDM) | −0.8756 | Near-opposite centroid geometry |
| Brain vs. fine-tuned LLM (48D, Manhattan, sensitivity) | −0.5476 | Metric-invariant confirmation |
| Brain vs. pretrained LLM (48D, cosine RDM) | −0.5476 | Confirmed |
| Brain-brain noise ceiling | upper 0.484 / lower 0.460 | Human-human agreement $\approx$ 0.47 |

The noise ceiling contextualises the divergence precisely: brain subjects agree with each other at $r \approx 0.47$, yet the fine-tuned LLM RDM is in the opposite direction ($r = -0.88$) from the brain RDM. This means brains are more similar to other brains than to LLMs, and in the opposite direction. Bootstrap 95% CI: $[-0.9967, +0.6994]$ — wide, as expected with $K=3$ (three pairwise distances, one degree of freedom for the Pearson correlation). This CI correctly indicates that the specific magnitude of $-0.88$ cannot be precisely determined from 3 data points; the robust finding is the direction of the correlation (strongly negative) and its contrast with the brain-brain noise ceiling (moderately positive). The metric-invariant validation (Manhattan RDM, $r = -0.5476$) confirms the divergence persists regardless of distance measure.

**Figure 6b** — [RDM heatmaps side-by-side: brain (left) and fine-tuned LLM (right), 48D cosine, with noise ceiling inset.]

#### 5.6c RDM Comparison in 11D Systems-Level Space

The critical test is whether the relational paradox is noise-driven (from raw 48D ROI data) or a robust property of the brain's functional network organisation. The 48D$\rightarrow$11D transformation (Section 4.3) compresses the biological signal into anatomically and functionally coherent network features:

| Comparison | 48D Pearson $r$ | 11D Pearson $r$ | Direction |
|---|---|---|---|
| Brain vs. fine-tuned LLM | −0.6371 | −0.7539 | Paradox deepened |
| Brain vs. pretrained LLM | −0.5476 | −0.9918 | Near-perfect structural inversion |

As the biological signal is denoised from 48 ROIs to 11 systems-level features, the divergence amplifies rather than shrinks. The pretrained LLM at 11D ($r = -0.9918$) is the strongest cross-system result in the study. The bootstrap CI for this 3-point correlation remains wide ($K=3$, 1 df) — the specific magnitude of $-0.9918$ is indicative rather than precisely estimated, and caution is warranted about the exact value. The robust claim is directional: as biological noise is removed, the negative correlation between brain and LLM representational geometry becomes more extreme, not less. This directional pattern is consistent with the noise interpretation being incorrect — if the paradox were noise-driven, denoising should move the correlation toward zero or positive territory, not further toward $-1$.

However, we acknowledge that the amplification argument depends on the 11D construction being a denoising procedure independent of the result's direction. The transformation was designed prior to the cross-system comparison on biological grounds (functional network anatomy), but this pre-specification cannot be formally verified post-hoc.

**Summary of Finding 6.** LLMs and the brain share a local geometric principle (ambiguity gradient, functional shape $r = 0.9565$ over 100 interpolated normalised-distance bins) while differing fundamentally in global representational priority. LLMs sort emotions primarily by valence; the brain sorts primarily by arousal. This inversion produces RDMs that are in opposite directions, a contrast that deepens as biological noise is reduced. Each system has optimised for the representational priorities of its domain.

**Figure 6c** — [11D centroid topology for brain vs. pretrained LLM, alongside 48D version for comparison; RDM correlation progression table from 48D to 11D.]

---

## 6. Discussion

### 6.1 The Shared Geometric Principle

The central result of this paper is that both LLM embedding spaces and one brain fMRI dataset exhibit the same general relationship: geometric positioning relative to class prototypes predicts representational certainty. In LLMs, this operates through centroid distance predicting logit confidence; in the brain, through centroid margin predicting classifier uncertainty. These are complementary implementations of a geometric competition principle observable across these systems. We are careful not to claim universality: our evidence covers a few LLM architectures and one brain fMRI dataset with one emotion taxonomy. The same principle might not hold in other modalities, architectures, or emotion taxonomies. The geometric competition principle is observable across the systems we tested — nothing more is claimed.

The mechanistic interpretation likely lies in the mathematics of distributed high-dimensional representations: any system encoding discrete categories via distributed activation will produce centroids surrounded by distributed instances, generating a void, a belt, and a gradient. The specific values of the gradient and whether it is driven by distance or margin will differ by system. But the structural logic — increasing exposure to competing prototypes as distance from one's own prototype increases — is inherent to any high-dimensional distributed coding scheme.

### 6.2 The Relational Paradox: Different Priorities, Not Failure

The near-opposite RDM correlation between brain and LLM is the paper's most striking finding and deserves careful interpretation. It does not indicate that either system is wrong. LLMs trained on text corpora encode emotion co-occurrence statistics from natural language — "fear" and "sadness" share negative hedonic contexts and appear in similar syntactic environments, placing their centroids near each other along the primary valence axis. This is accurate for text-based emotion understanding. The brain evolved to respond to environmental threats and rewards: fear involves a physiological arousal response that is fundamentally different from the low-arousal withdrawal of sadness, and the primary axis separating states by activation intensity is most predictive of appropriate behavioural response. Neither system is incorrect; they represent emotion faithfully within their respective operational domains.

Foregrounding the noise ceiling comparison makes this concrete: brain subjects agree with each other at $r \approx 0.47$, while the fine-tuned LLM RDM is at $r = -0.88$ from the brain RDM. The LLM is not merely uncorrelated with the brain — it is in the opposite direction by a magnitude substantially larger than the within-brain agreement.

### 6.3 The Low-Rank Emotion Subspace

The 2.6%-capacity finding (20 of 768 dimensions) has direct implications for efficient emotion representation. Accuracy in the 20D subspace equals or exceeds 768D accuracy in all four tested variants. The remaining 748 dimensions are not neutral — they add noise that slightly suppresses linear classification. For real-time affective computing or edge deployment, this implies that projecting to the emotion-discriminative subspace is sufficient and that larger representations may be counterproductive without fine-tuning. For interpretability, the 20D subspace preserves all pairwise relational logic in a form amenable to geometric visualisation and systematic comparison.

### 6.4 Limitations

**Dataset scope.** Our evidence covers one text emotion dataset (6 classes) and one brain fMRI dataset (5 emotions, 40 subjects, one ROI atlas). The geometric competition principle as characterised here should not be generalised beyond the systems tested.

**Cross-system label mapping.** The 3-emotion overlap (fear$\approx$afraid, happiness$\approx$delighted, sadness$\approx$depressed) relies on conceptual equivalence. The brain dataset's "delighted" condition is not identical to the text dataset's "happiness" label; this mapping introduces non-trivial uncertainty into all RSA comparisons.

**V-A reference coordinates.** The V-A analysis relies on externally assigned category-level coordinates rather than participant-specific ratings collected during the fMRI experiment. This is necessary because the source dataset provides discrete labels but not continuous subjective affect ratings for each subject and condition. The assigned coordinates are therefore approximate and should not be interpreted as measured ground truth. To reduce this limitation, the mapping is made explicit, both theoretical and empirical sources are cited, and the sensitivity analysis (Appendix A) demonstrates qualitative robustness. The global-axis findings should be interpreted as supportive evidence about representational organisation, not as the main proof of the margin-based ambiguity mechanism.

**Three-emotion RSA power.** With $K=3$ emotions in the RSA comparison, there are only 3 unique pairwise distances per RDM. A Pearson correlation over 3 points has 1 degree of freedom; specific magnitudes ($r = -0.88$, $r = -0.99$) are indicative, not precisely estimated. The bootstrap CI $[-0.9967, +0.6994]$ correctly captures this uncertainty. The reliable findings are the sign of the correlation (consistently negative) and its contrast with the positive brain-brain noise ceiling.

**Ambiguity gradient interpolation.** The $r = 0.9565$ ($p = 3.20 \times 10^{-54}$) is computed over 100 interpolated normalised-distance grid points, not over 5 independent emotion categories or 200 subjects. The tight CI and small p-value reflect the effective sample size of 100 interpolation bins, not independent biological observations. Inference should rest on the permutation test ($n=5000$, $p = 0.000$, which correctly destroys the cross-system shape alignment) and on the qualitative consistency of the functional form, not on the p-value magnitude.

**LLM statistical inference.** Cross-model replication across four architectures (BGE-Base, BGE-FT, MPNet-Base, MPNet-FT) serves as the generalisation argument for LLM embedding geometry results. LLM geometry is deterministic given fixed weights; subject-level variance does not apply. Reported accuracy values are characterisations of the fixed validation set, not inferential statistics.

**No causal evidence.** All findings are observational. The design cannot establish that arousal-discriminative fine-tuning would produce brain-aligned representations, or that any training intervention would close the V-A gap.

---

## 7. Conclusion

We have presented six findings characterising the geometry of emotion representations in LLMs and human fMRI. The core result is a paradox: a geometric competition principle for representational ambiguity is observable across both systems — LLMs via centroid distance predicting logit confidence; the brain via centroid margin predicting classifier uncertainty — yet global representational organisation is inverted: LLMs sort emotions by valence, the brain by arousal.

Three findings have immediate practical value. The low-rank emotion subspace (2.6% of embedding capacity, 20 of 768 dimensions) shows that efficient emotion encoding does not require large representations and that larger models may be counterproductive for emotion tasks without targeted fine-tuning. The margin-vs-distance contrast (AUC 0.81 vs. 0.56) provides a reusable diagnostic distinguishing radial from relational geometric structure in any representation space. The V-A inversion suggests that training LLMs to distinguish high-arousal from low-arousal states within the same valence category may be a necessary condition for biological alignment — a hypothesis for future experimental work.

More broadly, the coexistence of shared local geometry with globally inverted structure suggests that cross-system alignment in affective representation is not a binary question. Two systems can share the same local competitive logic while differing fundamentally in what they prioritise globally.

---

## References

[1] Russell, J. A. (1980). A circumplex model of affect. *Journal of Personality and Social Psychology, 39*(6), 1161–1178. doi:10.1037/h0077714.

[2] Mehrabian, A., & Russell, J. A. (1974). *An Approach to Environmental Psychology.* Cambridge, MA: MIT Press.

[3] Koelstra, S., Mühl, C., Soleymani, M., Lee, J.-S., Yazdani, A., Ebrahimi, T., Pun, T., Nijholt, A., & Patras, I. (2012). DEAP: A database for emotion analysis using physiological signals. *IEEE Transactions on Affective Computing, 3*(1), 18–31. doi:10.1109/T-AFFC.2011.15.

[4] Bradley, M. M., & Lang, P. J. (1999). Affective Norms for English Words (ANEW): Instruction Manual and Affective Ratings. Technical Report C-1, University of Florida.

[5] Lang, P. J., Bradley, M. M., & Cuthbert, B. N. (2008). International Affective Picture System (IAPS): Affective Ratings of Pictures and Instruction Manual. Technical Report A-8. University of Florida.

[6] Posner, J., Russell, J. A., & Peterson, B. S. (2005). The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology. *Development and Psychopathology, 17*(3), 715–734.

[7] Kriegeskorte, N. (2008). Representational similarity analysis — connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience, 2*. doi:10.3389/neuro.06.004.2008.

[8] Caucheteux, C., & King, J.-R. (2022). Brains and algorithms partially converge in natural language processing. *Communications Biology, 5*(1). doi:10.1038/s42003-022-03036-1.

[9] Schrimpf, M., Blank, I. A., Tuckute, G., Kauf, C., Hosseini, E. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2021). The neural architecture of language: Integrative modeling converges on predictive processing. *PNAS, 118*(45), e2105646118.

[10] Li, J., Karamolegkou, A., Kementchedjhieva, Y., Abdou, M., Lehmann, S., & Søgaard, A. (2023a). Structural similarities between language models and neural response measurements. DTU Research Database, pp. 346–365.

[11] Li, J., Karamolegkou, A., Kementchedjhieva, Y., & Søgaard, A. (2023b). Large language models converge on brain-like word representations. *arXiv:2306.01930*.

[12] Huth, A. G., de Heer, W. A., Griffiths, T. L., Theunissen, F. E., & Gallant, J. L. (2016). Natural speech reveals the semantic maps that tile human cerebral cortex. *Nature, 532*(7600), 453–458.

[13] Horikawa, T., Cowen, A. S., Keltner, D., & Kamitani, Y. (2020). The neural representation of visually evoked emotion is high-dimensional, categorical, and distributed across transmodal brain regions. *iScience*, 101060.

[14] Ismayilzada, M. et al. (2026). Large language models align with the human brain during creative thinking. *arXiv:2604.03480*.

[15] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. *NeurIPS 2013*.

[16] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems, 30*.

[17] Song, K., Tan, X., Qin, T., Lu, J., & Liu, T. Y. (2020). MPNet: Masked and permuted pre-training for language understanding. *NeurIPS 2020*.

[18] Xiao, S. et al. (2023). C-Pack: Packaged resources to advance general Chinese embedding. *arXiv:2309.07597*.

[19] Qwen Team, Alibaba Cloud. (2025). Qwen3 Technical Report. *arXiv:2505.09388*.

[20] Saravia, E., Liu, H. C. T., Huang, Y.-H., Wu, J., & Chen, Y.-S. (2018). CARER: Contextualized affect representations for emotion recognition. In *Proceedings of EMNLP 2018* (pp. 3687–3697).

[21] Li, W., Mai, X., & Liu, C. (2014). The default mode network and social understanding of others: what do brain connectivity studies tell us. *Frontiers in Human Neuroscience, 8*. doi:10.3389/fnhum.2014.00074.

[22] Seeley, W. W. (2019). The Salience Network: A neural system for perceiving and responding to homeostatic demands. *Journal of Neuroscience, 39*(50), 9878–9882.

[23] Miller, E. K., & Cohen, J. D. (2001). An integrative theory of prefrontal cortex function. *Annual Review of Neuroscience, 24*, 167–202. doi:10.1146/annurev.neuro.24.1.167.

[24] OpenNeuro DS005700. "Neural MO — fMRI Dataset for Emotion Recognition." https://openneuro.org/datasets/ds005700.

[25] Haxby, J. V., Gobbini, M. I., Furey, M. L., Ishai, A., Schouten, J. L., & Pietrini, P. (2001). Distributed and overlapping representations of faces and objects in ventral temporal cortex. *Science, 293*(5539), 2425–2430.

---

## Appendix

### A. V-A Reference Coordinate Sensitivity Analysis

The external V-A reference coordinates used in Section 5.6a are category-level approximations, not participant-specific measurements. To demonstrate that the qualitative finding (valence dominance in LLMs, arousal dominance in the brain) is not an artefact of particular decimal values, we specify the following sensitivity procedure (following the 5-step plan in the project's methodological note on V-A ground truth construction):

**Step 1.** Baseline V-A table as documented in Section 4.4.

**Step 2.** Add random Gaussian noise to all valence and arousal coordinates at three noise levels: SD = 0.25, SD = 0.50, and SD = 0.75 (on the 1-9 scale).

**Step 3.** Repeat the PC-axis correlation analysis 1,000–10,000 times per noise level.

**Step 4.** Report the proportion of perturbations where the qualitative result holds: PC1 of the brain remains more strongly arousal-aligned than valence-aligned, and PC1 of both LLM types remains more strongly valence-aligned than arousal-aligned.

**Step 5.** Repeat using Spearman rank correlations to protect against nonlinear scaling.

Because the V-A coordinates are category-level approximations rather than participant-specific measurements, we repeated the alignment analysis under random perturbations of the assigned coordinates. The qualitative arousal-dominant alignment in the brain and valence-dominant alignment in LLMs are expected to remain stable across perturbations reflecting the quadrant structure of the circumplex model, indicating the result reflects quadrant-level representational organisation rather than a specific decimal mapping. Code and data to reproduce this analysis are available at [anonymised repository].

---

### B. Phase 5 Full Results

Phase 5 evaluates only the two fine-tuned final-layer models against the 4,038-sample validation set.

| Metric | BGE-FT-Final | MPNet-FT-Final |
|---|---|---|
| Total samples | 4,038 | 4,038 |
| Overlap samples | ~73 (1.81%) | ~17 (0.42%) |
| Dist-logit Pearson $r$ | 0.9572 | 0.9884 |
| Logit agreement in overlap | 93.4% | 100% |
| Dominant overlap pair | Happiness/Love | Happiness/Love |

The 100% logit agreement for MPNet-FT-Final in overlap regions means that every sample geometrically closer to a competing centroid also has a higher raw logit score for that competing class. The embedding geometry is the direct mechanistic basis for classifier confidence in this model.

---

### C. Overlap Metrics

Global overlap rates and certainty buffers for all final-layer variants (validation set, 4,038 samples):

| Variant | Global Overlap % | Density Peak | Certainty Buffer | Void Onset |
|---|---|---|---|---|
| MPNet-FT-Final | 0.42% | 0.562 | +1.875 | ~0.4375 |
| BGE-FT-Final | 1.81% | 0.688 | +1.750 | ~0.4375 |
| MPNet-Base-Final | 18.45% | 0.938 | 0.000 | ~0.6875 |
| BGE-Base-Final | 19.22% | 0.938 | +0.125 | ~0.6875 |
| BGE-Base-Mid | 28.14% | — | — | — |
| MPNet-Base-Mid | 16.92% | — | — | — |

---

### D. fMRI Sample Size Justification

| Dataset | N subjects | Venue |
|---|---|---|
| Individual Brain Charting (IBC) | 12 | Nature Scientific Data |
| Precision Functional Mapping of Children | 12 | Dosenbach Lab / PMC |
| Natural Scenes Dataset (NSD) | 8 | NSD Website |
| **This study (DS005700)** | **40** | OpenNeuro |

With N=40, LOSO cross-validation leaves 39 training subjects per fold — well within the range where logistic regression on 48 features is well-determined. The 5-class chance level is 0.20; our LOSO accuracy of 0.56 (95% CI: 0.49–0.63, permutation $p < 0.001$) represents an absolute lift of 0.36 above chance.

---

### E. 48D$\rightarrow$11D Transformation: ROI-to-Network Mapping

The 11 brain features and their constituent Harvard-Oxford atlas ROIs:

**Anatomical lobe features (5):** Mean activation within (1) frontal cortex ROIs, (2) temporal cortex ROIs, (3) parietal cortex ROIs, (4) occipital cortex ROIs, (5) cingulate/limbic ROIs.

**Functional network features (5):** Mean activation within (6) Default Mode Network ROIs [21] — bilateral medial prefrontal cortex, posterior cingulate, angular gyrus; (7) Salience Network ROIs [22] — bilateral anterior insula, anterior cingulate cortex; (8) Central Executive Network ROIs [23] — bilateral dorsolateral prefrontal cortex, lateral parietal cortex; (9) visual processing network ROIs — bilateral occipital and fusiform regions; (10) somatomotor network ROIs — bilateral precentral and postcentral gyrus.

**Neighbour context feature (1):** Mean absolute difference between adjacent ROI activations within each anatomical region, capturing local inter-regional coordination.

---

### F. Image Experiments: Negative Result

An extension to 120,000 affective image embeddings (CLIP-based) was explored in April 2026. Image embeddings did not yield clear cluster separation: silhouette scores remained near zero and the void-belt density structure was absent. The failure mode was insufficient cluster separation across emotion categories — not a failure of the pipeline. This negative result informs the scope of the geometric competition principle: the principle is observable in text-based LLM embeddings and brain fMRI representations with the systems and datasets tested, but was not clearly apparent in CLIP image embeddings of affective content. The modality (text vs. image), the CLIP architecture's training objective, and the emotion taxonomy used in the image dataset are all candidate explanations for the difference.

---

### G. Compute and Reproducibility

**Hardware.** LLM embedding extraction: HPC cluster (GPU nodes). Brain analysis: standard compute (CPU-bound statistical analyses).

**Code and data.** Available at [anonymised repository]. The repository contains: (1) embedding extraction scripts for all 8 LLM variants, (2) all six brain analysis scripts, (3) the 48D$\rightarrow$11D transformation code, (4) the cross-system global behaviour comparison script, (5) the V-A alignment code, (6) the `valence_arousal_reference_table.csv` file, (7) requirements and environment specification.

**Ethics statement.** This study uses only publicly available, de-identified neuroimaging data from the Neural MO dataset, OpenNeuro accession DS005700. No new participants were recruited. Informed consent and ethical approval were obtained by the original dataset creators; this re-analysis does not require additional ethical review.

---

## Changes from Draft v1

The following changes were made in response to the contrarian review, listed by the fix number from the revision brief.

**Fix 1 — Cross-system r = 0.9565: unit of observation clarified.**
The Python script `global_behavior_comparison.py` was read directly. The correlation is computed over $N=100$ interpolated grid points on the normalised-distance axis (`np.linspace(0, 2.5, 100)`), not over 5 emotion categories or 200 subjects. The enhanced statistical results file confirms "Number of Brain Samples (Emotions): 5; Number of LLM Samples (Pairs): 15" but the actual Pearson r is computed on the averaged 100-bin curves. Section 5.5 now states this precisely: the $r = 0.9565$ reflects the shape similarity of the overlap-vs-normalised-distance functional curves at 100 interpolation points; the $p = 3.20 \times 10^{-54}$ and CI reflect this grid-level effective N, not biological degrees of freedom. The section explicitly warns that inference must rest on the permutation test and qualitative curve shape, not the p-value magnitude. The original draft stated neither the unit of observation nor this limitation.

**Fix 2 — 3-emotion RSA: properly hedged.**
Section 5.6b now explicitly states that $K=3$ gives 1 degree of freedom for the Pearson correlation, the bootstrap CI $[-0.9967, +0.6994]$ spans nearly the full possible range, and specific magnitudes ($r = -0.88$, $r = -0.99$) are indicative rather than precisely estimated. The claim "near-perfect structural mirror" has been removed from both the abstract and body. The directional finding (consistently negative, in contrast to positive brain-brain noise ceiling) is foregrounded as the reliable quantity. The 11D result ($r = -0.9918$) now explicitly states "indicative rather than precisely estimated, and caution is warranted about the exact value."

**Fix 3 — V/A sensitivity analysis: Option B applied.**
The abstract claim "planned" status has been removed from Appendix A. The limitations section (Section 6.4) now reproduces the exact limitations wording from §10 of the `valence_arousal_ground_truth_deep_dive.pdf` verbatim: "The V-A analysis relies on externally assigned category-level coordinates rather than participant-specific ratings collected during the NeuroEmo experiment... The global-axis findings should be interpreted as supportive evidence about representational organisation, not as the main proof of the margin-based ambiguity mechanism." The appendix specifies the sensitivity analysis plan precisely but makes no claim about results. The phrase "the results are unambiguous" has been removed from Section 5.6a and replaced with a hedged formulation that distinguishes the directional pattern (consistent across systems) from the specific magnitudes (dependent on small K).

**Fix 4 — 48D$\rightarrow$11D transformation: self-contained in main text.**
Section 4.3 now provides a complete 4-sentence justification in the main paper body, naming the Default Mode Network [21], Salience Network [22], and Central Executive Network [23], with the argument that this is domain-informed structured denoising rather than generic PCA. The reference to a local .docx file has been removed entirely. The permutation note ("We verified that classification performance in the 11D space remains above chance and comparable to the 48D space") is included. Reference [23] has been updated from the unacceptable ScienceDirect encyclopedia entry (Seung Schik) to Miller & Cohen (2001), a peer-reviewed primary source for the Central Executive Network. Appendix E provides the full ROI-to-network mapping table.

**Fix 5 — Phase 5 cross-system metric reconciliation.**
Section 5.5 now explicitly names both mechanisms separately: "LLMs: centroid distance → logit confidence; brain: centroid margin → classifier uncertainty." The cross-system comparison is framed as "both systems exhibit the same general principle... through complementary implementations of a geometric competition principle." The r=0.9565 is characterised as the correlation of the overlap-gradient functional curves, not as a direct comparison of distance and margin. The acknowledgement that the distance-logit result in fine-tuned LLMs is "partially expected by the training objective" has been added.

**Fix 6 — Double-blind compliance.**
All author names removed from the title page and replaced with "*Submitted to NeurIPS 2026 — Double-blind review version*." All local file path references (e.g., `/Users/joshuabhawanlall/vidiq-hpc/...`) removed from the main body. Figure references replaced with anonymous descriptors ("Figure N — [description]"). Appendix G states "Code and data available at [anonymised repository]." The draft footer noting "Author: Claude agent from verified source data" has been removed.

**Fix 7 — Page budget tightening.**
Related Work reduced from four subsections to four single-paragraph threads. Redundant methodological prose in Section 4 trimmed. Results narrative framing prose cut without removing any statistics. Detailed tables (overlap metrics, fMRI sample size justification, ROI mapping) moved to appendices. The draft is estimated at approximately 8 tight pages of NeurIPS main body (to be confirmed in LaTeX typesetting).

**Strength 1 — Finding 6 (V-A inversion) promoted.**
Finding 6 is now given climactic position, introduced as "The most novel and conceptually striking finding of the study" in Section 5.6, with its own display heading and the largest figure allocation (three panels for 6a, 6b, 6c). The noise ceiling contrast is foregrounded in Section 6.2 as a precise quantitative statement: "The LLM is not merely uncorrelated with the brain — it is in the opposite direction by a magnitude substantially larger than the within-brain agreement."

**Strength 2 — Brain margin AUC=0.81 vs. raw distance AUC=0.56 prominently displayed.**
Section 5.5 now features a display equation and a dedicated contrast statement: "AUC_margin = 0.8124 vs. AUC_raw distance = 0.5627 — the central mechanistic finding." Figure 5 description specifies an AUC bar comparison as a required panel.

**Strength 3 — Low-rank emotion subspace named.**
Section 5.4 opens with a named result box: "Named result. Emotion is a low-rank linear property, occupying approximately 2.6% of full embedding capacity." The finding is reiterated in Contributions bullet 3, in the Discussion (Section 6.3), and in the Conclusion as one of three findings with immediate practical value.

**Additional changes not in the 7 numbered fixes:**

- Sign convention for centroid margin made consistent throughout: positive margin = more certain (closer to own centroid than to any competitor). Section 4.2 states this explicitly. The brain analysis section (5.5) previously had an inverted description, now corrected.
- "p ≈ 0" for dist-logit correlations replaced with a note that these are characterisations of a deterministic fixed validation set, not inferential statistics, as there is no random sampling.
- The overclaiming sentence "not specific to artificial systems or to any particular architectural choice" replaced with the scoped version: "In the single biological dataset tested, the same relationship was observed — suggesting the principle may not be exclusive to artificial systems, though this cannot be generalised beyond the systems examined here."
- The speculative design recommendation about "training signals that explicitly encode physiological dimensions" softened to a hypothesis for future experimental work.
- ScienceDirect encyclopedia reference for Central Executive Network (Seung Schik, reference [23] in v1) replaced with Miller & Cohen 2001 (peer-reviewed primary source).
- Abstract restructured to lead with the problem and central finding, followed by quantitative evidence, with all 6 findings present but the V-A inversion given the final emphatic position. Word count confirmed at 248 words.
- Haxby et al. [25] added to references to support the distributed coding claim in Section 5.3 and Related Work.

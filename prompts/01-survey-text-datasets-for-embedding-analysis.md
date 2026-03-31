Survey existing text datasets that could be used to study LLM embedding space with simple semantic contrasts.

Objective:
- Find datasets that are well-suited for small, controlled embedding experiments.
- Prioritize binary or low-complexity semantic distinctions such as `good vs bad`, `happy vs sad`, `positive vs negative`, `formal vs informal`, or other simple human-interpretable contrasts.
- Prefer datasets that are easy to obtain, easy to subset, and easy to explain in a research note.

What to optimize for:
- Short text examples rather than long documents.
- Clear labels or obvious class structure.
- Public availability with a straightforward license or usage terms for research.
- Enough examples to support exploratory analysis, but not so large or messy that preprocessing becomes the main task.
- Coverage of simple semantic axes that can plausibly appear as separable directions, clusters, or gradients in embedding space.

What to avoid:
- Datasets that require heavy annotation cleanup before they are usable.
- Tasks that are primarily syntactic, retrieval-heavy, multilingual, or domain-specialized unless they are unusually clean and simple.
- Benchmarks whose labels are ambiguous, noisy, or poorly aligned with human-interpretable semantic dimensions.
- Very large corpora where the practical first step would be infrastructure work instead of analysis.

Deliverable:
- Produce a ranked shortlist of approximately 8 to 12 candidate datasets.
- For each candidate, include:
  - Dataset name
  - Source or host
  - What the label structure is
  - Typical text length
  - Why it is useful for embedding-space analysis
  - Main drawback or risk
  - Whether it is especially good for a simple contrast such as `good vs bad` or `happy vs sad`
- Call out the top 3 recommendations explicitly.

Analysis expectations:
- Distinguish between sentiment datasets and other semantic-contrast datasets.
- Note whether the dataset supports:
  - binary classification
  - ordinal structure
  - clustered categories
  - continuous or near-continuous semantic gradients
- Note any likely confounds, such as topic leakage, class imbalance, strong lexical shortcuts, or label noise.
- Prefer datasets where we can plausibly test whether embeddings encode a clean semantic direction instead of merely memorizing surface tokens.

Output structure:
1. A short summary of the best dataset types for this purpose.
2. A ranked table of candidate datasets.
3. A recommendation section naming the top 3 datasets to start with and why.
4. A final section called `Suggested first experiments` with 3 to 5 concrete experiments we could run on the shortlisted datasets using embeddings.

Working assumptions:
- The downstream goal is research and intuition-building, not product benchmarking.
- Simplicity is a feature.
- If there is a tradeoff between benchmark prestige and experimental clarity, choose experimental clarity.

If uncertain between two datasets, prefer the one that is cleaner, smaller, and easier to interpret.

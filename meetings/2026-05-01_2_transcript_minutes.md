# Meeting Minutes — Brain Context Retention, Valence/Arousal Mapping, and Paper Storyline

**Date:** 1 May 2026, 20:44  
**Duration:** 40m 48s  
**Attendees:** Daniel Sikar, Pritish Ranjan (PG-Verma), Amy  
**Transcript:** `2026-05-01_2_transcript.md`

---

## Meeting Purpose

- Review the remaining brain-side experiment phases needed for the paper.
- Clarify the context-retention comparison between fine-tuned LLM embeddings, pretrained embeddings, and brain data.
- Resolve the immediate citation question around the valence/arousal table.

---

## Brain Context Retention Across Dimensions

- The team reviewed the `checking_context_retention_across_dimensions` phase under `experiments/brain_embedding_understanding/`.
- The procedure mirrors the earlier LLM analysis:
  - train a regression/classification model
  - identify the most important direction from the learned weights
  - remove that direction
  - retrain iteratively
  - track how predictive accuracy decays as more directions are removed
- Three curves were compared:
  - pretrained MPNet embeddings
  - fine-tuned Qwen embeddings
  - brain data

### Main Interpretation

- The fine-tuned model starts with the highest accuracy but decays fastest.
- This was interpreted as evidence that fine-tuning compresses emotional signal into a smaller set of high-priority directions.
- The pretrained model retains signal more gradually, suggesting a more distributed representation.
- The brain data starts from a lower accuracy level but remains above chance and decays more steadily, again suggesting a more distributed retention of information across directions.
- The agreed reading is not that the brain has no emotional context, but that its emotional information appears less compressed than in the fine-tuned model.

---

## Valence / Arousal Dimensional Reduction

- The next reviewed phase reduced both brain and embedding representations down to two dimensions using PCA.
- Those PCA dimensions were then compared with valence/arousal labels assigned to the emotion classes.
- The intent of the test is to ask whether the low-dimensional geometry aligns with the familiar valence/arousal framing used in affective science.

### Working Result

- For the LLM embeddings, one PCA axis was reported to align strongly with valence and another with arousal.
- The strongest reported correlations were approximately:
  - **0.98** for one axis with valence
  - **0.82** for another axis with arousal
- For the brain data, the stronger alignment appeared to be with **arousal**, while valence was less clearly recovered in the same way.

### Interpretation Agreed in the Meeting

- The current working interpretation is that both systems encode both kinds of affective information, but they may prioritize them differently in their geometry.
- The LLM space appeared to separate classes more strongly by valence.
- The brain-space analysis appeared to emphasize arousal more strongly.
- The team was careful not to overclaim: this is an interpretation of the current representation and preprocessing, not a claim that the brain lacks valence information.

---

## Citation and Grounding of the Valence/Arousal Table

- A major concern raised during the meeting was that the valence/arousal table must not look arbitrary if it is central to the paper’s findings.
- The exact numeric values were acknowledged as operational labels rather than directly measured physical quantities.
- The team agreed they must be backed by cited affective-science literature and presented carefully.

### Sources Identified During the Meeting

- **J. A. Russell (1980s circumplex model of affect)** was identified as a foundational source.
- **DEAP: A Database for Emotion Analysis Using Physiological Signals** was identified as a more recent, heavily cited operational source using similar valence/arousal framing.
- Amy confirmed that these references, plus additional relevant material, would be assembled into the working `LLM/brain` notes bundle for the team.

### Wording Decision

- The paper should not present the table as if the numbers are physically exact brain constants.
- The safer framing is that these are **well-documented affective coordinates / operational labels** used to examine whether the geometry aligns with established affective structure.

---

## Paper Storyline Implications

- The meeting reinforced the central paper claim: geometric structure may be shared across LLM embeddings and brain-derived representations.
- If both systems show comparable organization under centroid, density, margin, and valence/arousal-style analyses, that strengthens the broader “universal geometry” narrative.
- Reproducibility was explicitly discussed as important. The long-term value of the work would be much stronger if the same geometric story transfers across datasets, systems, or even species.
- The intended shift is from “interesting one-off result” to “candidate universal property of intelligence-related representation.”

---

## Constraints and Caveats Raised

- The valence/arousal mapping must be backed by literature or presented as an example/operational framework rather than as unquestionable ground truth.
- Terminology should avoid overclaiming what the brain “does” versus what the current representation makes visible.
- Missing or misplaced report artifacts, such as the HTML summary for the valence/arousal step, need to be consolidated so the evidence trail is clean before paper drafting.

---

## Action Points

| Owner | Action |
|---|---|
| Pritish | Consolidate the context-retention and valence/arousal results into a clean, referenceable report set. |
| Pritish | Recover or regenerate the missing HTML/output summary for the valence/arousal phase if needed. |
| Pritish | Keep the interpretation conservative: LLM and brain may prioritize different affective axes, but avoid overclaiming. |
| Amy | Provide the cited valence/arousal references and related affective-science material in the promised `LLM/brain` notes bundle. |
| Daniel | Use the clarified citation framing when drafting the paper’s affective-geometry section. |
| Team | Carry these results into the paper-structure discussion and keep the shared-geometry storyline consistent across text and brain sections. |

---

## Open Questions

- What is the cleanest way to describe the valence/arousal table: exact values, ranges, or literature-guided operational labels?
- How should the paper present the difference between LLM valence emphasis and brain arousal emphasis without oversimplifying either system?
- Which parts of this analysis are strong enough for the main paper, and which should remain supporting evidence or appendix material?

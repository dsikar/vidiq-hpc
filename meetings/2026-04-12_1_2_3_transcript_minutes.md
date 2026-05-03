# Meeting Minutes — Density Geometry Corrections, Overlap Definition, and HPC Training Setup

**Date:** 12 April 2026  
**Sessions:** 17:48, 18:09, 19:43  
**Attendees:** Daniel Sikar, Pritish Ranjan (PG-Verma)  
**Transcripts:** `2026-04-12_1_transcript.md`, `2026-04-12_2_transcript.md`, `2026-04-12_3_transcript.md`

---

## Meeting Purpose

- Correct the interpretation of the radial-density plots used in the embedding geometry work.
- Clarify the exact definition of overlap between emotion classes.
- Record the current centroid-distance and radial-distance observations.
- Capture the initial setup for training a dedicated multi-class text model on HPC.

---

## Density Geometry Correction

- A key correction was made to the previous interpretation of the radial plots.
- The old plot was effectively measuring **number of points per band**, not density in the strict geometric sense.
- Because the shell volume increases as radius increases in high-dimensional space, the apparent increase in “density” was partly caused by larger capture volume, not by a genuinely denser region.
- Once volume is taken into account, the **true density per unit volume** does not rise to a later peak. Instead, it falls away from the centroid.
- The y-axis on the previous plots should therefore be renamed away from “density” if it is still counting points in bands.

### Main Conceptual Consequence

- There is a **void near the centroid**: for the current setup, no points appear between the centroid and roughly distance **7.5**.
- This was treated as an important qualitative finding:
  - there is no data exactly at the centroid
  - there may be no such thing as a “pure” emotion point in this dataset
  - every observed point appears to be some mixture, not a perfect centroidal prototype

---

## Overlap Definition Clarified

- The overlap rule was explicitly corrected.
- Overlap is **not** defined by crossing some radius threshold.
- A point is counted as overlapping when its distance to another class centroid is **less** than its distance to its own labeled class centroid.
- This means a point may overlap with:
  - one other class
  - several other classes
  - or none at all

### Agreed Interpretation

- Overlap is therefore a **relative centroid-distance condition**, not a shell-distance condition.
- This clarification should be preserved in future write-ups so the overlap metric is not misdescribed.

---

## Centroid and Radial Observations

- A centroid-distance matrix for the balanced dataset was noted for the record.
- In that matrix:
  - sadness is nearest to anger
  - sadness is furthest from joy
- A radial distance scatter summary was also discussed.
- The current reading from that plot is:
  - **love** is the tightest class around its centroid
  - **fear** has the furthest extreme points from its centroid
  - **anger** is also relatively spread out

### Numbers Explicitly Mentioned

- Love:
  - closest point about **7.44**
  - furthest point about **11.32**
- Anger:
  - closest point about **7.32**
  - furthest point about **12.70**
- Fear:
  - furthest point about **12.77**

- Percentile summaries were also mentioned, but the interpretation was left less settled than the centroid/radial observations above.

---

## HPC Training Setup

- The later session shifted from geometry interpretation to execution planning on HPC.
- A new experiment directory had been created under `experiments/text_model`.
- The intended model setup is:
  - a transformer backbone
  - a 768-dimensional embedding layer of interest
  - fully connected layers producing logits
  - a five-class output corresponding to the five target emotions

### Model Choice

- The current starting point is **Qwen 3 1.7B**.
- The reasoning was that 600M parameters felt too small, while 1.7B looked like a viable initial size.
- The exact model choice is not fixed; changing the model later is acceptable so long as:
  - a 768-sized embedding representation is still available for downstream analysis
  - the final model still produces the five-class logits needed for the experiment

### HPC Constraints

- Current HPC access goes through a Windows machine.
- The immediate next step is to confirm that a basic run can be launched successfully on the cluster.
- Once that works:
  - clone the repo on HPC
  - create the required environment
  - run the training
  - later extract embeddings from the relevant internal layer(s)

### Intended Follow-On

- After training, the same family of geometry experiments used earlier for pretrained embeddings should be rerun on the trained model.
- The team explicitly left room to modify scripts, model choice, and HPC batch setup as needed; the experiment itself matters more than preserving the exact initial implementation.

---

## Decisions Recorded

- Re-label the current radial plot output so it does not claim to be density if it is only counting points per band.
- Keep the void-near-centroid observation as a potentially important qualitative result.
- Use the corrected overlap definition going forward: overlap is based on relative centroid distance.
- Proceed with the HPC training experiment using the current text-model setup as a starting point.

---

## Action Points

| Owner | Action |
|---|---|
| Pritish | Update the interpretation and labelling of the radial plots so band counts are not described as true density. |
| Pritish | Preserve the void-near-centroid finding in the notes and future write-up. |
| Pritish | Use the corrected overlap definition consistently in code, reports, and discussion. |
| Pritish | Keep the new text-model experiment structure and README ready for HPC execution. |
| Daniel | Bring up the HPC environment, verify cluster execution works, and then clone/run the experiment there. |
| Daniel + Pritish | After training, rerun the embedding-geometry workflow on the new trained model outputs. |

---

## Open Questions

- What is the best formal term for the old radial plot if it is not true density: point count per shell, point count per band, or something more precise?
- How prominent should the “no pure emotion / centroid void” observation be in the paper?
- Which trained-model embeddings should be extracted for the next geometry pass: only the 768 layer, or multiple internal layers for comparison?

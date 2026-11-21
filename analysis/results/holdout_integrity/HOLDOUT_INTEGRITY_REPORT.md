# Temporal holdout integrity report (paper-ready)

**Author:** Aly Dhedhi  
**Scripts:** `scripts/17_temporal_holdout_integrity.py`, `scripts/18_cryptic_temporal_split.py`  
**Artifacts:** this directory

---

## Executive take

The WHO 2021→2023 holdout **passes integrity checks on pool ranking** and **fails the within-gene pncA test**. That is not a bug in the audit — it is the scientific result. The paper must split claims accordingly.

| Claim | Verdict |
|-------|---------|
| Leakage-free model beats drug-distance on pool AUPRC / top-100 | **Holds** |
| Label shuffle clean (0/200) | **Holds** |
| gyrA / rpoB new residues rank near the top *within gene* | **Holds (anecdotes)** |
| Within-gene pncA AUROC ≫ 0.5 on 2023 additions | **Does not hold** |
| Model “just learned pncA” as a gene label | **Ruled out by within-gene z-scoring**; residual failure is **distribution shift** in new pncA sites |
| gyrB S447 | **Clear miss — explained, not patched** |

---

## 1. Reproducible split

| Split | Definition | N | pncA |
|-------|------------|---|------|
| **Train** | WHO 2021 Group 1/2 Assoc-w-R missense residues in the feature table | **130** | **82** |
| **Test** | Newly Assoc-w-R in WHO 2023, same filter | **20** | **16** (+ gyrA 89, rpoB 438, gyrB 447, katG 1) |

Catalogues: `data/who_catalogues/WHO_2021_GARC.csv`, `WHO_2023_GARC.csv`.

### Leakage-free controls applied

1. **gyrB QRDR-B = 461–499** for `inner_distance` (was 495–556)
2. **pncA core = catalytic / Fe-site residues** (was the entire 1–186 domain, which forced `inner_distance=0` for every pncA residue)
3. **Within-gene z-score of all features** before training (blocks gene-identity shortcut)

---

## 2. What holds (pool ranking; train positives removed)

| Metric | Model | Drug-proximity baseline |
|--------|-------|-------------------------|
| AUPRC | **0.186** | 0.015 |
| AUROC | **0.946** | 0.854 |
| Top-100 new residues | **15 / 20** | **0 / 20** |
| Label-shuffle exceedances | **0 / 200** | — |

**Defendable sentence:** Ranking near the drug pocket does not recover the new residues; the model does (15 vs 0 in the top 100 of the pool).

Global top-50 recovers almost none of the 20, because the 130 training positives occupy the top of the full list. **Always report pool ranks beside global ranks** (`test_residue_ranks.csv`).

---

## 3. Paper-facing split (required)

### 3a. pncA — statistics, with a hard caveat

| Check | Result |
|-------|--------|
| Pool top-20 | **3 / 16** |
| Within-gene AUROC | **0.392** (random 0.5) |
| Within-gene AUPRC | **0.072** (random 0.086) |
| Within-gene ranks of the 16 | 84–146 (all mid/low) |

**Why it fails:** 2023 pncA additions are a different class than 2021 training positives.

| Set | Mean homoplasy | Mean SASA | Mean ESM-2 intolerance |
|-----|----------------|-----------|------------------------|
| Train pncA (n=82) | 2.43 | 0.097 | 0.51 |
| Test pncA (n=16) | 0.75 | 0.318 | 1.68 |

Training positives are **frequent, buried, constrained** sites. New catalogue entries are **rarer, more exposed, less constrained** — classic LoF-spread behavior. The model recovers train positives within pncA at AUROC **1.0** (position signal exists for that class) but does **not** generalize to the 2023 LoF-like tail.

**Paper language:** Do **not** claim within-gene discovery of new pncA positions. State that pyrazinamidase resistance is distributed (LoF), and catalogue expansions in pncA are not hotspot-like.

### 3b. gyrA / rpoB / gyrB / katG — anecdotes, not statistics

| Residue | Within-gene rank | Pool rank | Note |
|---------|------------------|-----------|------|
| **gyrA 89** | **6 / 838** | 49 | Strong anecdote |
| **rpoB 438** | **27 / 1166** | 10 | Strong anecdote |
| katG 1 | 284 / 740 | 1991 | Weak |
| **gyrB 447** | **381 / 675** | 3853 | Miss (below) |

Leave-pncA-out: gyrA 89 and rpoB 438 **improve or hold** pool rank when all pncA train positives are dropped — they are not piggy-backing on a pncA gene prior.

---

## 4. gyrB S447 miss (do not patch)

- **Domain:** transducer / TOPRIM junction (not the N-terminal ATPase). Immediately N-terminal to QRDR-B (461–499).
- After the QRDR-B fix, `inner_distance=14`, `drug_proximity≈0.33` — weak pocket signal versus true QRDR residues.
- Pool rank **3853 / ~6220**; within-gene rank **381 / 675**.
- **Write-up only.** Retuning features to catch S447 would be test-set fitting.

---

## 5. Second clinical split (CRyPTIC)

`scripts/18_cryptic_temporal_split.py` + enrichment table.

**Pocket-gene prospective set** (WHO-2021 train; FDR q<0.05 resistance-enriched residues in rpoB/gyrA/gyrB/embB/inhA/eis/rpsL that are *not* in WHO 2021):

| | |
|--|--|
| Test N | **70** (rpoB 36, embB 21, gyrA 6, gyrB 4, inhA 3) — **all non-pncA** |
| Pool AUPRC | 0.029 vs distance 0.012 |
| Pool AUROC | 0.586 |
| Top-100 | 6 vs distance 1 |
| Within-gene AUROC | embB 0.56, gyrA 0.60, gyrB 0.63, inhA **0.79**, rpoB 0.59 |

Useful as a **supplementary non-pncA check**, not as the headline. LoF-heavy katG (477 FDR hits) must stay out of the headline set for the same reason as pncA.

---

## 6. What to put in the paper

### Lead with

1. Leakage-free pool metrics on the WHO time split (AUPRC, top-100 vs distance, shuffle).
2. Pocket-gene anecdotes (gyrA 89, rpoB 438) + leave-pncA-out stability.
3. Existing Tier-4 / Q538L structural story (orthogonal to this holdout).

### State as limitations

1. Within-gene failure on new pncA sites due to LoF distribution shift (show the feature table).
2. gyrB S447 miss with domain explanation.
3. CRyPTIC pocket split is supportive but modest.

### Do not claim

- “8 of 15 new pncA residues in the top 20” under the leakage-free within-gene model (that number is **not** reproduced; honest figure is **3/16** pool top-20, **0/16** within-gene top-20).
- Gene-level statistics for gyrA/rpoB/gyrB from n=1 each.

---

## 7. Files

| File | Contents |
|------|----------|
| `train_positives_who2021.csv` | 130 train residues |
| `test_positives_who2023_new.csv` | 20 test residues |
| `test_residue_ranks.csv` | Global + pool + within-gene ranks |
| `within_gene_metrics.csv` | Per-gene AUROC/AUPRC |
| `leave_pnca_out_non_pnca_hits.csv` | Leave-pncA-out stability |
| `gyrb_S447_miss.json` | Miss write-up |
| `metrics.json` | Machine-readable WHO holdout metrics |
| `cryptic_pocket_metrics.json` | CRyPTIC pocket-gene prospective metrics |
| `cryptic_pocket_test_ranks.csv` | Ranks for 70 pocket-gene test residues |
| `cryptic_residue_enrichment.csv` | Per-residue Fisher enrichment in CRyPTIC |

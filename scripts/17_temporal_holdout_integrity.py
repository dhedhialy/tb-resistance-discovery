#!/usr/bin/env python3
"""
WHO 2021 → 2023 temporal holdout integrity suite.

Reconstructs the catalog time-split (130 Group-1/2 train positives from the
2021 WHO catalogue; new 2023 Assoc-w-R residues as the test set), trains a
leakage-free XGBoost model, and runs the paper-critical checks:

  1. Global holdout metrics vs drug-distance baseline + label shuffle
  2. Pool ranks (train positives removed from the ranking pool)
  3. Within-gene ranks / AUROC / AUPRC (esp. pncA)
  4. Leave-pncA-out retrain (non-pncA hit stability)
  5. Split reporting: pncA (statistics) vs other genes (anecdotes)
  6. gyrB S447 miss write-up (no model patching)

Outputs under analysis/results/holdout_integrity/
"""
from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
WHO_DIR = BASE / "data" / "who_catalogues"
FEAT_PATH = BASE / "analysis" / "results" / "hotspot_model" / "residue_hotspot_data.csv"
OUT = BASE / "analysis" / "results" / "holdout_integrity"
OUT.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "inner_distance",
    "homoplasy_count",
    "homoplasy_alleles",
    "helix_propensity",
    "strand_propensity",
    "hydrophobicity",
    "volume",
    "charge",
    "hbond",
    "rel_position",
    "conservation_blosum",
    "contact_density_seq",
    "sasa_relative",
    "esm2_intolerance",
    "contact_density_3d",
    "drug_proximity",
]

# Leakage-free / conservative core pockets
# - gyrB QRDR-B fixed to 461–499 (was 495–556)
# - pncA uses catalytic / Fe-site pocket residues, NOT the entire 186-aa
#   domain (domain-wide core made inner_distance=0 for every pncA residue
#   and destroyed within-gene ranking)
CORE_BINDING = {
    "rpoB": set(range(426, 453)),
    "katG": set(range(104, 116)) | set(range(270, 331)),
    "embB": set(range(295, 531)),
    "gyrA": set(range(74, 151)),
    "gyrB": set(range(461, 500)),  # QRDR-B fix
    "pncA": {4, 7, 8, 9, 10, 11, 12, 13, 20, 49, 51, 71, 85, 96, 125, 133, 138},
    "rpsL": set(range(23, 99)),
    "inhA": set(range(1, 270)),
    "eis": set(range(48, 105)),
    "tlyA": set(),
    "tap": set(),
    "mmpR5": set(range(36, 158)),
}

# gyrB domain notes (H37Rv / UniProt P9WG45 approximate)
GYRB_DOMAINS = {
    "ATPase": (1, 220),
    "transducer": (221, 460),
    "TOPRIM_QRDR_B": (461, 556),
    "C_terminal": (557, 675),
}


def parse_grade(blob: str) -> int | None:
    if re.search(r"1\)\s*Assoc w R(?!\s*-\s*Interim)", blob):
        return 1
    if "Assoc w R - Interim" in blob or "2) Assoc w R" in blob:
        return 2
    return None


def extract_assoc_residues(csv_path: Path, feat_keys: set[tuple]) -> set[tuple[str, int]]:
    df = pd.read_csv(csv_path)
    rows: set[tuple[str, int]] = set()
    for _, r in df.iterrows():
        blob = " ".join(str(r.get(c, "")) for c in ["PREDICTION", "EVIDENCE", "OTHER"])
        grade = parse_grade(blob)
        if grade is None and r["PREDICTION"] == "R":
            grade = 1 if "Interim" not in blob else 2
        if grade not in (1, 2):
            continue
        mut = str(r["MUTATION"])
        m = re.match(r"^([A-Za-z0-9_]+)@([A-Z\*])(\d+)([A-Z\*])", mut)
        if not m:
            continue
        gene, pos = m.group(1), int(m.group(3))
        if (gene, pos) in feat_keys:
            rows.add((gene, pos))
    return rows


def recompute_inner_distance(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dists = []
    for _, row in out.iterrows():
        core = CORE_BINDING.get(row["gene"], set())
        if not core:
            dists.append(500)
        else:
            dists.append(min(abs(int(row["residue_pos"]) - p) for p in core))
    out["inner_distance"] = dists
    return out


def within_gene_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score features inside each gene so the model cannot use gene identity."""
    out = df.copy()
    for c in cols:
        out[c] = out.groupby("gene")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8)
        )
    return out


def train_xgb(X: np.ndarray, y: np.ndarray) -> XGBClassifier:
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    clf = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    clf.fit(X, y)
    return clf


def bootstrap_auprc(y: np.ndarray, s: np.ndarray, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = []
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0:
        return float("nan"), float("nan"), float("nan")
    for _ in range(n):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        vals.append(average_precision_score(y[idx], s[idx]))
    arr = np.asarray(vals)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def label_shuffle_test(X: np.ndarray, y: np.ndarray, real_auprc: float, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    exceed = 0
    scores = []
    for i in range(n):
        y_shuf = rng.permutation(y)
        # Keep feature matrix; only labels shuffled. Fit quick model on all rows
        # then score — for speed use the same X with shuffled y and evaluate
        # AUPRC of probabilities from a model trained on shuffled labels.
        clf = train_xgb(X, y_shuf)
        s = clf.predict_proba(X)[:, 1]
        a = average_precision_score(y, s)  # real labels vs shuffle-trained scores
        scores.append(a)
        if a >= real_auprc:
            exceed += 1
    return exceed, scores


def within_gene_metrics(df: pd.DataFrame, test_set: set[tuple[str, int]], score_col: str):
    rows = []
    for gene, gdf in df.groupby("gene"):
        y = gdf.apply(lambda r: 1 if (r["gene"], int(r["residue_pos"])) in test_set else 0, axis=1).values
        if y.sum() == 0:
            continue
        s = gdf[score_col].values
        # ranks within gene (1 = best)
        order = np.argsort(-s)
        ranks = np.empty(len(s), dtype=int)
        ranks[order] = np.arange(1, len(s) + 1)
        test_ranks = ranks[y == 1]
        try:
            auroc = float(roc_auc_score(y, s))
        except ValueError:
            auroc = float("nan")
        auprc = float(average_precision_score(y, s))
        rows.append(
            {
                "gene": gene,
                "n_residues": int(len(gdf)),
                "n_test_positives": int(y.sum()),
                "auroc": auroc,
                "auprc": auprc,
                "random_auprc": float(y.mean()),
                "median_test_rank": float(np.median(test_ranks)),
                "mean_test_rank": float(np.mean(test_ranks)),
                "top10_hits": int((test_ranks <= 10).sum()),
                "top20_hits": int((test_ranks <= 20).sum()),
                "test_ranks": ",".join(str(int(x)) for x in sorted(test_ranks)),
            }
        )
    return pd.DataFrame(rows)


def domain_for_gyrb(pos: int) -> str:
    for name, (lo, hi) in GYRB_DOMAINS.items():
        if lo <= pos <= hi:
            return name
    return "unknown"


def main():
    feat = pd.read_csv(FEAT_PATH)
    feat_keys = set(zip(feat["gene"], feat["residue_pos"].astype(int)))
    print(f"Feature table: {len(feat)} residues, {feat['gene'].nunique()} genes")

    s21 = extract_assoc_residues(WHO_DIR / "WHO_2021_GARC.csv", feat_keys)
    s23 = extract_assoc_residues(WHO_DIR / "WHO_2023_GARC.csv", feat_keys)
    train_pos = sorted(s21)
    test_pos = sorted(s23 - s21)
    print(f"Train positives (WHO 2021 G1/G2): {len(train_pos)}  pncA={sum(g=='pncA' for g,_ in train_pos)}")
    print(f"Test positives (new in 2023):     {len(test_pos)}  {Counter(g for g,_ in test_pos)}")

    # Primary paper split: exclude katG from the *headline* test (anecdote genes
    # are gyrA/rpoB/gyrB singles); keep full test for completeness tables.
    test_primary = [(g, p) for g, p in test_pos if g != "katG"]
    test_set = set(test_pos)
    test_primary_set = set(test_primary)
    train_set = set(train_pos)

    pd.DataFrame(train_pos, columns=["gene", "residue_pos"]).assign(split="train_2021").to_csv(
        OUT / "train_positives_who2021.csv", index=False
    )
    pd.DataFrame(test_pos, columns=["gene", "residue_pos"]).assign(split="test_new_2023").to_csv(
        OUT / "test_positives_who2023_new.csv", index=False
    )

    # Leakage-free features: recompute inner_distance with QRDR-B 461–499
    # and pncA active-site pocket (not whole domain).
    df0 = recompute_inner_distance(feat)
    g447 = df0[(df0.gene == "gyrB") & (df0.residue_pos == 447)].iloc[0]
    print(
        f"gyrB S447 after QRDR-B fix: inner_distance={g447['inner_distance']} "
        f"drug_proximity={g447['drug_proximity']:.3f} domain={domain_for_gyrb(447)}"
    )
    pnca_id = df0.loc[df0.gene == "pncA", "inner_distance"]
    print(
        f"pncA inner_distance after active-site core: "
        f"unique={sorted(pnca_id.unique())[:8]}... mean={pnca_id.mean():.1f}"
    )

    # Within-gene z-scored features: removes gene-mean effects so the model
    # must learn *position* signal (answers "did it just learn pncA?").
    df = within_gene_zscore(df0, FEATURE_COLS)

    # Labels for training: only 2021 positives (test residues are negatives during train)
    y_train_mask = df.apply(lambda r: (r["gene"], int(r["residue_pos"])) in train_set, axis=1).astype(int)
    y_test = df.apply(lambda r: (r["gene"], int(r["residue_pos"])) in test_set, axis=1).astype(int)
    y_test_primary = df.apply(lambda r: (r["gene"], int(r["residue_pos"])) in test_primary_set, axis=1).astype(int)

    X_raw = df[FEATURE_COLS].fillna(0).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("\nTraining leakage-free WITHIN-GENE model on WHO-2021 positives...")
    clf = train_xgb(X, y_train_mask.values)
    scores = clf.predict_proba(X)[:, 1]
    df = df.copy()
    df["holdout_score"] = scores
    df["is_train_pos"] = y_train_mask.values
    df["is_test_pos"] = y_test.values
    # Keep raw (non-z) drug proximity for the baseline from df0
    df["drug_proximity_raw"] = df0["drug_proximity"].values
    df["inner_distance_raw"] = df0["inner_distance"].values

    # --- Global metrics on full table (test vs all non-test; train may dominate top) ---
    def metrics_block(y, s, label):
        auroc = float(roc_auc_score(y, s)) if y.sum() and (1 - y).sum() else float("nan")
        auprc = float(average_precision_score(y, s))
        mean, lo, hi = bootstrap_auprc(y, s, n=1000)
        return {
            "label": label,
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "auroc": auroc,
            "auprc": auprc,
            "auprc_bootstrap_mean": mean,
            "auprc_ci95_lo": lo,
            "auprc_ci95_hi": hi,
            "random_auprc": float(y.mean()),
        }

    # Evaluate only on non-train residues (pool): removes 2021 positives from ranking
    pool_mask = ~y_train_mask.values.astype(bool)
    y_pool = y_test.values[pool_mask]
    s_pool = scores[pool_mask]
    y_pool_primary = y_test_primary.values[pool_mask]
    s_pool_primary = scores[pool_mask]
    dist_pool = df.loc[pool_mask, "drug_proximity_raw"].values

    global_metrics = [
        metrics_block(y_test.values, scores, "global_all_residues"),
        metrics_block(y_pool, s_pool, "pool_exclude_train_positives"),
        metrics_block(y_pool_primary, s_pool_primary, "pool_primary_no_katG"),
        metrics_block(y_pool, dist_pool, "pool_drug_proximity_baseline"),
        metrics_block(y_test.values, df["drug_proximity_raw"].values, "global_drug_proximity_baseline"),
    ]

    # Top-K recall on pool
    def topk(y, s, k):
        order = np.argsort(-s)[:k]
        return int(y[order].sum()), float(y[order].sum() / max(y.sum(), 1))

    for k in (20, 50, 100):
        n, frac = topk(y_pool, s_pool, k)
        global_metrics.append({"label": f"pool_model_top{k}", "n_pos_recovered": n, "recall": frac})
        n2, frac2 = topk(y_pool, dist_pool, k)
        global_metrics.append({"label": f"pool_distance_top{k}", "n_pos_recovered": n2, "recall": frac2})
        n3, frac3 = topk(y_pool_primary, s_pool_primary, k)
        global_metrics.append({"label": f"pool_primary_model_top{k}", "n_pos_recovered": n3, "recall": frac3})

    print("\n=== Pool metrics (train positives removed) ===")
    pool_auprc = average_precision_score(y_pool, s_pool)
    dist_auprc = average_precision_score(y_pool, dist_pool)
    pool_auroc = roc_auc_score(y_pool, s_pool)
    dist_auroc = roc_auc_score(y_pool, dist_pool)
    print(f"Model  AUPRC={pool_auprc:.3f} AUROC={pool_auroc:.3f}")
    print(f"Dist   AUPRC={dist_auprc:.3f} AUROC={dist_auroc:.3f}")
    for k in (20, 50, 100):
        print(f"Top-{k}: model={topk(y_pool, s_pool, k)}  dist={topk(y_pool, dist_pool, k)}")

    # Label shuffle on pool task: train on shuffled train labels, score, compare pool AUPRC
    print("\nLabel-shuffle test (200)...")
    # Faster honest shuffle: permute train labels only, retrain, measure pool AUPRC on real test
    rng = np.random.default_rng(0)
    y_tr = y_train_mask.values.copy()
    exceed = 0
    shuffle_scores = []
    for i in range(200):
        y_shuf = y_tr.copy()
        # shuffle among all residues
        y_shuf = rng.permutation(y_shuf)
        clf_s = train_xgb(X, y_shuf)
        s_s = clf_s.predict_proba(X)[:, 1][pool_mask]
        a = average_precision_score(y_pool, s_s)
        shuffle_scores.append(a)
        if a >= pool_auprc:
            exceed += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/200  exceed={exceed}")
    print(f"Shuffle exceedances: {exceed}/200")

    # --- Per-residue ranks: global and pool ---
    df["global_rank"] = df["holdout_score"].rank(ascending=False, method="min").astype(int)
    pool_df = df.loc[pool_mask].copy()
    pool_df["pool_rank"] = pool_df["holdout_score"].rank(ascending=False, method="min").astype(int)
    # within-gene ranks
    df["within_gene_rank"] = df.groupby("gene")["holdout_score"].rank(ascending=False, method="min").astype(int)

    test_rows = []
    for g, p in test_pos:
        row = df[(df.gene == g) & (df.residue_pos == p)].iloc[0]
        pr = pool_df[(pool_df.gene == g) & (pool_df.residue_pos == p)]
        test_rows.append(
            {
                "gene": g,
                "residue_pos": p,
                "wt_aa": row["wt_aa"],
                "holdout_score": float(row["holdout_score"]),
                "global_rank": int(row["global_rank"]),
                "pool_rank": int(pr["pool_rank"].iloc[0]) if len(pr) else None,
                "within_gene_rank": int(row["within_gene_rank"]),
                "n_gene_residues": int((df.gene == g).sum()),
                "inner_distance": int(row["inner_distance_raw"]),
                "drug_proximity": float(row["drug_proximity_raw"]),
                "homoplasy_count": int(row["homoplasy_count"]),
                "primary_test": g != "katG",
                "report_class": "statistic" if g == "pncA" else "anecdote",
            }
        )
    test_table = pd.DataFrame(test_rows).sort_values(["report_class", "pool_rank"])
    test_table.to_csv(OUT / "test_residue_ranks.csv", index=False)

    pnca = test_table[test_table.gene == "pncA"]
    print("\n=== pncA pool top-20 claim ===")
    print(f"pncA test in pool top-20: {(pnca['pool_rank'] <= 20).sum()} / {len(pnca)}")
    print(f"pncA test in pool top-50: {(pnca['pool_rank'] <= 50).sum()} / {len(pnca)}")
    print(f"All test in global top-50: {(test_table['global_rank'] <= 50).sum()} / {len(test_table)}")

    # Within-gene metrics
    wg = within_gene_metrics(df, test_set, "holdout_score")
    wg.to_csv(OUT / "within_gene_metrics.csv", index=False)
    print("\n=== Within-gene ===")
    print(wg.to_string(index=False))

    # --- Leave-pncA-out retrain ---
    print("\nLeave-pncA-out retrain...")
    y_nopnca = y_train_mask.values.copy()
    pnca_train_mask = df["gene"].eq("pncA") & y_train_mask.values.astype(bool)
    y_nopnca[pnca_train_mask.values] = 0
    print(f"  train positives after dropping pncA: {int(y_nopnca.sum())} (was {int(y_train_mask.sum())})")
    clf2 = train_xgb(X, y_nopnca)
    scores2 = clf2.predict_proba(X)[:, 1]
    df["score_leave_pnca_out"] = scores2
    df["within_gene_rank_lpo"] = df.groupby("gene")["score_leave_pnca_out"].rank(
        ascending=False, method="min"
    ).astype(int)
    pool_df2 = df.loc[pool_mask].copy()
    pool_df2["pool_rank_lpo"] = pool_df2["score_leave_pnca_out"].rank(ascending=False, method="min").astype(int)

    lpo_rows = []
    for g, p in test_pos:
        if g == "pncA":
            continue
        row = df[(df.gene == g) & (df.residue_pos == p)].iloc[0]
        pr = pool_df[(pool_df.gene == g) & (pool_df.residue_pos == p)]
        pr2 = pool_df2[(pool_df2.gene == g) & (pool_df2.residue_pos == p)]
        lpo_rows.append(
            {
                "gene": g,
                "residue_pos": p,
                "pool_rank_full": int(pr["pool_rank"].iloc[0]),
                "pool_rank_leave_pnca_out": int(pr2["pool_rank_lpo"].iloc[0]),
                "within_gene_rank_full": int(row["within_gene_rank"]),
                "within_gene_rank_leave_pnca_out": int(row["within_gene_rank_lpo"]),
                "score_full": float(row["holdout_score"]),
                "score_leave_pnca_out": float(row["score_leave_pnca_out"]),
            }
        )
    lpo = pd.DataFrame(lpo_rows)
    lpo.to_csv(OUT / "leave_pnca_out_non_pnca_hits.csv", index=False)
    print(lpo.to_string(index=False))

    # gyrB 447 write-up
    miss = {
        "gene": "gyrB",
        "residue": "S447",
        "domain": domain_for_gyrb(447),
        "domain_note": (
            "S447 sits in the transducer/TOPRIM junction immediately N-terminal to "
            "QRDR-B (461–499), not in the N-terminal ATPase domain. It is adjacent "
            "to the fluoroquinolone-linked region but outside the conservative "
            "pocket used for inner_distance."
        ),
        "inner_distance_after_qrdr_fix": int(df0[(df0.gene == "gyrB") & (df0.residue_pos == 447)].iloc[0]["inner_distance"]),
        "drug_proximity": float(df0[(df0.gene == "gyrB") & (df0.residue_pos == 447)].iloc[0]["drug_proximity"]),
        "pool_rank": int(test_table.set_index(["gene", "residue_pos"]).loc[("gyrB", 447), "pool_rank"]),
        "global_rank": int(test_table.set_index(["gene", "residue_pos"]).loc[("gyrB", 447), "global_rank"]),
        "within_gene_rank": int(
            test_table.set_index(["gene", "residue_pos"]).loc[("gyrB", 447), "within_gene_rank"]
        ),
        "explanation": (
            "gyrB S447 is a clear miss under the leakage-free model: after fixing "
            "QRDR-B to 461–499 its inner_distance remains large relative to true "
            "pocket residues, so the conservative pocket features give almost no "
            "positive signal. We report the miss and do not retune the model to "
            "recover it (that would be test-set fitting)."
        ),
        "paper_treatment": "anecdote_not_statistics",
    }
    (OUT / "gyrb_S447_miss.json").write_text(json.dumps(miss, indent=2))

    # Summary markdown for the paper
    pnca_top20 = int((pnca["pool_rank"] <= 20).sum())
    if (wg.gene == "pncA").any():
        wg_pnca = wg[wg.gene == "pncA"].iloc[0]
        pnca_auroc_s = f"{wg_pnca['auroc']:.3f}"
        pnca_auprc_s = f"{wg_pnca['auprc']:.3f}"
        pnca_rand_s = f"{wg_pnca['random_auprc']:.3f}"
        pnca_ranks_s = str(wg_pnca["test_ranks"])
    else:
        pnca_auroc_s = pnca_auprc_s = pnca_rand_s = pnca_ranks_s = "n/a"

    top100_model, _ = topk(y_pool, s_pool, 100)
    top100_dist, _ = topk(y_pool, dist_pool, 100)

    summary = f"""# Temporal holdout integrity report

## Split (reproducible from WHO GARC catalogues)

| Split | Definition | N | pncA |
|-------|------------|---|------|
| Train | WHO 2021 Group 1/2 Assoc-w-R missense residues present in the feature table | **{len(train_pos)}** | **{sum(g=='pncA' for g,_ in train_pos)}** |
| Test  | Residues newly Assoc-w-R in WHO 2023 (not in 2021), same filter | **{len(test_pos)}** | **{sum(g=='pncA' for g,_ in test_pos)}** |

Catalogue files: `data/who_catalogues/WHO_2021_GARC.csv`, `WHO_2023_GARC.csv`.
Leakage-free changes:
- `gyrB` core pocket / `inner_distance` uses QRDR-B **461–499**
- `pncA` core pocket uses catalytic/Fe-site residues (not the entire domain)
- All features are **z-scored within gene** before training (blocks gene-identity shortcut)


## What holds (pool ranking: 2021 positives removed)

| Metric | Model | Drug-proximity baseline |
|--------|-------|-------------------------|
| AUPRC | **{pool_auprc:.3f}** | {dist_auprc:.3f} |
| AUROC | **{pool_auroc:.3f}** | {dist_auroc:.3f} |
| Top-100 new residues recovered | **{top100_model} / {int(y_pool.sum())}** | {top100_dist} / {int(y_pool.sum())} |
| Label-shuffle exceedances | **{exceed} / 200** | — |

Bootstrap 95% CI for pool AUPRC: see `metrics.json`.

## Paper-facing split

### pncA (only gene with enough test positives for statistics)

- New pncA residues in pool top-20: **{pnca_top20} / {len(pnca)}**
- Within-gene AUROC: **{pnca_auroc_s}**
- Within-gene AUPRC: **{pnca_auprc_s}** (random = {pnca_rand_s})
- Within-gene ranks of test hits: {pnca_ranks_s}

If within-gene AUROC stays well above 0.5, the model is ranking the right *positions inside pncA*, not merely boosting the gene.

### gyrA / rpoB / gyrB / katG (anecdotes, not statistics)

One new residue each (katG optional). Report ranks; do not claim gene-level performance.

## gyrB S447 miss (do not patch)

{miss['explanation']}

- Domain: **{miss['domain']}** — {miss['domain_note']}
- inner_distance={miss['inner_distance_after_qrdr_fix']}, pool_rank={miss['pool_rank']}, within_gene_rank={miss['within_gene_rank']}

## Leave-pncA-out

See `leave_pnca_out_non_pnca_hits.csv`. If non-pncA pool ranks barely move, gene-identity leakage from pncA is not driving those hits.

## Global vs pool ranks

Only a few test residues enter the global top-50 because the 2021 training positives occupy the top of the full list. **Always report pool ranks alongside global ranks** (`test_residue_ranks.csv`).
"""
    (OUT / "HOLDOUT_INTEGRITY_REPORT.md").write_text(summary)

    metrics_out = {
        "train_n": len(train_pos),
        "train_pnca": sum(g == "pncA" for g, _ in train_pos),
        "test_n": len(test_pos),
        "test_by_gene": dict(Counter(g for g, _ in test_pos)),
        "pool_n": int(pool_mask.sum()),
        "pool_model_auprc": float(pool_auprc),
        "pool_model_auroc": float(pool_auroc),
        "pool_dist_auprc": float(dist_auprc),
        "pool_dist_auroc": float(dist_auroc),
        "pool_model_auprc_ci95": list(bootstrap_auprc(y_pool, s_pool)[1:]),
        "shuffle_exceedances": exceed,
        "shuffle_n": 200,
        "pnca_pool_top20": pnca_top20,
        "pnca_n": int(len(pnca)),
        "global_top50_test_hits": int((test_table["global_rank"] <= 50).sum()),
        "blocks": global_metrics,
        "gyrb_447": miss,
        "within_gene": wg.to_dict(orient="records"),
        "leave_pnca_out": lpo.to_dict(orient="records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics_out, indent=2))
    df[
        [
            "gene",
            "residue_pos",
            "wt_aa",
            "is_train_pos",
            "is_test_pos",
            "holdout_score",
            "score_leave_pnca_out",
            "global_rank",
            "within_gene_rank",
            "inner_distance_raw",
            "drug_proximity_raw",
        ]
    ].to_csv(OUT / "scored_residues.csv", index=False)

    print("\nWrote", OUT)
    print(summary)


if __name__ == "__main__":
    main()

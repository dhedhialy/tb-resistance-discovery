#!/usr/bin/env python3
"""
CRyPTIC phenotype-enriched temporal split.

1. First-seen order of each nonsynonymous residue among phenotyped isolates
2. Resistance enrichment (Fisher exact) vs the gene's primary drug
3. Train = WHO-2021 Assoc-w-R residues first seen in the early half
4. Test  = residues first seen only in the late half AND FDR-significant
   resistance enrichment (gives many non-pncA positives)

Uses phenotype-table order as a release-order proxy when collection years
are unavailable (documented in the output JSON).
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

BASE = Path(__file__).resolve().parent.parent
MUT = BASE / "data" / "cryptic" / "MUTATIONS.csv.gz"
PHENO = BASE / "data" / "metadata" / "cryptic_phenotypes.csv"
FEAT = BASE / "analysis" / "results" / "hotspot_model" / "residue_hotspot_data.csv"
WHO21 = BASE / "analysis" / "results" / "holdout_integrity" / "train_positives_who2021.csv"
OUT = BASE / "analysis" / "results" / "holdout_integrity"

FEATURE_COLS = [
    "inner_distance", "homoplasy_count", "homoplasy_alleles",
    "helix_propensity", "strand_propensity", "hydrophobicity", "volume",
    "charge", "hbond", "rel_position", "conservation_blosum",
    "contact_density_seq", "sasa_relative", "esm2_intolerance",
    "contact_density_3d", "drug_proximity",
]

CORE_BINDING = {
    "rpoB": set(range(426, 453)),
    "katG": set(range(104, 116)) | set(range(270, 331)),
    "embB": set(range(295, 531)),
    "gyrA": set(range(74, 151)),
    "gyrB": set(range(461, 500)),
    "pncA": {4, 7, 8, 9, 10, 11, 12, 13, 20, 49, 51, 71, 85, 96, 125, 133, 138},
    "rpsL": set(range(23, 99)),
    "inhA": set(range(1, 270)),
    "eis": set(range(48, 105)),
    "tlyA": set(),
    "tap": set(),
    "mmpR5": set(range(36, 158)),
}

# Primary CRyPTIC binary phenotype column per gene
GENE_DRUG = {
    "rpoB": "RIF_BINARY_PHENOTYPE",
    "katG": "INH_BINARY_PHENOTYPE",
    "inhA": "INH_BINARY_PHENOTYPE",
    "embB": "EMB_BINARY_PHENOTYPE",
    "gyrA": "MXF_BINARY_PHENOTYPE",
    "gyrB": "MXF_BINARY_PHENOTYPE",
    "eis": "AMI_BINARY_PHENOTYPE",
}


def train_xgb(X, y):
    n_pos = max(int(y.sum()), 1)
    n_neg = max(len(y) - n_pos, 1)
    clf = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos, eval_metric="logloss",
        random_state=42, n_jobs=4,
    )
    clf.fit(X, y)
    return clf


def main():
    if not MUT.exists():
        raise SystemExit("Missing data/cryptic/MUTATIONS.csv.gz")

    drug_cols = sorted(set(GENE_DRUG.values()))
    pheno = pd.read_csv(PHENO, usecols=["UNIQUEID"] + drug_cols)
    pheno = pheno.drop_duplicates("UNIQUEID").reset_index(drop=True)
    pheno["order_idx"] = np.arange(len(pheno))
    cutoff = int(len(pheno) * 0.5)
    order_map = dict(zip(pheno["UNIQUEID"], pheno["order_idx"]))
    print(f"Isolates={len(pheno)} cutoff={cutoff}")

    # Per residue: first_seen, and carrier phenotype counts for its drug
    first_seen: dict[tuple[str, int], int] = {}
    # counts[gene,pos] = [n_R, n_S] among carriers with binary phenotype
    counts: dict[tuple[str, int], list[int]] = defaultdict(lambda: [0, 0])
    # background for each drug: total R and S among all phenotyped
    bg = {}
    for col in drug_cols:
        s = pheno[col]
        bg[col] = (int((s == "R").sum()), int((s == "S").sum()))

    pheno_idx = pheno.set_index("UNIQUEID")
    usecols = [
        "UNIQUEID", "GENE", "AMINO_ACID_NUMBER", "IS_SNP",
        "IS_NONSYNONYMOUS", "IS_FILTER_PASS", "IN_CDS",
    ]
    genes = set(GENE_DRUG)
    n_rows = 0
    for chunk in pd.read_csv(MUT, compression="gzip", usecols=usecols, chunksize=500_000):
        n_rows += len(chunk)
        chunk = chunk[
            chunk["GENE"].isin(genes)
            & chunk["IS_SNP"].astype(bool)
            & chunk["IS_NONSYNONYMOUS"].astype(bool)
            & chunk["IS_FILTER_PASS"].astype(bool)
            & chunk["IN_CDS"].astype(bool)
            & chunk["AMINO_ACID_NUMBER"].notna()
        ].copy()
        if chunk.empty:
            continue
        chunk["AMINO_ACID_NUMBER"] = chunk["AMINO_ACID_NUMBER"].astype(int)
        chunk["order_idx"] = chunk["UNIQUEID"].map(order_map)
        chunk = chunk.dropna(subset=["order_idx"])
        for (gene, pos), sub in chunk.groupby(["GENE", "AMINO_ACID_NUMBER"], sort=False):
            key = (gene, int(pos))
            m = int(sub["order_idx"].min())
            if key not in first_seen or m < first_seen[key]:
                first_seen[key] = m
            drug = GENE_DRUG[gene]
            # unique carriers only
            carriers = sub["UNIQUEID"].unique()
            for uid in carriers:
                if uid not in pheno_idx.index:
                    continue
                val = pheno_idx.at[uid, drug]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                if val == "R":
                    counts[key][0] += 1
                elif val == "S":
                    counts[key][1] += 1
        if n_rows % 5_000_000 < 500_000:
            print(f"  scanned {n_rows:,}; timed={len(first_seen)}")

    # Enrichment table
    rows = []
    for key, (n_r, n_s) in counts.items():
        gene, pos = key
        drug = GENE_DRUG[gene]
        bg_r, bg_s = bg[drug]
        # carriers vs non-carriers contingency
        # [[R_carrier, S_carrier], [R_non, S_non]]
        r_non = bg_r - n_r
        s_non = bg_s - n_s
        if n_r + n_s < 3:
            continue
        table = [[n_r, n_s], [max(r_non, 0), max(s_non, 0)]]
        oddsr, p = fisher_exact(table, alternative="greater")
        rows.append(
            {
                "gene": gene,
                "residue_pos": pos,
                "first_seen": first_seen.get(key, 10**9),
                "n_R": n_r,
                "n_S": n_s,
                "odds_ratio": oddsr,
                "pvalue": p,
                "drug_col": drug,
            }
        )
    enr = pd.DataFrame(rows)
    enr["qvalue"] = multipletests(enr["pvalue"], method="fdr_bh")[1]
    enr.to_csv(OUT / "cryptic_residue_enrichment.csv", index=False)
    sig = enr[enr["qvalue"] < 0.05].copy()
    print(f"FDR-significant resistance residues: {len(sig)}")

    who21 = set()
    if WHO21.exists():
        w = pd.read_csv(WHO21)
        who21 = set(zip(w.gene, w.residue_pos.astype(int)))

    feat = pd.read_csv(FEAT)
    feat_keys = set(zip(feat.gene, feat.residue_pos.astype(int)))

    train_pos = sorted(
        {
            (g, p)
            for g, p in who21
            if (g, p) in feat_keys and first_seen.get((g, p), cutoff + 1) < cutoff
        }
    )
    # Fallback: all WHO21 in feature table if early filter too strict
    if len(train_pos) < 40:
        train_pos = sorted([(g, p) for g, p in who21 if (g, p) in feat_keys])

    test_pos = sorted(
        {
            (r.gene, int(r.residue_pos))
            for r in sig.itertuples()
            if r.first_seen >= cutoff
            and (r.gene, int(r.residue_pos)) not in who21
            and (r.gene, int(r.residue_pos)) in feat_keys
        }
    )
    print(f"Train={len(train_pos)}  Test(late+FDR)={len(test_pos)}")
    print("Test by gene:", pd.Series([g for g, _ in test_pos]).value_counts().to_dict())

    if len(test_pos) < 5:
        # loosen: late-first-seen with raw p<0.01 and OR>2
        loose = enr[(enr.pvalue < 0.01) & (enr.odds_ratio > 2) & (enr.first_seen >= cutoff)]
        test_pos = sorted(
            {
                (r.gene, int(r.residue_pos))
                for r in loose.itertuples()
                if (r.gene, int(r.residue_pos)) not in who21
                and (r.gene, int(r.residue_pos)) in feat_keys
            }
        )
        print(f"Loosened test={len(test_pos)}", pd.Series([g for g, _ in test_pos]).value_counts().to_dict())

    # Features
    df = feat.copy()
    dists = []
    for _, row in df.iterrows():
        core = CORE_BINDING.get(row["gene"], set())
        dists.append(500 if not core else min(abs(int(row["residue_pos"]) - p) for p in core))
    df["inner_distance"] = dists
    for c in FEATURE_COLS:
        df[c] = df.groupby("gene")[c].transform(lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-8))

    train_set, test_set = set(train_pos), set(test_pos)
    y_tr = df.apply(lambda r: (r.gene, int(r.residue_pos)) in train_set, axis=1).astype(int).values
    y_te = df.apply(lambda r: (r.gene, int(r.residue_pos)) in test_set, axis=1).astype(int).values
    print(f"Matrix train={y_tr.sum()} test={y_te.sum()}")

    X = StandardScaler().fit_transform(df[FEATURE_COLS].fillna(0).values)
    clf = train_xgb(X, y_tr)
    scores = clf.predict_proba(X)[:, 1]
    df["score"] = scores
    pool = ~y_tr.astype(bool)
    y_pool, s_pool = y_te[pool], scores[pool]
    auprc = float(average_precision_score(y_pool, s_pool)) if y_pool.sum() else float("nan")
    auroc = float(roc_auc_score(y_pool, s_pool)) if y_pool.sum() and (1 - y_pool).sum() else float("nan")
    top100 = int(y_pool[np.argsort(-s_pool)[:100]].sum()) if y_pool.sum() else 0

    # Within-gene for genes with >=3 test positives
    wg_rows = []
    df["within_gene_rank"] = df.groupby("gene")["score"].rank(ascending=False, method="min")
    for gene, gdf in df.groupby("gene"):
        y = gdf.apply(lambda r: 1 if (r.gene, int(r.residue_pos)) in test_set else 0, axis=1).values
        if y.sum() < 2:
            continue
        s = gdf["score"].values
        wg_rows.append(
            {
                "gene": gene,
                "n_test": int(y.sum()),
                "auroc": float(roc_auc_score(y, s)),
                "auprc": float(average_precision_score(y, s)),
                "random": float(y.mean()),
            }
        )
    wg = pd.DataFrame(wg_rows)
    wg.to_csv(OUT / "cryptic_temporal_within_gene.csv", index=False)

    df.loc[pool, "pool_rank"] = pd.Series(scores[pool]).rank(ascending=False, method="min").values
    rank_rows = []
    for g, p in test_pos:
        r = df[(df.gene == g) & (df.residue_pos == p)].iloc[0]
        rank_rows.append(
            {
                "gene": g,
                "residue_pos": p,
                "first_seen": first_seen[(g, p)],
                "pool_rank": int(r["pool_rank"]),
                "within_gene_rank": int(r["within_gene_rank"]),
                "score": float(r["score"]),
            }
        )
    ranks = pd.DataFrame(rank_rows).sort_values("pool_rank")
    ranks.to_csv(OUT / "cryptic_temporal_test_ranks.csv", index=False)
    pd.DataFrame(train_pos, columns=["gene", "residue_pos"]).to_csv(
        OUT / "cryptic_temporal_train.csv", index=False
    )
    pd.DataFrame(test_pos, columns=["gene", "residue_pos"]).to_csv(
        OUT / "cryptic_temporal_test.csv", index=False
    )

    summary = {
        "n_isolates": int(len(pheno)),
        "cutoff_index": cutoff,
        "train_n": int(y_tr.sum()),
        "test_n": int(y_te.sum()),
        "test_by_gene": pd.Series([g for g, _ in test_pos]).value_counts().to_dict(),
        "test_non_pnca": int(sum(g != "pncA" for g, _ in test_pos)),
        "pool_auprc": auprc,
        "pool_auroc": auroc,
        "pool_top100": top100,
        "within_gene": wg.to_dict(orient="records"),
        "top20_test": ranks.head(20).to_dict(orient="records"),
        "note": "Phenotype order proxy; FDR q<0.05 resistance enrichment required for test positives.",
    }
    (OUT / "cryptic_temporal_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

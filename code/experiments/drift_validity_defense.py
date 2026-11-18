"""
Robustness defense of the PUBLISHED drift score (answers R3 W1/Q1) — no method change.

Two tests on the paper's own per-platform drift scores (results/*_ngram_analysis/*.jsonl):

1. CONVERGENT VALIDITY: terms in the Mendelsohn et al. glossary (known dog whistles,
   independent external ground truth) should score higher on the published drift score
   than the rest of each platform's scored vocabulary. Mann-Whitney U + AUC per platform
   and pooled. (AUC = P(random glossary term outranks random other term).)

2. CROSS-PLATFORM REPLICATION: the five platform models were trained INDEPENDENTLY.
   If drift were mostly coordinate noise, a term in one platform's top decile would land
   in another platform's top decile at ~chance (10%). Above-chance replication = the
   independently-trained models agree on which terms drifted -> shared signal, not noise.
"""
import json, re
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.stats import mannwhitneyu

ROOT = Path(".")
FILES = {
    "truth":   ROOT/"results/platform_ngram_analysis/drift_truth_social.jsonl",
    "bluesky": ROOT/"results/platform_ngram_analysis/drift_bluesky.jsonl",
    "mastodon":ROOT/"results/platform_ngram_analysis/drift_mastodon.jsonl",
    "reddit":  ROOT/"results/tr_platform_ngram_analysis/semantic_drift_reddit.jsonl",
    "twitter": ROOT/"results/tr_platform_ngram_analysis/semantic_drift_twitter.jsonl",
}

def load_scores(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("data_type") == "drift_score" and "term" in d:
                rows.append((str(d["term"]).lower().strip(), float(d["drift_score"])))
    df = pd.DataFrame(rows, columns=["term", "drift"]).drop_duplicates("term")
    return df

# glossary surface forms (single tokens + phrases; we match tokens present in vocab)
g = pd.read_csv(ROOT/"data/allen_ai/glossary.tsv", sep="\t", engine="python", on_bad_lines="skip")
surface = set()
for _, r in g.iterrows():
    for col in ["Dogwhistle", "Surface Forms"]:
        v = str(r.get(col, ""))
        for s in v.split(";"):
            s = s.strip().lower().lstrip("#")
            if s and s != "nan":
                surface.add(s)
                surface.add(s.replace(" ", "_"))   # pipeline ngram form
print(f"Glossary surface forms: {len(surface)}")

# ---------------------------------------------------------------- test 1
print("\n=== 1. Convergent validity (published drift, per platform) ===")
res, platform_scores = [], {}
for plat, f in FILES.items():
    df = load_scores(f)
    platform_scores[plat] = df
    df["is_glossary"] = df["term"].isin(surface)
    gl = df[df.is_glossary]["drift"].to_numpy()
    ot = df[~df.is_glossary]["drift"].to_numpy()
    if len(gl) < 5:
        print(f"{plat}: only {len(gl)} glossary terms in vocab, skipping")
        continue
    u, p = mannwhitneyu(gl, ot, alternative="greater")
    auc = u / (len(gl) * len(ot))
    res.append({"platform": plat, "n_glossary": len(gl), "n_other": len(ot),
                "auc": round(auc, 3), "p": p,
                "gl_median": round(float(np.median(gl)), 4),
                "other_median": round(float(np.median(ot)), 4)})
    print(f"{plat:9s} glossary n={len(gl):4d} vs other n={len(ot):6d} | "
          f"AUC={auc:.3f} p={p:.2e} | medians {np.median(gl):.3f} vs {np.median(ot):.3f}")
pd.DataFrame(res).to_csv(ROOT/"experiments/procrustes/drift_convergent_validity.csv", index=False)

# ---------------------------------------------------------------- test 2
print("\n=== 2. Cross-platform replication of top-decile flags ===")
deciles = {}
for plat, df in platform_scores.items():
    thr = df["drift"].quantile(0.9)
    deciles[plat] = set(df[df["drift"] >= thr]["term"])
rows = []
for a, b in combinations(platform_scores, 2):
    va = set(platform_scores[a]["term"]); vb = set(platform_scores[b]["term"])
    shared = va & vb
    if len(shared) < 100:
        continue
    ta = deciles[a] & shared; tb = deciles[b] & shared
    if not ta or not tb:
        continue
    overlap = len(ta & tb) / len(ta)
    expected = len(tb) / len(shared)     # chance rate
    lift = overlap / expected if expected else float("nan")
    rows.append({"pair": f"{a}-{b}", "shared_vocab": len(shared),
                 "P(top_b | top_a)": round(overlap, 3),
                 "chance": round(expected, 3), "lift": round(lift, 2)})
    print(f"{a:9s}->{b:9s} shared={len(shared):6d}  P(top|top)={overlap:.3f}  "
          f"chance={expected:.3f}  lift={lift:.2f}x")
pd.DataFrame(rows).to_csv(ROOT/"experiments/procrustes/drift_crossplatform_replication.csv", index=False)
print("\nWrote drift_convergent_validity.csv + drift_crossplatform_replication.csv")

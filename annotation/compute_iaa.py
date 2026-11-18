"""
Score the inter-coder annotations against the hidden key.

Expects coder_*.csv files (from annotation_sheet.csv, filled in) in this folder.
Reports:
  - inter-coder agreement: Cohen/Fleiss kappa on binary signal; Krippendorff alpha (ordinal) on likelihood
  - human (majority) vs LLM agreement  -> de-circularization evidence
  - precision of LLM-HIGH and LLM-MEDIUM buckets under blind human labels
  - false negatives: human-confirmed signal terms among LLM NOT_FLAGGED / mid-low drift
Run:  uv run python compute_iaa.py
"""
import glob
import itertools
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
key = pd.read_csv(HERE / "private" / "annotation_key.csv").set_index("item_id")

# --- blinding sanity check ---
# NOT_FLAGGED is a legacy team label treated as NONE (confirmed by Mao, 2026-07-12).
# Still warn if a coder's labels track the hidden key suspiciously closely.
def check_contamination(name, d):
    lik = d["likelihood"].astype(str).str.strip().str.upper().replace({"NOT_FLAGGED": "NONE"})
    exact = (lik.reindex(key.index) == key["llm_likelihood"].astype(str).str.upper()).mean()
    if exact > 0.6:
        print(f"WARNING coder '{name}': {exact:.0%} exact match with hidden key labels — "
              f"verify blinding before trusting.")

coder_files = sorted(glob.glob(str(HERE / "coder_*.csv")))
if len(coder_files) < 2:
    raise SystemExit(f"Need >=2 coder_*.csv files; found {len(coder_files)}. "
                     "Coders fill annotation_sheet.csv and save as coder_<name>.csv.")

coders = {}
for f in coder_files:
    d = pd.read_csv(f).set_index("item_id")
    name = Path(f).stem.replace("coder_", "")
    check_contamination(name, d)
    d["signal_bin"] = d["is_signal_language"].str.strip().str.lower().map(
        {"yes": 1, "no": 0, "cant_tell": np.nan})
    d["lik_ord"] = d["likelihood"].str.strip().str.upper().replace(
        {"NOT_FLAGGED": "NONE"}).map(
        {"NONE": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3})
    coders[name] = d
names = list(coders)
print(f"Coders: {names}  |  items: {len(key)}")


def cohen_kappa(a, b):
    m = ~(a.isna() | b.isna())
    a, b = a[m].astype(int), b[m].astype(int)
    n = len(a)
    po = (a == b).mean()
    cats = sorted(set(a) | set(b))
    pe = sum((np.mean(a == c)) * (np.mean(b == c)) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, n


def krippendorff_ordinal(mat):
    # mat: rows=items, cols=coders, values ordinal ints or NaN
    # requires the `krippendorff` package (uv run --with krippendorff);
    # a hand-rolled version here previously produced wrong (negative) values.
    import krippendorff
    return krippendorff.alpha(reliability_data=mat.values.T.astype(float),
                              level_of_measurement="ordinal")


# ---- inter-coder: binary signal (Cohen if 2, else pairwise mean) ----
print("\n=== inter-coder agreement (binary is_signal) ===")
pair_k = []
for a, b in itertools.combinations(names, 2):
    k, n = cohen_kappa(coders[a]["signal_bin"], coders[b]["signal_bin"])
    pair_k.append(k)
    print(f"  Cohen kappa {a} vs {b}: {k:.3f}  (n={n})")
print(f"  mean pairwise kappa: {np.mean(pair_k):.3f}")

lik = pd.DataFrame({nm: coders[nm]["lik_ord"] for nm in names})
print(f"\n=== inter-coder agreement (ordinal likelihood) ===")
print(f"  Krippendorff alpha (ordinal): {krippendorff_ordinal(lik):.3f}")

# ---- human majority vs LLM ----
sig = pd.DataFrame({nm: coders[nm]["signal_bin"] for nm in names})
human_signal = (sig.mean(axis=1) >= 0.5).astype(int)   # majority yes
llm_hi = key["llm_likelihood"].isin(["HIGH"]).astype(int)
llm_flagged = key["llm_likelihood"].isin(["HIGH", "MEDIUM"]).astype(int)
j = pd.DataFrame({"human": human_signal, "llm_flagged": llm_flagged,
                  "llm": key["llm_likelihood"], "bucket": key["drift_bucket"]}).dropna()
print("\n=== human (majority) vs LLM ===")
print(f"  agreement (human-signal == llm-flagged): {(j.human == j.llm_flagged).mean():.3f}")
kf, nf = cohen_kappa(j.human, j.llm_flagged)
print(f"  Cohen kappa (human vs llm-flagged): {kf:.3f} (n={nf})")

print("\n=== precision of LLM buckets under blind human labels ===")
for lab in ["HIGH", "MEDIUM"]:
    sub = j[j.llm == lab]
    if len(sub):
        print(f"  LLM {lab}: humans confirm signal {sub.human.mean():.1%}  (n={len(sub)})")

print("\n=== false negatives: human-signal among LLM NOT_FLAGGED ===")
nfsub = j[j.llm == "NOT_FLAGGED"]
print(f"  humans flag {int(nfsub.human.sum())}/{len(nfsub)} not-flagged terms as signal")
for b in ["high", "mid", "low"]:
    bb = nfsub[nfsub.bucket == b]
    if len(bb):
        print(f"    drift={b}: {int(bb.human.sum())}/{len(bb)} recovered")

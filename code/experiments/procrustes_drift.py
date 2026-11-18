"""
Procrustes-aligned drift ablation  (answers R3 / 2N2W W1 & Q1).

The paper's drift = 1 - cos(domain_vec, general_vec) between two SEPARATELY trained
FastText spaces (src/fasttext_drift.py:compute_semantic_divergence). Reviewer W1: per
Hamilton et al. (2016), separately trained spaces must be orthogonally aligned first,
else raw cosine mixes true semantic change with arbitrary coordinate mismatch.

This script re-computes drift after orthogonal Procrustes alignment on a large set of
frequent shared (assumed-stable) words, and compares:
  - Spearman rho between unaligned vs aligned drift over the candidate terms
  - how the 6 focal terms and the HIGH candidate ordering move
  - sanity: aligned drift on the anchor words should collapse toward 0
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from gensim.models import FastText
from gensim.models import KeyedVectors

ROOT = Path(".")
OUT = ROOT / "experiments" / "procrustes"
OUT.mkdir(parents=True, exist_ok=True)
N_ANCHOR = 40000     # frequent shared words used to fit the rotation
FOCAL = ["adolph", "banker", "fairy", "caliphate", "frog", "maga"]


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-9, None)


print("Loading domain FastText (raw, from-scratch)...")
dom = FastText.load(str(ROOT / "models/domain_fasttext_raw.model")).wv
print(f"  domain vocab {len(dom.index_to_key):,}, dim {dom.vector_size}")

print("Loading general baseline (wiki.en.vec)... [slow, ~minutes]")
gen = KeyedVectors.load_word2vec_format(str(ROOT / "models/pretrained/wiki.en.vec"), binary=False)
print(f"  general vocab {len(gen.index_to_key):,}, dim {gen.vector_size}")
gen_set = set(gen.key_to_index)

# ---- anchors: frequent domain words that exist in general ----
anchors = [w for w in dom.index_to_key[:200000] if w in gen_set][:N_ANCHOR]
D = unit(np.vstack([dom[w] for w in anchors]))
G = unit(np.vstack([gen[w] for w in anchors]))
R, _ = orthogonal_procrustes(D, G)
print(f"Fitted rotation on {len(anchors):,} shared frequent words.")


def drift_pair(word):
    if word not in gen_set:
        return None
    d = unit(dom[word].reshape(1, -1))[0]
    g = unit(gen[word].reshape(1, -1))[0]
    un = 1 - float(d @ g)
    al = 1 - float((d @ R) @ g)
    return un, al


# sanity on anchors
an_un = np.array([1 - float(d @ g) for d, g in zip(D, G)])
an_al = np.array([1 - float(d @ g) for d, g in zip(D @ R, G)])
print(f"\nAnchor-word drift: unaligned mean {an_un.mean():.3f} -> aligned mean {an_al.mean():.3f} "
      f"(should drop sharply if alignment works)")

# ---- candidate terms (the 492 vetted + focal) ----
lab = pd.concat([pd.read_csv(ROOT / "notebooks/labeled_dog_whistle.csv"),
                 pd.read_csv(ROOT / "notebooks/tr_labeled_dog_whistle.csv")], ignore_index=True)
lab["term"] = lab["term"].astype(str).str.lower().str.strip()
cand = lab.drop_duplicates("term")[["term", "likelihood", "drift_score"]]

rows = []
for _, r in cand.iterrows():
    dp = drift_pair(r["term"])
    if dp:
        rows.append({"term": r["term"], "likelihood": r["likelihood"],
                     "paper_drift": r["drift_score"], "unaligned": dp[0], "aligned": dp[1]})
res = pd.DataFrame(rows)
res.to_csv(OUT / "candidate_drift_aligned_vs_unaligned.csv", index=False)

rho, p = spearmanr(res["unaligned"], res["aligned"])
print(f"\n=== candidate terms ({len(res)} in both vocabs) ===")
print(f"Spearman rho (unaligned vs aligned drift): {rho:.3f}  (p={p:.2e})")
print(f"unaligned drift mean {res.unaligned.mean():.3f} | aligned mean {res.aligned.mean():.3f}")

# ranking overlap of top-K
for K in (50, 94):
    top_un = set(res.sort_values("unaligned", ascending=False).head(K)["term"])
    top_al = set(res.sort_values("aligned", ascending=False).head(K)["term"])
    print(f"  top-{K} overlap: {len(top_un & top_al)}/{K}")

print("\n=== focal terms ===")
foc = []
for t in FOCAL:
    dp = drift_pair(t)
    foc.append({"term": t, "unaligned": None if not dp else round(dp[0], 4),
                "aligned": None if not dp else round(dp[1], 4),
                "in_general_vocab": t in gen_set})
fdf = pd.DataFrame(foc)
print(fdf.to_string(index=False))
fdf.to_csv(OUT / "focal_drift.csv", index=False)
print(f"\nWrote {OUT}/candidate_drift_aligned_vs_unaligned.csv and focal_drift.csv")

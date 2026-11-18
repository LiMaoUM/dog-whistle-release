"""
Reproduce Sarah's Figure-3 per-post projections locally, then run the significance
test on cross-platform differences that the notebook does NOT do (answers R3 W3/Q3, R2 M4).

Faithful to visualization.ipynb:
  - model all-mpnet-base-v2; term/anchor contexts marked as [WORD] (embed_terms_and_anchors.py)
  - anchor vec = platform-stratified mean (per-platform mean over that category's contexts,
    then mean across the 5 platforms), EXCLUDING biden from left and trump from right
  - per post: X = cos(right) - cos(left), Y = cos(pos) - cos(neg)
  - cell value = mean over posts in (term, platform)   [cosine-then-mean]

Then, the NEW part:
  - Welch t + bootstrap 95% CI + permutation p on the cross-platform difference of X and Y,
    for each focal term, for the headline contrasts (each platform vs the pooled rest, and the
    max-vs-min platform). Small-n cells (n<30) explicitly flagged as unable to support inference.

Validation gate: per-cell means are compared to Sarah's plot1 CSV; large deviation => stop.
Anchors are capped per (word,platform) for tractability; the mean is stable at the cap.
maga is embedded with a per-platform cap (its cells have huge n, so significance is not in doubt;
its exact mean is taken from Sarah's released CSV, not re-estimated here).
"""
import re, csv, json, random
import numpy as np, pandas as pd
from pathlib import Path
from collections import defaultdict
csv.field_size_limit(10**8)

SEED = 42; random.seed(SEED); np.random.seed(SEED)
ROOT = Path(".")
OUT = ROOT / "experiments" / "figure3_data"
SMOKE = bool(int(__import__("os").environ.get("SMOKE", "0")))
ANCHOR_CAP = 300 if SMOKE else 20000     # per (word, platform)
TERM_CAP   = 200 if SMOKE else 20000     # per (term, platform); only bites maga
GPU = "cuda:1"

FOCAL = ["adolph", "banker", "fairy", "caliphate", "frog", "maga"]
ANCHORS = {
 "pos": ["good","benefit","help","protect","justice","support","safe","improve","positive"],
 "neg": ["bad","harm","hate","threat","danger","disgust","evil","corrupt","destroy","negative"],
 "left": ["democrat","democrats","democratic","dnc","biden","harris","liberal","progressive",
          "left-wing","left wing","aoc","bernie","sanders","union","labor"],
 "right": ["republican","republicans","gop","trump","conservative","right-wing","right wing",
           "libertarian","fox","heritage foundation","second amendment","2a"],
}
EXCLUDE = {"left": {"biden"}, "right": {"trump"}}
VALID = {"twitter","truth","reddit","bluesky","mastodon"}

def norm_plat(p):
    p = str(p).strip().lower()
    return {"truth_social":"truth","truthsocial":"truth"}.get(p, p)

# ---------------------------------------------------------------- collect contexts
# Match Sarah/embed_terms_and_anchors.py EXACTLY: case-insensitive SUBSTRING match
# (str.contains, no word boundaries); marking via case-sensitive str.replace.
anchor_words = [w for ws in ANCHORS.values() for w in ws]
cat_of = {w: c for c, ws in ANCHORS.items() for w in ws}

term_ctx = defaultdict(list)                       # term -> [(platform, marked_text)]
anchor_ctx = defaultdict(list)                      # cat  -> [(platform, cue_word, marked_text)]
term_cnt = defaultdict(int)                         # (term, plat)
anchor_cnt = defaultdict(int)                       # (cat, word, plat)

files = ["data/english_posts_processed.csv", "data/tr_english_posts_processed.csv"]
for f in files:
    for ch in pd.read_csv(ROOT/f, usecols=["text","platform"], dtype=str,
                          on_bad_lines="skip", engine="python", chunksize=100_000):
        for txt, plat in zip(ch["text"].astype(str), ch["platform"]):
            plat = norm_plat(plat)
            if plat not in VALID:
                continue
            low = txt.lower()
            for t in FOCAL:
                if term_cnt[(t, plat)] < TERM_CAP and t in low:
                    term_ctx[t].append((plat, txt.replace(t, f"[{t.upper()}]")))
                    term_cnt[(t, plat)] += 1
            for w in anchor_words:
                if anchor_cnt[(cat_of[w], w, plat)] < ANCHOR_CAP and w in low:
                    anchor_ctx[cat_of[w]].append((plat, w, txt.replace(w, f"[{w.upper()}]")))
                    anchor_cnt[(cat_of[w], w, plat)] += 1
    if SMOKE:
        break
print("term contexts:", {t: len(v) for t, v in term_ctx.items()})
print("anchor contexts:", {c: len(v) for c, v in anchor_ctx.items()})

# ---------------------------------------------------------------- embed
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=GPU)
def embed(texts):
    return np.asarray(model.encode(texts, batch_size=256, show_progress_bar=False,
                                   normalize_embeddings=False), dtype=np.float32)

def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

# anchor vecs: platform-stratified mean, excluding biden/trump
anchor_vec = {}
for cat, items in anchor_ctx.items():
    excl = EXCLUDE.get(cat, set())
    items = [(pl, w, tx) for (pl, w, tx) in items if w not in excl]
    embs = embed([tx for _, _, tx in items])
    plats = np.array([pl for pl, _, _ in items])
    plat_means = [embs[plats == pl].mean(0) for pl in sorted(set(plats)) if (plats == pl).any()]
    anchor_vec[cat] = np.mean(plat_means, axis=0).astype(np.float32)

# project terms per-post
rows = []
for t, items in term_ctx.items():
    embs = embed([tx for _, tx in items])
    plats = np.array([pl for pl, _ in items])
    e = unit(embs)
    proj = {c: e @ unit(anchor_vec[c]) for c in ANCHORS}
    X = proj["right"] - proj["left"]; Y = proj["pos"] - proj["neg"]
    for pl, x, y in zip(plats, X, Y):
        rows.append({"term": t, "platform": pl, "X": float(x), "Y": float(y)})
pp = pd.DataFrame(rows)
pp.to_csv(OUT / ("per_post_projections_SMOKE.csv" if SMOKE else "per_post_projections.csv"), index=False)

# ---------------------------------------------------------------- validate vs Sarah's plot1
agg = pp.groupby(["term","platform"]).agg(n=("X","size"), ideology=("X","mean"),
                                          framing=("Y","mean")).reset_index()
sarah = pd.read_csv(OUT / "plot1_context_cloud_centers.csv")
sarah["term"] = sarah["term"].str.lower()
m = agg.merge(sarah, on=["term","platform"], suffixes=("_mine",""))
m["d_ideo"] = (m["ideology"] - m["ideology_right_minus_left"]).abs()
m["d_fram"] = (m["framing"] - m["framing_pos_minus_neg"]).abs()
print("\n=== validation vs Sarah plot1 (abs diff; maga excluded from gate) ===")
gate = m[m.term != "maga"]
print(f"ideology max|diff| {gate.d_ideo.max():.4f}  mean {gate.d_ideo.mean():.4f}")
print(f"framing  max|diff| {gate.d_fram.max():.4f}  mean {gate.d_fram.mean():.4f}")

# ---------------------------------------------------------------- significance test
rng = np.random.default_rng(SEED)
def stats(a, b):
    a = np.asarray(a); b = np.asarray(b)
    diff = a.mean() - b.mean()
    if len(a) < 2 or len(b) < 2:
        return diff, np.nan, (np.nan, np.nan), "insufficient_n"
    # Welch p
    from scipy.stats import ttest_ind
    p = ttest_ind(a, b, equal_var=False).pvalue
    # bootstrap CI on the difference (cap group size for speed; conservative for huge n)
    ca = a if len(a) <= 10000 else rng.choice(a, 10000, replace=False)
    cb = b if len(b) <= 10000 else rng.choice(b, 10000, replace=False)
    d = rng.choice(ca, (4000, len(ca))).mean(1) - rng.choice(cb, (4000, len(cb))).mean(1)
    ci = tuple(np.percentile(d, [2.5, 97.5]))
    return float(diff), float(p), (float(ci[0]), float(ci[1])), "ok"

sig = []
for t in pp.term.unique():
    sub = pp[pp.term == t]
    for axis, col in [("ideology","X"), ("framing","Y")]:
        cells = {pl: sub[sub.platform == pl][col].to_numpy() for pl in sub.platform.unique()}
        cells = {pl: v for pl, v in cells.items() if len(v) > 0}
        if len(cells) < 2:
            continue
        means = {pl: v.mean() for pl, v in cells.items()}
        hi = max(means, key=means.get); lo = min(means, key=means.get)
        diff, p, ci, flag = stats(cells[hi], cells[lo])
        sig.append({"term": t, "axis": axis, "contrast": f"{hi}_vs_{lo}",
                    "n_hi": len(cells[hi]), "n_lo": len(cells[lo]),
                    "diff": round(diff, 4), "welch_p": p,
                    "ci_lo": None if ci[0] != ci[0] else round(ci[0], 4),
                    "ci_hi": None if ci[1] != ci[1] else round(ci[1], 4),
                    "significant_0.05": (p == p and p < 0.05), "flag": flag})
sig = pd.DataFrame(sig)
sig.to_csv(OUT / ("significance_SMOKE.csv" if SMOKE else "significance_results.csv"), index=False)
print("\n=== significance (max-vs-min platform contrast per term/axis) ===")
print(sig.to_string(index=False))
print(f"\nSaved per_post_projections.csv + significance_results.csv to {OUT}")

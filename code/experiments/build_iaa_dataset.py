"""
Build a blinded inter-coder annotation dataset for the dog-whistle paper rebuttal.

Answers R3 (2N2W) W4 and R2 (MJPh) M3: human ground truth produced INDEPENDENTLY
of the LLM, with inter-annotator agreement.

Design:
  - Universe: terms_3k.csv (15,000 drift-ranked terms; drift bucket high/mid/low).
  - LLM label: HIGH/MEDIUM from the 492 vetted terms; everything else NOT_FLAGGED.
  - Sample ~150 terms across (LLM label x drift bucket) so we can report:
      * inter-coder kappa (reliability)
      * human vs LLM agreement (de-circularization / independence)
      * precision of LLM-HIGH and LLM-MEDIUM
      * false negatives: human-confirmed signal terms the LLM did NOT flag
  - Contexts sampled UNIFORMLY for every term (single pass, up to 8) so flagged and
    unflagged terms look identical to coders.
  - Coders see term + contexts ONLY; drift/bucket/platform/LLM label held in a key.
"""
import re
import csv
import random
import pandas as pd
from pathlib import Path

csv.field_size_limit(int(1e8))

SEED = 42
random.seed(SEED)
ROOT = Path(".")
OUT = ROOT / "experiments" / "iaa"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = {"HIGH": 45, "MEDIUM": 45, "NF_high": 20, "NF_mid": 20, "NF_low": 20}
OVERSAMPLE = 2.2      # select extra, trim after context check
MAX_CTX, MIN_CTX, POOL = 8, 3, 40

# ---------------------------------------------------------------- labels + universe
labeled = pd.concat([
    pd.read_csv(ROOT / "notebooks/labeled_dog_whistle.csv"),
    pd.read_csv(ROOT / "notebooks/tr_labeled_dog_whistle.csv"),
], ignore_index=True).dropna(subset=["term"])
labeled["term"] = labeled["term"].astype(str).str.strip().str.lower()
labeled = labeled.drop_duplicates(subset=["term"])
llm_map = dict(zip(labeled["term"], labeled["likelihood"]))
plat_map = dict(zip(labeled["term"], labeled["platform"]))
tgt_map = dict(zip(labeled["term"], labeled["target_group"]))

universe = pd.read_csv(ROOT / "notebooks/terms_3k.csv")
universe["term"] = universe["term"].astype(str).str.strip().str.lower()
universe = universe.drop_duplicates(subset=["term"])
drift_map = dict(zip(universe["term"], universe["drift_score"]))
flagged = set(llm_map)

# keep single-token terms only (context sampling uses token-set membership)
def one_tok(t):
    return bool(re.fullmatch(r"[a-z0-9][a-z0-9''\-]*", t))

def pick(pool, n):
    pool = [t for t in dict.fromkeys(pool) if t and one_tok(t)]
    random.shuffle(pool)
    return pool[:n]

high_terms = [t for t in llm_map if llm_map[t] == "HIGH"]
med_terms = [t for t in llm_map if llm_map[t] == "MEDIUM"]
notflag = universe[~universe["term"].isin(flagged)]
nf = {b: notflag[notflag["bucket"] == b]["term"].tolist() for b in ["high", "mid", "low"]}

cand = {}   # term -> (llm_label, drift_bucket)
for t in pick(high_terms, int(TARGET["HIGH"] * OVERSAMPLE)):
    cand[t] = ("HIGH", "high")
for t in pick(med_terms, int(TARGET["MEDIUM"] * OVERSAMPLE)):
    cand[t] = ("MEDIUM", "high")
for b in ["high", "mid", "low"]:
    for t in pick(nf[b], int(TARGET[f"NF_{b}"] * OVERSAMPLE)):
        cand[t] = ("NOT_FLAGGED", b)
targets = set(cand)
print(f"Candidate terms to seek contexts for: {len(targets)}")

# ---------------------------------------------------------------- single-pass context sampling
print("Loading corpus + sampling contexts (single pass)...")
tok_re = re.compile(r"[a-z0-9''\-]+")
pools = {t: [] for t in targets}
need = set(targets)
for f in ["data/english_posts_processed.csv", "data/tr_english_posts_processed.csv"]:
    for chunk in pd.read_csv(ROOT / f, usecols=["text"], dtype={"text": "object"},
                             on_bad_lines="skip", engine="python", chunksize=100_000):
        for t in chunk["text"].astype(str):
            low = t.lower()
            toks = set(tok_re.findall(low))
            hit = toks & need
            if not hit:
                continue
            snip = " ".join(t.split())
            if not (20 <= len(snip) <= 600):
                continue
            for term in hit:
                p = pools[term]
                if len(p) < POOL:
                    p.append(snip)
            need = {term for term in need if len(pools[term]) < POOL}
        if not need:
            break
    if not need:
        break

# ---------------------------------------------------------------- assemble, trim to target
by_cell = {k: [] for k in TARGET}
for term, (llm, bucket) in cand.items():
    ctx = pools[term]
    if len(ctx) < MIN_CTX:
        continue
    random.shuffle(ctx)
    cell = llm if llm != "NOT_FLAGGED" else f"NF_{bucket}"
    by_cell[cell].append((term, llm, bucket, ctx[:MAX_CTX]))

rows = []
for cell, want in TARGET.items():
    items = by_cell[cell]
    random.shuffle(items)
    rows.extend(items[:want])
random.shuffle(rows)
print(f"Final items: {len(rows)}  (per-cell available: "
      + ", ".join(f'{k}:{len(by_cell[k])}' for k in TARGET) + ")")

# ---------------------------------------------------------------- write
sheet, key = [], []
for i, (term, llm, bucket, ctx) in enumerate(rows):
    iid = f"DW{i+1:03d}"
    ctx_cols = {f"context_{j+1}": (ctx[j] if j < len(ctx) else "") for j in range(MAX_CTX)}
    sheet.append({"item_id": iid, "term": term, **ctx_cols,
                  "is_signal_language": "", "likelihood": "",
                  "coded_meaning_or_target": "", "confidence_1to3": ""})
    key.append({"item_id": iid, "term": term, "llm_likelihood": llm,
                "drift_bucket": bucket, "drift_score": drift_map.get(term),
                "platform": plat_map.get(term, "various"),
                "target_group_llm": tgt_map.get(term, ""), "n_contexts": len(ctx)})

pd.DataFrame(sheet).to_csv(OUT / "annotation_sheet.csv", index=False)
kdf = pd.DataFrame(key)
kdf.to_csv(OUT / "annotation_key.csv", index=False)
print("\n=== composition (LLM label x drift bucket) ===")
print(pd.crosstab(kdf["llm_likelihood"], kdf["drift_bucket"], margins=True))
print(f"\nWrote {OUT}/annotation_sheet.csv  ({len(sheet)} items)")
print(f"Wrote {OUT}/annotation_key.csv")

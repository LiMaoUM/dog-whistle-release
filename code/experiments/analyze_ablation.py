"""Build labels.csv and summary.md from raw/ caches."""
import json, glob, csv, collections
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

import sys
OUT = Path(sys.argv[1] if len(sys.argv)>1 else "./experiments/ablation")
PIN, POUT = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv)>3 else (0.25, 2.00)
NB = Path("./notebooks")
PLATFORMS = ["bluesky", "mastodon", "truth_social", "reddit", "twitter"]
CONDS = ["with_score", "no_score", "with_score_rep2"]
DATA = Path("./results/drift_scores_csv")

df = pd.concat([pd.read_csv(DATA / f"{p}_unigram_top1000.csv").assign(platform=p) for p in PLATFORMS], ignore_index=True)
df["batch"] = (df["rank"] - 1) // 50  # rows are in rank order, ranks 1..1000
assert (df.groupby("platform")["rank"].apply(lambda s: list(s) == list(range(1, 1001)))).all()

labels = {c: {} for c in CONDS}
stats = {c: collections.Counter() for c in CONDS}
retried_list = []
unm_examples = collections.defaultdict(list)
usage = collections.Counter(); models = set(); finish = collections.Counter(); times = []
for fp in sorted(glob.glob(str(OUT / "raw/*.json"))):
    r = json.load(open(fp)); c, p, bi = r["condition"], r["platform"], r["batch_index"]
    if c not in CONDS: continue
    models.add(r["model_returned"]); finish[(c, r["finish_reason"])] += 1; times.append(r["timestamp"])
    for k in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
        usage[(c, k)] += r[k] or 0
    st = stats[c]; st["batches"] += 1
    if r.get("retried_with_16000"):
        st["retried"] += 1; retried_list.append(f"{c}/{p}/{bi}"); usage[(c,'prompt_tokens')] += r['first_attempt_usage']['prompt_tokens']; usage[(c,'completion_tokens')] += r['first_attempt_usage']['completion_tokens']
    inp = {t.lower(): t for t in r["input_terms"]}
    for ln in r["lines"]:
        if not ln["parsed"]:
            st["unparsed_lines"] += 1; continue
        st["returned_lines"] += 1
        lk = ln["likelihood"]
        if lk not in ("HIGH", "MEDIUM", "LOW"): st["odd_likelihood_value"] += 1
        if lk not in ("HIGH", "MEDIUM"): st["low_or_other"] += 1; continue
        key = ln["term"].lower()
        if key in inp:
            t = inp[key]
            if (p, t) in labels[c]:
                st["duplicate_returned"] += 1
                if lk == "HIGH": labels[c][(p, t)] = "HIGH"
            else:
                labels[c][(p, t)] = lk
        else:
            st["unmatched_HM_lines"] += 1
            if len(unm_examples[c]) < 8: unm_examples[c].append(f"{p}:{ln['term']}")
for c in CONDS:
    df[c] = [labels[c].get((p, t), "NONE") for p, t in zip(df.platform, df.term)]
df["batches_ok"] = True
out = df.rename(columns={"with_score": "label_with_score", "no_score": "label_no_score", "with_score_rep2": "label_with_score_rep2"})
out[["platform", "term", "drift_score", "rank", "label_with_score", "label_no_score", "label_with_score_rep2"]].to_csv(OUT / "labels.csv", index=False)

# published sets
pub = {}
for f, m in (("screening_results.json", {"bluesky": "bluesky", "mastodon": "mastodon", "truth": "truth_social"}),
             ("screening_results_tr.json", {"reddit": "reddit", "twitter": "twitter"})):
    d = json.load(open(NB / f))
    for k, v in d.items():
        pub[m[k]] = {x["term"].lower(): x["likelihood"] for x in v}
df["pub"] = [pub[p].get(t.lower(), "NONE") for p, t in zip(df.platform, df.term)]
pub_unmatched = {p: sum(1 for t in pub[p] if t not in set(df[df.platform == p].term.str.lower())) for p in PLATFORMS}

def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b); cats = sorted(set(a) | set(b))
    po = (a == b).mean(); pe = sum((a == k).mean() * (b == k).mean() for k in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")
def jac(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b) if a | b else float("nan")
def agree(sub, c1, c2):
    f1, f2 = sub[c1] != "NONE", sub[c2] != "NONE"
    h1, h2 = sub[c1] == "HIGH", sub[c2] == "HIGH"
    idx = sub.index
    return dict(pct_flag=100 * (f1 == f2).mean(), kappa_flag=kappa(f1, f2),
                pct_3way=100 * (sub[c1] == sub[c2]).mean(), kappa_3way=kappa(sub[c1], sub[c2]),
                jac_flag=jac(idx[f1], idx[f2]), jac_high=jac(idx[h1], idx[h2]))

groups = [(p, df[df.platform == p]) for p in PLATFORMS] + [("ALL", df)]
L = []
w = L.append
w("# Stage-1 prompt ablation: effect of showing the drift score\n")
w(f"Run date: {min(times)[:10]} to {max(times)[:10]}. Model requested: see returned id; model id returned by API: {', '.join(sorted(models))}.")
w("Terms: 5 platforms x 1000 unigrams, batches of 50 in rank order, 20 batches per platform per condition. Conditions: `with_score` (original prompt), `no_score` (term line without the drift score), `with_score_rep2` (second with-score run, noise floor).\n")
w("Label rule: HIGH or MEDIUM only if the model returned the term on a line with that exact likelihood string (original parser); LOW, missing, or unmatched is NONE.\n")

w("## Counts\n")
w("| platform | cond | HIGH | MEDIUM | flagged | flagged rate % |"); w("|---|---|---|---|---|---|")
for g, s in groups:
    for c in CONDS:
        h, m = (s[c] == "HIGH").sum(), (s[c] == "MEDIUM").sum()
        w(f"| {g} | {c} | {h} | {m} | {h+m} | {100*(h+m)/len(s):.1f} |")
w("")
w("## Agreement between conditions (per 1000 terms per platform; 5000 overall)\n")
w("pct = percent agreement, kappa = Cohen's kappa; flag = flagged vs not; 3way = HIGH/MEDIUM/NONE; Jaccard of flagged sets and of HIGH sets.\n")
w("| platform | pair | % agree flag | kappa flag | % agree 3-way | kappa 3-way | Jaccard flagged | Jaccard HIGH |"); w("|---|---|---|---|---|---|---|---|")
pairs = [("with_score", "no_score", "with vs no score"), ("with_score", "with_score_rep2", "with vs with rep2 (noise floor)"),
         ("with_score_rep2", "no_score", "rep2 vs no score")]
for g, s in groups:
    for c1, c2, nm in pairs:
        x = agree(s, c1, c2)
        w(f"| {g} | {nm} | {x['pct_flag']:.1f} | {x['kappa_flag']:.3f} | {x['pct_3way']:.1f} | {x['kappa_3way']:.3f} | {x['jac_flag']:.3f} | {x['jac_high']:.3f} |")
w("")
w("## Relation to drift (within each condition)\n")
w("Spearman rho between drift rank (1 = highest drift) and flagged (0/1); negative rho means flagged terms sit at higher drift. Mean drift score of flagged vs unflagged terms.\n")
w("| platform | cond | rho(rank, flagged) | p | mean drift flagged | mean drift unflagged | diff |"); w("|---|---|---|---|---|---|---|")
for g, s in groups:
    for c in CONDS:
        f = (s[c] != "NONE").astype(int)
        if f.nunique() > 1:
            rho, pv = spearmanr(s["rank"], f)
        else: rho = pv = float("nan")
        a, b = s.loc[f == 1, "drift_score"].mean(), s.loc[f == 0, "drift_score"].mean()
        w(f"| {g} | {c} | {rho:.3f} | {pv:.2g} | {a:.4f} | {b:.4f} | {a-b:+.4f} |")
w("")
w("## Overlap with the paper's published flagged set\n")
w(f"Published sets: notebooks/screening_results.json (bluesky, mastodon, truth) and screening_results_tr.json (reddit, twitter). Published terms absent from the top-1000 file: {pub_unmatched}. (The notebook call used `batch_size=100`; see notes.)\n")
w("| platform | cond | n published | n flagged | intersection | Jaccard | share of published recovered % | share of flagged that is published % |"); w("|---|---|---|---|---|---|---|---|")
for g, s in groups:
    P = set(s.index[s.pub != "NONE"])
    for c in CONDS:
        F = set(s.index[s[c] != "NONE"])
        i = len(P & F)
        w(f"| {g} | {c} | {len(P)} | {len(F)} | {i} | {jac(P,F):.3f} | {100*i/len(P):.1f} | {100*i/max(1,len(F)):.1f} |")
w("")
w("## Parsing and API diagnostics\n")
w("| cond | batches | returned lines (>=4 fields) | HIGH/MEDIUM lines unmatched to input | LOW/other lines | odd likelihood strings | unparseable lines | duplicate returned terms |"); w("|---|---|---|---|---|---|---|---|")
for c in CONDS:
    s = stats[c]
    w(f"| {c} | {s['batches']} | {s['returned_lines']} | {s['unmatched_HM_lines']} | {s['low_or_other']} | {s['odd_likelihood_value']} | {s['unparsed_lines']} | {s['duplicate_returned']} |")
w("")
for c in CONDS:
    if unm_examples[c]: w(f"Unmatched examples ({c}): {', '.join(unm_examples[c])}")
w("")
w(f"Batches retried with max_completion_tokens=16000 after an empty length-truncated first attempt: {len(retried_list)} ({'list in raw/*.json field retried_with_16000' if len(retried_list)>20 else (', '.join(retried_list) or 'none')}). First-attempt tokens of retried batches are included in the usage table.\n")
w("Finish reasons per condition (final attempt): " + "; ".join(f"{c}/{r}: {n}" for (c, r), n in sorted(finish.items())) + "\n")
tot_cost = 0
w("| cond | prompt tokens | completion tokens (incl. reasoning) | reasoning tokens | total | approx cost USD |"); w("|---|---|---|---|---|---|")
for c in CONDS:
    pt, ct, rt = usage[(c, "prompt_tokens")], usage[(c, "completion_tokens")], usage[(c, "reasoning_tokens")]
    cost = pt / 1e6 * PIN + ct / 1e6 * POUT; tot_cost += cost
    w(f"| {c} | {pt:,} | {ct:,} | {rt:,} | {pt+ct:,} | {cost:.2f} |")
tp = sum(usage[(c, 'prompt_tokens')] for c in CONDS); tc = sum(usage[(c, 'completion_tokens')] for c in CONDS)
w(f"| total | {tp:,} | {tc:,} | {sum(usage[(c,'reasoning_tokens')] for c in CONDS):,} | {tp+tc:,} | {tot_cost:.2f} |")
w(f"\nCost uses assumed list prices of ${PIN} per 1M input and ${POUT} per 1M output tokens (not read from the API). \n")
(OUT / "summary.md").write_text("\n".join(L), encoding="utf-8")
print("\n".join(L))

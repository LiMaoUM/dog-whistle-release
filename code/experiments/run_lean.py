"""Political-lean descriptive table: sample, classify, summarize.

Usage (from project root):
  uv run python experiments/lean/run_lean.py sample
  uv run python experiments/lean/run_lean.py classify [--limit N]   # --limit: smoke test, writes smoke_labels.csv
  uv run python experiments/lean/run_lean.py summarize
"""
import asyncio, csv, hashlib, json, math, os, random, sys, datetime
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "experiments/lean"
SEED = 20261113
N_PER = 2000
MODEL = "gpt-5-mini-2025-08-07"
SOURCES = {  # platform -> (file, expected Table 1 count)
    "twitter": (ROOT / "results/tr_english_posts_raw.csv", 4084887),
    "truth_social": (ROOT / "data/english_posts_raw.csv", 2233482),
    "reddit": (ROOT / "results/tr_english_posts_raw.csv", 1632879),
    "bluesky": (ROOT / "data/english_posts_raw.csv", 1078237),
    "mastodon": (ROOT / "data/english_posts_raw.csv", 58091),
}
SYSTEM = "You label the political leaning expressed by the author of a social media post about U.S. politics. Answer with one word."
USER = ('Post:\n"""\n{text}\n"""\n\nWhich political side does the author of this post express support for or alignment with, '
        'in the U.S. left-right sense? Answer LEFT if the post expresses a left-leaning, liberal, progressive or pro-Democratic '
        'position (including criticism of Trump or Republicans from the left). Answer RIGHT if it expresses a right-leaning, '
        'conservative or pro-Republican position (including criticism of Biden, Harris or Democrats from the right). '
        'Answer NEITHER if the post expresses no clear side, is neutral reporting, or cannot be judged. '
        'Answer with exactly one word: LEFT, RIGHT or NEITHER.')


def do_sample():
    """Pass 1: count rows per platform (all rows are English by file construction; verified en-only).
    Pass 2: draw seeded simple random sample of row positions per platform within its file, extract text."""
    rng = np.random.default_rng(SEED)
    files = {}
    for p, (f, _) in SOURCES.items():
        files.setdefault(f, []).append(p)
    rows = []
    counts = {}
    for f, plats in files.items():
        # pass 1: per-platform row counts and per-platform ordinal (the row id = ordinal within platform in this file)
        cnt = {p: 0 for p in plats}
        for ch in pd.read_csv(f, usecols=["platform", "detected_language"], chunksize=500000, dtype=str):
            ch = ch[ch.detected_language == "en"]
            for p in plats:
                cnt[p] += int((ch.platform == p).sum())
        counts.update(cnt)
        picks = {}
        for p in plats:
            picks[p] = set(rng.choice(cnt[p], size=N_PER, replace=False).tolist())
        seen = {p: 0 for p in plats}
        for ch in pd.read_csv(f, usecols=["text", "platform", "detected_language"], chunksize=500000, dtype=str):
            ch = ch[ch.detected_language == "en"]
            for p in plats:
                sub = ch[ch.platform == p]
                n = len(sub)
                local = [i for i in range(n) if (seen[p] + i) in picks[p]]
                for i in local:
                    rows.append((p, f"{p}-{seen[p] + i}", sub.text.iloc[i]))
                seen[p] += n
    df = pd.DataFrame(rows, columns=["platform", "row_id", "text"])
    order = {p: i for i, p in enumerate(SOURCES)}
    df = df.sort_values(["platform", "row_id"], key=lambda s: s.map(order) if s.name == "platform" else s.str.rsplit("-", n=1).str[1].astype(int))
    df.to_csv(OUT / "sample.csv", index=False)
    json.dump(counts, open(OUT / "population_counts.json", "w"), indent=1)
    print(counts, len(df), df.groupby("platform").size().to_dict())


def load_env_key():
    for line in open(".env"):
        line = line.strip()
        if line.startswith("OPENAI_API_KEY"):
            v = line.split("=", 1)[1].strip().strip('"').strip("'")
            os.environ["OPENAI_API_KEY"] = v
            return
    raise SystemExit("no key")


def norm(raw):
    w = (raw or "").strip().strip('."\'*`').upper()
    return w if w in ("LEFT", "RIGHT", "NEITHER") else "NEITHER"


async def do_classify(limit=None):
    from openai import AsyncOpenAI
    load_env_key()
    client = AsyncOpenAI()
    df = pd.read_csv(OUT / "sample.csv", dtype=str, keep_default_na=False)
    if limit:  # smoke: 4 per platform
        df = df.groupby("platform", sort=False).head(max(1, limit // 5)).reset_index(drop=True)
    cache_path = OUT / ("smoke_cache.jsonl" if limit else "cache.jsonl")
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            d = json.loads(l)
            cache[d["row_id"]] = d
    sem = asyncio.Semaphore(16)
    lock = asyncio.Lock()
    fh = open(cache_path, "a")
    done = len(cache)

    async def one(rid, text):
        nonlocal done
        if rid in cache:
            return
        user = ('Post:\n"""\n' + text[:1500] + '\n"""\n\n' + USER.split('"""\n\n', 1)[1])
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        async with sem:
            for attempt in range(8):
                try:
                    r = await client.chat.completions.create(model=MODEL, messages=msgs)
                    c = r.choices[0].message.content
                    rec = {"row_id": rid, "raw": c, "model": r.model,
                           "pt": r.usage.prompt_tokens, "ct": r.usage.completion_tokens,
                           "rt": (r.usage.completion_tokens_details.reasoning_tokens if r.usage.completion_tokens_details else 0)}
                    break
                except Exception as e:
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
                    rec = {"row_id": rid, "raw": None, "error": type(e).__name__, "model": None, "pt": 0, "ct": 0, "rt": 0}
            if rec["raw"] is None and "error" in rec:
                return  # not cached, so a rerun retries it
        async with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush()
            cache[rid] = rec
            done += 1
            if done % 250 == 0:
                print(done, flush=True)

    await asyncio.gather(*[one(r, t) for r, t in zip(df.row_id, df.text)])
    fh.close()
    lab = df[["platform", "row_id"]].copy()
    lab["label"] = [norm(cache[r]["raw"]) if r in cache else "MISSING" for r in lab.row_id]
    lab["raw"] = [cache[r]["raw"] if r in cache else "" for r in lab.row_id]
    if limit:
        lab.merge(df[["row_id", "text"]], on="row_id").to_csv(OUT / "smoke_labels.csv", index=False)
    else:
        lab[["platform", "row_id", "label"]].to_csv(OUT / "labels.csv", index=False)
        lab[["row_id", "raw"]].to_csv(OUT / "labels_raw.csv", index=False)
    print("missing", (lab.label == "MISSING").sum(), lab.label.value_counts().to_dict())


def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{100*p:.1f} [{100*lo:.1f}, {100*hi:.1f}]"


def do_summarize():
    lab = pd.read_csv(OUT / "labels.csv", dtype=str)
    raw = pd.read_csv(OUT / "labels_raw.csv", dtype=str, keep_default_na=False)
    cache = [json.loads(l) for l in open(OUT / "cache.jsonl")]
    pop = json.load(open(OUT / "population_counts.json"))
    n_other = sum(1 for c in cache if (c["raw"] or "").strip().strip('."\'*`').upper() not in ("LEFT", "RIGHT", "NEITHER"))
    models = sorted({c["model"] for c in cache})
    pt = sum(c["pt"] for c in cache); ct = sum(c["ct"] for c in cache); rt = sum(c["rt"] for c in cache)
    cost = pt * 0.25 / 1e6 + ct * 2.00 / 1e6  # gpt-5-mini list price per 1M tokens: $0.25 in, $2.00 out (reasoning tokens bill as output)
    L = []
    L.append("# Political lean of sampled posts, by platform\n")
    L.append(f"Run date: {datetime.date.today()}. Model requested `{MODEL}`; model id returned: {', '.join('`'+m+'`' for m in models)}. "
             f"Simple random sample of {N_PER} posts per platform (seed {SEED}), one post per request, text truncated to 1,500 characters, identical prompt across platforms.\n")
    L.append("| Platform | Corpus N (Table 1) | n sampled | LEFT % [95% CI] | RIGHT % [95% CI] | NEITHER % [95% CI] | RIGHT / (LEFT + RIGHT) % [95% CI] |")
    L.append("|---|---|---|---|---|---|---|")
    names = {"twitter": "Twitter/X", "truth_social": "Truth Social", "reddit": "Reddit", "bluesky": "BlueSky", "mastodon": "Mastodon"}
    for p in SOURCES:
        s = lab[lab.platform == p].label
        n = len(s); l = (s == "LEFT").sum(); r = (s == "RIGHT").sum(); nn = (s == "NEITHER").sum()
        L.append(f"| {names[p]} | {SOURCES[p][1]:,} | {n} | {fmt(l,n)} | {fmt(r,n)} | {fmt(nn,n)} | {fmt(r,l+r)} (side n={l+r}) |")
    L.append("\nCounts:\n")
    L.append("| Platform | LEFT | RIGHT | NEITHER |\n|---|---|---|---|")
    for p in SOURCES:
        s = lab[lab.platform == p].label
        L.append(f"| {names[p]} | {(s=='LEFT').sum()} | {(s=='RIGHT').sum()} | {(s=='NEITHER').sum()} |")
    L.append(f"\nOutputs outside LEFT/RIGHT/NEITHER (mapped to NEITHER): {n_other} of {len(cache)} responses.\n")
    L.append(f"Token usage: {pt:,} prompt, {ct:,} completion (of which {rt:,} reasoning). Approximate cost at list price ($0.25 per 1M input, $2.00 per 1M output): ${cost:.2f}.\n")
    L.append("## Reproduction of Table 1 counts\n")
    L.append("English posts (`detected_language == en`) per platform. Twitter/X and Reddit come from `results/tr_english_posts_raw.csv`; Truth Social, BlueSky and Mastodon from `data/english_posts_raw.csv`. Row id is the 0-based ordinal of the English row within its platform in that file.\n")
    L.append("| Platform | Table 1 | Reproduced | Match |\n|---|---|---|---|")
    for p in SOURCES:
        L.append(f"| {names[p]} | {SOURCES[p][1]:,} | {pop[p]:,} | {'exact' if pop[p]==SOURCES[p][1] else 'differs'} |")
    L.append("\nThe date window (30 May to 30 June 2024) and the Biden, Harris or Trump keyword filter are applied upstream of these files; they were not re-applied here.\n")
    (OUT / "summary.md").write_text("\n".join(L))

    # spot check
    sm = pd.read_csv(OUT / "sample.csv", dtype=str, keep_default_na=False).merge(lab[["row_id", "label"]], on="row_id")
    rng = np.random.default_rng(SEED)
    S = ["# Spot check: 15 random posts per label, pooled across platforms\n", "Seed " + str(SEED) + ". Text shown in full as sampled (classifier saw the first 1,500 characters).\n"]
    for lb in ["LEFT", "RIGHT", "NEITHER"]:
        sub = sm[sm.label == lb]
        pick = sub.iloc[rng.choice(len(sub), size=min(15, len(sub)), replace=False)]
        S.append(f"\n## {lb}\n")
        for i, (_, r) in enumerate(pick.iterrows(), 1):
            t = r.text.replace("\n", " ")
            S.append(f"{i}. [{r.platform}, {r.row_id}] **{lb}**: {t}\n")
    (OUT / "spot_check.md").write_text("\n".join(S))
    print("\n".join(L))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "sample":
        do_sample()
    elif cmd == "classify":
        lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
        asyncio.run(do_classify(lim))
    elif cmd == "summarize":
        do_summarize()

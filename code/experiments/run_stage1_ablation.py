"""Stage-1 prompt ablation: score shown vs score removed (plus a with-score replicate for a noise floor).

Prompt text, system message, batch size 50, max_completion_tokens=5000 and the line parser are copied
from notebooks/03_llm_evaluation.ipynb (cell 19). Only the term line format differs in no_score.

Usage:
  uv run python run_stage1_ablation.py --smoke      # 2 batches (bluesky batch 0, with_score + no_score)
  uv run python run_stage1_ablation.py              # full run, 3 conditions x 5 platforms x 20 batches
"""
import argparse, csv, json, os, sys, time, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "experiments/ablation"
RAW = OUT / "raw"
DATA = ROOT / "results/drift_scores_csv"
PLATFORMS = ["bluesky", "mastodon", "truth_social", "reddit", "twitter"]
CONDITIONS = ["with_score", "no_score", "with_score_rep2"]
MODEL = "gpt-5-mini-2025-08-07"  # overridden by --model
RETRY_CAP = 16000  # one retry cap when first attempt returns empty text with finish_reason=length
BATCH = 50


def load_key():
    for line in open(".env", encoding="utf-8"):
        line = line.strip()
        if line.startswith("export "):
            line = line[7:]
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"').strip("'")
            return
    raise SystemExit("OPENAI_API_KEY not found")


def load_terms(platform):
    with open(DATA / f"{platform}_unigram_top1000.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [dict(rank=int(r["rank"]), term=r["term"], drift_score=float(r["drift_score"])) for r in rows]


def create_screening_prompt(terms_batch, drift_scores, show_score):
    if show_score:
        terms_list = "\n".join([f"{i+1}. {term} (drift_score: {score:.3f})"
                                for i, (term, score) in enumerate(zip(terms_batch, drift_scores))])
    else:
        terms_list = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms_batch)])
    prompt = f"""You are an expert in identifying dog whistle terminology and coded language used in social media.

I will provide you a list of terms that show high semantic drift (meaning they've changed meaning significantly in online discourse). 
Your task is to identify which terms are likely functioning as dog whistles - coded language that targets specific groups.

TERMS TO EVALUATE:
{terms_list}

For each term, respond with:
TERM | LIKELIHOOD (HIGH/MEDIUM/LOW) | TARGET_GROUP (or N/A) | BRIEF_REASON

Focus on identifying terms that:
1. Appear to have coded or hidden meaning targeting specific groups
2. Are used as substitutes for more explicit language
3. Have shifted meaning in political/social discourse
4. Commonly appear in communities known for such coded language

Only return lines for terms you believe are likely dog whistles or unclear (HIGH/MEDIUM likelihood)."""
    return prompt


def parse_response(response_text):
    """Original parser logic, but every line is recorded (original kept only HIGH/MEDIUM matches)."""
    out = []
    for line in response_text.split('\n'):
        line = line.strip()
        if not line or line.startswith('TERM'):
            continue
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 4:
            out.append(dict(term=parts[0], likelihood=parts[1],
                            target_group=parts[2] if parts[2] != 'N/A' else None,
                            reason=parts[3], parsed=True, line=line))
        else:
            out.append(dict(term=None, likelihood=None, target_group=None, reason=None, parsed=False, line=line))
    return out


def cache_path(platform, cond, bi):
    return RAW / f"{platform}__{cond}__{bi:02d}.json"


def run_one(client, platform, cond, bi, terms):
    p = cache_path(platform, cond, bi)
    if p.exists():
        return p, "cached"
    batch = terms[bi * BATCH:(bi + 1) * BATCH]
    prompt = create_screening_prompt([t["term"] for t in batch], [t["drift_score"] for t in batch],
                                     show_score=(cond != "no_score"))
    last = None
    for attempt in range(8):
        try:
            def call(cap):
                return client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": "You are an expert in identifying dog whistle terminology and coded language."},
                              {"role": "user", "content": prompt}],
                    max_completion_tokens=cap)
            r = call(5000)
            retried, first_usage = False, None
            if not (r.choices[0].message.content or "").strip() and r.choices[0].finish_reason == "length":
                retried = True
                first_usage = dict(prompt_tokens=r.usage.prompt_tokens, completion_tokens=r.usage.completion_tokens)
                r = call(RETRY_CAP)
            text = r.choices[0].message.content or ""
            u = r.usage
            rec = dict(retried_with_16000=retried, first_attempt_usage=first_usage, platform=platform, condition=cond, batch_index=bi, model_requested=MODEL,
                       model_returned=r.model, finish_reason=r.choices[0].finish_reason,
                       prompt_tokens=u.prompt_tokens, completion_tokens=u.completion_tokens,
                       total_tokens=u.total_tokens,
                       reasoning_tokens=getattr(getattr(u, "completion_tokens_details", None), "reasoning_tokens", None),
                       n_input_terms=len(batch), input_terms=[t["term"] for t in batch],
                       prompt=prompt, response_text=text, lines=parse_response(text),
                       timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
            tmp.replace(p)
            return p, "ok"
        except Exception as e:
            last = e
            time.sleep(min(60, 2 ** attempt) + random.random())
    return p, f"FAILED: {type(last).__name__}: {str(last)[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="gpt-5-mini-2025-08-07")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    global MODEL, RAW
    MODEL = a.model
    RAW = Path(a.out) / "raw"
    load_key()
    from openai import OpenAI
    client = OpenAI(max_retries=2, timeout=600)
    RAW.mkdir(parents=True, exist_ok=True)
    data = {p: load_terms(p) for p in PLATFORMS}
    if a.smoke:
        jobs = [("bluesky", c, 0) for c in ["with_score", "no_score"]]
    else:
        jobs = [(p, c, bi) for c in CONDITIONS for p in PLATFORMS for bi in range(len(data[p]) // BATCH)]
    print(f"{len(jobs)} jobs", flush=True)
    fails = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_one, client, p, c, bi, data[p]): (p, c, bi) for p, c, bi in jobs}
        for n, f in enumerate(as_completed(futs), 1):
            path, st = f.result()
            if st.startswith("FAILED"):
                fails += 1
            print(f"[{n}/{len(jobs)}] {path.name} {st}", flush=True)
    print(f"done, failures={fails}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()

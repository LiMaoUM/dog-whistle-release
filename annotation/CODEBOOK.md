# Inter-Coder Annotation Codebook — Dog-Whistle Signal Language

Purpose: provide human ground truth, produced **independently of the LLM**, so we can
report (a) inter-annotator agreement and (b) how far the LLM's HIGH/MEDIUM screening
agrees with blind human judgment. This answers reviewers who noted the pipeline has no
independent ground truth and no reported IAA.

## Rules for coders
- Work from `annotation_sheet.csv`. Each row is one **term** plus up to 8 example posts (`context_1..8`).
- Judge **only** from the term and its contexts. Do **not** look up the term elsewhere, and do
  not discuss items with the other coder until all coding is done.
- You will **not** see the drift score, the platform, or the model's label. That is intentional.
- Fill four columns per row. Save your file as `coder_<yourname>.csv` (keep `item_id` intact).

## Labels

**`is_signal_language`** (primary, required) — one of:
- `yes` = in these contexts the term carries a coded / in-group meaning beyond its surface
  sense: it signals a shared stance, target group, or grievance to an in-group while remaining
  deniable to outsiders (a "dog whistle").
- `no` = ordinary usage; the term means what it plainly says, with no coded in-group layer.
- `cant_tell` = contexts too few, off-topic, or ambiguous to judge.

**`likelihood`** (ordinal, required) — your confidence that it is signal language:
`HIGH` / `MEDIUM` / `LOW` / `NONE`. (Mirrors the pipeline's scale so the two can be compared.)
- HIGH: clearly coded, recognizable dog whistle in these contexts.
- MEDIUM: plausibly coded but context-dependent or emerging.
- LOW: probably ordinary, faint signal at most.
- NONE: ordinary usage, no coded layer.

**`coded_meaning_or_target`** (free text, only if `yes`/HIGH/MEDIUM) — one short phrase: the coded
meaning or the group/idea being signaled (e.g. "antisemitic 'elites'", "anti-trans", "pro-MAGA in-group").

**`confidence_1to3`** — 1 = guess, 2 = fairly sure, 3 = certain.

## Guidance / edge cases
- A slur used openly and non-deniably is hateful but not necessarily a *dog whistle*; if it is fully
  explicit rather than coded, lean `no`/LOW on the signal question (note it in the free text).
- Reclaimed / in-group affectionate uses still count as signal language if they mark in-group membership.
- Spelling variants and morphology (e.g. deliberate misspellings) can be signal if they encode an in-group wink;
  ordinary typos are not.
- Judge the **term as used here**, not whether it could ever be a dog whistle in principle.

## After coding
Place `coder_A.csv`, `coder_B.csv` (and a third for adjudication if available) in this folder and run
`python compute_iaa.py`. It joins to the hidden key and reports kappa/alpha, human–LLM agreement,
per-bucket precision, and any signal terms recovered from the LLM's not-flagged / low-drift strata.

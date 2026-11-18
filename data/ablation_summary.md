# Stage-1 prompt ablation: effect of showing the drift score

Run date: 2026-11-13 to 2026-11-13. Model requested `gpt-5-mini-2025-08-07`; model id returned by API: gpt-5-mini-2025-08-07.
Terms: 5 platforms x 1000 unigrams, batches of 50 in rank order, 20 batches per platform per condition. Conditions: `with_score` (original prompt), `no_score` (term line without the drift score), `with_score_rep2` (second with-score run, noise floor).

Label rule: HIGH or MEDIUM only if the model returned the term on a line with that exact likelihood string (original parser); LOW, missing, or unmatched is NONE.

## Counts

| platform | cond | HIGH | MEDIUM | flagged | flagged rate % |
|---|---|---|---|---|---|
| bluesky | with_score | 24 | 110 | 134 | 13.4 |
| bluesky | no_score | 31 | 146 | 177 | 17.7 |
| bluesky | with_score_rep2 | 23 | 120 | 143 | 14.3 |
| mastodon | with_score | 23 | 150 | 173 | 17.3 |
| mastodon | no_score | 29 | 158 | 187 | 18.7 |
| mastodon | with_score_rep2 | 25 | 137 | 162 | 16.2 |
| truth_social | with_score | 30 | 137 | 167 | 16.7 |
| truth_social | no_score | 39 | 132 | 171 | 17.1 |
| truth_social | with_score_rep2 | 33 | 113 | 146 | 14.6 |
| reddit | with_score | 28 | 136 | 164 | 16.4 |
| reddit | no_score | 33 | 148 | 181 | 18.1 |
| reddit | with_score_rep2 | 26 | 141 | 167 | 16.7 |
| twitter | with_score | 28 | 115 | 143 | 14.3 |
| twitter | no_score | 37 | 124 | 161 | 16.1 |
| twitter | with_score_rep2 | 33 | 121 | 154 | 15.4 |
| ALL | with_score | 133 | 648 | 781 | 15.6 |
| ALL | no_score | 169 | 708 | 877 | 17.5 |
| ALL | with_score_rep2 | 140 | 632 | 772 | 15.4 |

## Agreement between conditions (per 1000 terms per platform; 5000 overall)

pct = percent agreement, kappa = Cohen's kappa; flag = flagged vs not; 3way = HIGH/MEDIUM/NONE; Jaccard of flagged sets and of HIGH sets.

| platform | pair | % agree flag | kappa flag | % agree 3-way | kappa 3-way | Jaccard flagged | Jaccard HIGH |
|---|---|---|---|---|---|---|---|
| bluesky | with vs no score | 90.7 | 0.647 | 88.5 | 0.575 | 0.540 | 0.341 |
| bluesky | with vs with rep2 (noise floor) | 92.9 | 0.703 | 91.0 | 0.631 | 0.592 | 0.382 |
| bluesky | rep2 vs no score | 90.6 | 0.651 | 88.7 | 0.591 | 0.546 | 0.421 |
| mastodon | with vs no score | 86.2 | 0.533 | 84.6 | 0.492 | 0.446 | 0.368 |
| mastodon | with vs with rep2 (noise floor) | 88.5 | 0.588 | 86.5 | 0.528 | 0.489 | 0.333 |
| mastodon | rep2 vs no score | 87.3 | 0.560 | 85.7 | 0.517 | 0.466 | 0.385 |
| truth_social | with vs no score | 87.6 | 0.559 | 85.6 | 0.504 | 0.463 | 0.468 |
| truth_social | with vs with rep2 (noise floor) | 91.1 | 0.663 | 88.7 | 0.585 | 0.557 | 0.432 |
| truth_social | rep2 vs no score | 89.7 | 0.614 | 88.1 | 0.569 | 0.510 | 0.565 |
| reddit | with vs no score | 87.9 | 0.576 | 86.5 | 0.541 | 0.481 | 0.525 |
| reddit | with vs with rep2 (noise floor) | 90.7 | 0.663 | 89.0 | 0.612 | 0.561 | 0.421 |
| reddit | rep2 vs no score | 90.6 | 0.673 | 88.4 | 0.608 | 0.575 | 0.405 |
| twitter | with vs no score | 90.4 | 0.628 | 88.9 | 0.582 | 0.520 | 0.477 |
| twitter | with vs with rep2 (noise floor) | 92.7 | 0.711 | 91.1 | 0.658 | 0.605 | 0.525 |
| twitter | rep2 vs no score | 89.7 | 0.612 | 87.7 | 0.551 | 0.507 | 0.489 |
| ALL | with vs no score | 88.6 | 0.587 | 86.8 | 0.537 | 0.487 | 0.438 |
| ALL | with vs with rep2 (noise floor) | 91.2 | 0.664 | 89.3 | 0.601 | 0.558 | 0.422 |
| ALL | rep2 vs no score | 89.6 | 0.622 | 87.7 | 0.567 | 0.520 | 0.458 |

## Relation to drift (within each condition)

Spearman rho between drift rank (1 = highest drift) and flagged (0/1); negative rho means flagged terms sit at higher drift. Mean drift score of flagged vs unflagged terms.

| platform | cond | rho(rank, flagged) | p | mean drift flagged | mean drift unflagged | diff |
|---|---|---|---|---|---|---|
| bluesky | with_score | -0.035 | 0.26 | 1.1177 | 1.1157 | +0.0020 |
| bluesky | no_score | 0.005 | 0.87 | 1.1160 | 1.1160 | +0.0000 |
| bluesky | with_score_rep2 | -0.032 | 0.31 | 1.1172 | 1.1157 | +0.0015 |
| mastodon | with_score | -0.012 | 0.7 | 1.0820 | 1.0806 | +0.0014 |
| mastodon | no_score | 0.010 | 0.76 | 1.0811 | 1.0808 | +0.0003 |
| mastodon | with_score_rep2 | -0.028 | 0.37 | 1.0822 | 1.0806 | +0.0017 |
| truth_social | with_score | 0.032 | 0.32 | 1.1369 | 1.1380 | -0.0012 |
| truth_social | no_score | 0.058 | 0.067 | 1.1354 | 1.1384 | -0.0029 |
| truth_social | with_score_rep2 | 0.048 | 0.13 | 1.1359 | 1.1382 | -0.0023 |
| reddit | with_score | -0.055 | 0.083 | 1.1239 | 1.1197 | +0.0042 |
| reddit | no_score | -0.024 | 0.45 | 1.1217 | 1.1201 | +0.0016 |
| reddit | with_score_rep2 | -0.028 | 0.38 | 1.1222 | 1.1200 | +0.0022 |
| twitter | with_score | 0.018 | 0.56 | 1.1437 | 1.1447 | -0.0011 |
| twitter | no_score | -0.021 | 0.5 | 1.1447 | 1.1446 | +0.0001 |
| twitter | with_score_rep2 | 0.022 | 0.48 | 1.1433 | 1.1448 | -0.0015 |
| ALL | with_score | -0.010 | 0.47 | 1.1199 | 1.1199 | +0.0000 |
| ALL | no_score | 0.006 | 0.69 | 1.1188 | 1.1202 | -0.0014 |
| ALL | with_score_rep2 | -0.004 | 0.78 | 1.1197 | 1.1200 | -0.0003 |

## Overlap with the paper's published flagged set

Published sets: notebooks/screening_results.json (bluesky, mastodon, truth) and screening_results_tr.json (reddit, twitter). Published terms absent from the top-1000 file: {'bluesky': 0, 'mastodon': 0, 'truth_social': 0, 'reddit': 0, 'twitter': 0}. (The published run used the notebook's `batch_size=100` call and a non-dated `gpt-5-mini` alias; see notes.)

| platform | cond | n published | n flagged | intersection | Jaccard | share of published recovered % | share of flagged that is published % |
|---|---|---|---|---|---|---|---|
| bluesky | with_score | 86 | 134 | 62 | 0.392 | 72.1 | 46.3 |
| bluesky | no_score | 86 | 177 | 70 | 0.363 | 81.4 | 39.5 |
| bluesky | with_score_rep2 | 86 | 143 | 71 | 0.449 | 82.6 | 49.7 |
| mastodon | with_score | 108 | 173 | 84 | 0.426 | 77.8 | 48.6 |
| mastodon | no_score | 108 | 187 | 75 | 0.341 | 69.4 | 40.1 |
| mastodon | with_score_rep2 | 108 | 162 | 80 | 0.421 | 74.1 | 49.4 |
| truth_social | with_score | 97 | 167 | 81 | 0.443 | 83.5 | 48.5 |
| truth_social | no_score | 97 | 171 | 78 | 0.411 | 80.4 | 45.6 |
| truth_social | with_score_rep2 | 97 | 146 | 82 | 0.509 | 84.5 | 56.2 |
| reddit | with_score | 99 | 164 | 80 | 0.437 | 80.8 | 48.8 |
| reddit | no_score | 99 | 181 | 73 | 0.353 | 73.7 | 40.3 |
| reddit | with_score_rep2 | 99 | 167 | 76 | 0.400 | 76.8 | 45.5 |
| twitter | with_score | 101 | 143 | 82 | 0.506 | 81.2 | 57.3 |
| twitter | no_score | 101 | 161 | 80 | 0.440 | 79.2 | 49.7 |
| twitter | with_score_rep2 | 101 | 154 | 79 | 0.449 | 78.2 | 51.3 |
| ALL | with_score | 491 | 781 | 389 | 0.441 | 79.2 | 49.8 |
| ALL | no_score | 491 | 877 | 376 | 0.379 | 76.6 | 42.9 |
| ALL | with_score_rep2 | 491 | 772 | 388 | 0.443 | 79.0 | 50.3 |

## Parsing and API diagnostics

| cond | batches | returned lines (>=4 fields) | HIGH/MEDIUM lines unmatched to input | LOW/other lines | odd likelihood strings | unparseable lines | duplicate returned terms |
|---|---|---|---|---|---|---|---|
| with_score | 100 | 791 | 9 | 1 | 1 | 5 | 0 |
| no_score | 100 | 879 | 2 | 0 | 0 | 4 | 0 |
| with_score_rep2 | 100 | 773 | 1 | 0 | 0 | 2 | 0 |

Unmatched examples (with_score): bluesky:13. globalists, bluesky:22. farmer, bluesky:5. breeders, bluesky:3. pedigree, bluesky:33. genetics, bluesky:4. trophies, bluesky:19. bbc, reddit:annointed (anointed)
Unmatched examples (no_score): truth_social:agenices (misspelling), twitter:schill (variant of "shill")
Unmatched examples (with_score_rep2): reddit:annointed (misspelling)

Finish reasons per condition: no_score/length: 2; no_score/stop: 98; with_score/stop: 100; with_score_rep2/stop: 100

| cond | prompt tokens | completion tokens (incl. reasoning) | reasoning tokens | total | approx cost USD |
|---|---|---|---|---|---|
| with_score | 88,216 | 287,479 | 247,040 | 375,695 | 0.60 |
| no_score | 43,216 | 370,987 | 326,288 | 414,203 | 0.75 |
| with_score_rep2 | 88,216 | 286,180 | 246,784 | 374,396 | 0.59 |
| total | 219,648 | 944,646 | 820,112 | 1,164,294 | 1.94 |

Cost assumes list prices of $0.25 per 1M input and $2.00 per 1M output tokens for gpt-5-mini (assumed, not read from the API). Excludes the 2 smoke-test calls, which are the cached bluesky batch 0 responses reused in the full run.

## Sensitivity: excluding the two empty no_score batches

Two no_score batches (reddit batch index 8, i.e. ranks 401-450; twitter batch index 19, ranks 951-1000) returned empty text with finish_reason `length` (all 5000 completion tokens spent on reasoning). Their 100 terms are labelled NONE in labels.csv, as the original notebook code would have done (original cap kept verbatim). Excluding those 100 terms in all conditions (4,900 terms):

- label_with_score: flagged 763 of 4900 (15.6%)
- label_no_score: flagged 877 of 4900 (17.9%)
- label_with_score_rep2: flagged 755 of 4900 (15.4%)
- with vs no: agree 88.7%, kappa 0.595
- with vs rep2: agree 91.1%, kappa 0.660

## Notes

- The notebook's actual call is `batch_screen_terms_with_gpt(llm_eval_data, batch_size=100)`, while the function default and this ablation use 50. The ablation therefore matches the prompt text and parser, not the batch size of the published run. The published run also used the undated alias `gpt-5-mini`.
- Some HIGH/MEDIUM lines were returned with a numbering prefix (for example "13. globalists") or a misspelling, so the original exact-match parser drops them; they are counted as unmatched (12 lines in total across the three conditions) and not rescued.
- Duplicate returned terms: none. Lines with fewer than 4 pipe-separated fields are counted as unparseable (11 in total), not labelled.
- Pipeline scripts: run_stage1_ablation.py (API runs, cached in raw/), analyze_ablation.py (this file and labels.csv).

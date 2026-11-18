# Silent but Evolving: resources

Resources for "Silent but Evolving: Detecting and Analyzing Emerging Dog Whistles with Semantic
Drift Modeling and LLMs Across Social Media Platforms" (EACL 2027).

- `data/candidates.csv`: the 492 platform-specific candidates with drift score, likelihood label
  (HIGH or MEDIUM), inferred target group and screening rationale.
- `data/drift_top1000_<platform>.csv`: the 1,000 highest-drift unigrams per platform that were screened.
- `data/high_term_coverage.csv`: whole-word post counts per platform for the 87 distinct HIGH terms.
- `data/per_post_projections.csv`, `data/projection_*.csv`: anchor projections of the six focal
  terms (numbers only) and the tests reported in the paper.
- `data/drift_*.csv`, `data/vetting_*.csv`, `data/lean_*`, `data/ablation_*`: inputs to the
  appendix tables.
- `annotation/`: codebook and scoring script for the 150-term blind sample.
- `code/`: pipeline modules and analysis scripts.

The release contains no post text and no user identifiers. Context windows for the candidate
terms are available from the authors on request for research use.

# Political lean of sampled posts, by platform

Run date: 2026-11-13. Model requested `gpt-5-mini-2025-08-07`; model id returned: `gpt-5-mini-2025-08-07`. Simple random sample of 2000 posts per platform (seed 20261113), one post per request, text truncated to 1,500 characters, identical prompt across platforms.

| Platform | Corpus N (Table 1) | n sampled | LEFT % [95% CI] | RIGHT % [95% CI] | NEITHER % [95% CI] | RIGHT / (LEFT + RIGHT) % [95% CI] |
|---|---|---|---|---|---|---|
| Twitter/X | 4,084,887 | 2000 | 32.2 [30.2, 34.3] | 50.0 [47.8, 52.2] | 17.8 [16.2, 19.5] | 60.8 [58.4, 63.2] (side n=1644) |
| Truth Social | 2,233,482 | 2000 | 12.7 [11.3, 14.2] | 40.9 [38.8, 43.1] | 46.4 [44.2, 48.5] | 76.3 [73.7, 78.8] (side n=1073) |
| Reddit | 1,632,879 | 2000 | 48.1 [45.9, 50.3] | 23.8 [22.0, 25.7] | 28.1 [26.2, 30.1] | 33.1 [30.7, 35.6] (side n=1438) |
| BlueSky | 1,078,237 | 2000 | 24.0 [22.2, 25.9] | 6.1 [5.1, 7.2] | 69.9 [67.9, 71.9] | 20.3 [17.2, 23.7] (side n=602) |
| Mastodon | 58,091 | 2000 | 46.9 [44.7, 49.1] | 6.5 [5.5, 7.6] | 46.7 [44.5, 48.8] | 12.1 [10.3, 14.2] (side n=1067) |

Counts:

| Platform | LEFT | RIGHT | NEITHER |
|---|---|---|---|
| Twitter/X | 644 | 1000 | 356 |
| Truth Social | 254 | 819 | 927 |
| Reddit | 962 | 476 | 562 |
| BlueSky | 480 | 122 | 1398 |
| Mastodon | 938 | 129 | 933 |

Outputs outside LEFT/RIGHT/NEITHER (mapped to NEITHER): 0 of 10000 responses.

Token usage: 2,293,126 prompt, 1,210,814 completion (of which 1,106,624 reasoning). Approximate cost at list price ($0.25 per 1M input, $2.00 per 1M output): $2.99.

## Reproduction of Table 1 counts

English posts (`detected_language == en`) per platform. Twitter/X and Reddit come from `results/tr_english_posts_raw.csv`; Truth Social, BlueSky and Mastodon from `data/english_posts_raw.csv`. Row id is the 0-based ordinal of the English row within its platform in that file.

| Platform | Table 1 | Reproduced | Match |
|---|---|---|---|
| Twitter/X | 4,084,887 | 4,084,887 | exact |
| Truth Social | 2,233,482 | 2,233,482 | exact |
| Reddit | 1,632,879 | 1,632,879 | exact |
| BlueSky | 1,078,237 | 1,078,237 | exact |
| Mastodon | 58,091 | 58,091 | exact |

The date window (30 May to 30 June 2024) and the Biden, Harris or Trump keyword filter are applied upstream of these files; they were not re-applied here.

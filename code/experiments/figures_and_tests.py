"""Camera-ready projection figures and significance tests from one per-post file.

Input: a CSV with columns term, platform, X, Y, one row per post, where
X = cos(e, a_right) - cos(e, a_left) and Y = cos(e, a_pos) - cos(e, a_neg)
(cosine per post, then averaged per cell).

Outputs (in --out):
  cells.csv              per term x platform: n, mean X, mean Y, bootstrap 95% CIs
  contrasts.csv          pre-specified pairwise contrasts (Welch t, bootstrap CI, Holm)
  omnibus.csv            Kruskal-Wallis across platforms per term and axis (Holm)
  centroid_difference.png, contrast_plots.png, context_clouds.png
  tables.tex             LaTeX rows for the cell table and the contrast table

Usage:
  uv run python experiments/camera_ready/figures_and_tests.py \
      --posts experiments/figure3_data/per_post_projections.csv \
      --out experiments/camera_ready/out
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

TERMS = ["adolph", "banker", "caliphate", "fairy", "frog", "maga"]
PLATFORMS = ["twitter", "truth", "reddit", "mastodon", "bluesky"]
PLABEL = {"twitter": "Twitter/X", "truth": "Truth", "reddit": "Reddit",
          "mastodon": "Mastodon", "bluesky": "BlueSky"}
COLORS = {"twitter": "#D55E00", "truth": "#8A6D00", "reddit": "#009E73",
          "mastodon": "#0072B2", "bluesky": "#9467BD"}
MARKERS = {"twitter": "o", "truth": "s", "reddit": "D", "mastodon": "^", "bluesky": "v"}
LOW_N = 30          # cells below this are drawn hollow
MIN_TEST_N = 30     # cells below this enter no test
N_BOOT = 5000
RNG = np.random.default_rng(20261113)
# Every pair of platforms is tested on both axes when both cells have at least
# MIN_TEST_N posts.
import itertools
PAIRS = list(itertools.combinations(["truth", "twitter", "reddit", "bluesky", "mastodon"], 2))

def boot_mean_ci(x):
    if len(x) < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(axis=1)
    return tuple(np.percentile(m, [2.5, 97.5]))


def boot_diff_ci(a, b):
    ia = RNG.integers(0, len(a), size=(N_BOOT, len(a)))
    ib = RNG.integers(0, len(b), size=(N_BOOT, len(b)))
    d = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return tuple(np.percentile(d, [2.5, 97.5]))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--full-n", default=None,
                    help="optional CSV term,platform,n_posts with full cell sizes")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(args.posts)
    d["term"] = d["term"].str.lower()
    d = d[d.term.isin(TERMS) & d.platform.isin(PLATFORMS)]
    full_n = {}
    if args.full_n:
        f = pd.read_csv(args.full_n)
        full_n = {(t.lower(), p): int(n) for t, p, n in zip(f.term, f.platform, f.n_posts)}

    # ---- cells
    rows = []
    for t in TERMS:
        for p in PLATFORMS:
            g = d[(d.term == t) & (d.platform == p)]
            x, y = g.X.to_numpy(), g.Y.to_numpy()
            xl, xh = boot_mean_ci(x)
            yl, yh = boot_mean_ci(y)
            rows.append(dict(term=t, platform=p, n_used=len(g),
                             n_full=full_n.get((t, p), len(g)),
                             X=x.mean() if len(x) else np.nan, X_lo=xl, X_hi=xh,
                             Y=y.mean() if len(y) else np.nan, Y_lo=yl, Y_hi=yh))
    cells = pd.DataFrame(rows)
    cells.to_csv(out / "cells.csv", index=False)

    # ---- contrasts
    rows = []
    for t in TERMS:
        for a, b in PAIRS:
            ga = d[(d.term == t) & (d.platform == a)]
            gb = d[(d.term == t) & (d.platform == b)]
            if min(len(ga), len(gb)) < MIN_TEST_N:
                continue
            for axis, name in (("Y", "framing"), ("X", "ideology")):
                xa, xb = ga[axis].to_numpy(), gb[axis].to_numpy()
                tt = stats.ttest_ind(xa, xb, equal_var=False)
                lo, hi = boot_diff_ci(xa, xb)
                rows.append(dict(term=t, axis=name, a=a, b=b, n_a=len(xa), n_b=len(xb),
                                 mean_a=xa.mean(), mean_b=xb.mean(),
                                 diff=xa.mean() - xb.mean(), ci_lo=lo, ci_hi=hi,
                                 welch_t=tt.statistic, p=tt.pvalue,
                                 sign_flip=bool(np.sign(xa.mean()) != np.sign(xb.mean()))))
    con = pd.DataFrame(rows)
    con["p_holm"] = holm(con.p.to_numpy())
    con.to_csv(out / "contrasts.csv", index=False)

    rows = []
    for t in TERMS:
        for axis, name in (("Y", "framing"), ("X", "ideology")):
            groups = [d[(d.term == t) & (d.platform == p)][axis].to_numpy() for p in PLATFORMS]
            groups = [g for g in groups if len(g) >= MIN_TEST_N]
            h = stats.kruskal(*groups)
            rows.append(dict(term=t, axis=name, k_platforms=len(groups), H=h.statistic, p=h.pvalue))
    omni = pd.DataFrame(rows)
    omni["p_holm"] = holm(omni.p.to_numpy())
    omni.to_csv(out / "omnibus.csv", index=False)

    # ---- figure: deviation from the cross-platform mean centroid
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), TERMS):
        c = cells[cells.term == t].set_index("platform")
        mx, my = c.X.mean(), c.Y.mean()
        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6, ls="--")
        for p in PLATFORMS:
            r = c.loc[p]
            if np.isnan(r.X):
                continue
            dx, dy = r.X - mx, r.Y - my
            low = r.n_full < LOW_N
            ax.plot([0, dx], [0, dy], color=COLORS[p], lw=1.2, alpha=0.5 if low else 0.9,
                    ls=":" if low else "-")
            if not np.isnan(r.X_lo):
                ax.errorbar(dx, dy, xerr=[[r.X - r.X_lo], [r.X_hi - r.X]],
                            yerr=[[r.Y - r.Y_lo], [r.Y_hi - r.Y]], fmt="none",
                            ecolor=COLORS[p], elinewidth=0.8, alpha=0.6)
            ax.scatter(dx, dy, marker=MARKERS[p], s=42, zorder=3,
                       facecolor="white" if low else COLORS[p], edgecolor=COLORS[p], linewidth=1.2)
        ax.set_title(t, fontstyle="italic")
        ax.set_xlim(-0.03, 0.03)
        ax.set_ylim(-0.02, 0.02)
        ax.set_xticks([-0.02, 0, 0.02])
    handles = [plt.Line2D([], [], marker=MARKERS[p], color=COLORS[p], ls="", label=PLABEL[p])
               for p in PLATFORMS]
    handles.append(plt.Line2D([], [], marker="o", markerfacecolor="white", color="0.3", ls="",
                              label=f"n < {LOW_N} (hollow)"))
    fig.legend(handles=handles, loc="center right", frameon=False)
    fig.supxlabel("Ideology: platform centroid minus cross-platform mean (right is positive)")
    fig.supylabel("Framing: platform centroid minus cross-platform mean (positive is up)")
    fig.subplots_adjust(right=0.86, wspace=0.08, hspace=0.18, left=0.09, bottom=0.1)
    fig.savefig(out / "centroid_difference.png", dpi=300)
    plt.close(fig)

    # ---- figure: per-platform values with CIs
    fig, axes = plt.subplots(2, 6, figsize=(12.5, 4.4), sharey="row", sharex=True)
    for j, t in enumerate(TERMS):
        c = cells[cells.term == t].set_index("platform")
        for i, (axis, lab) in enumerate((("X", "Ideology\n(right is positive)"),
                                         ("Y", "Framing\n(positive is up)"))):
            ax = axes[i, j]
            ax.axhline(0, color="0.3", lw=0.7)
            for k, p in enumerate(PLATFORMS):
                r = c.loc[p]
                if np.isnan(r[axis]):
                    continue
                low = r.n_full < LOW_N
                ax.plot([k, k], [0, r[axis]], color=COLORS[p], lw=1.5, alpha=0.5 if low else 0.9)
                if not np.isnan(r[f"{axis}_lo"]):
                    ax.errorbar(k, r[axis], yerr=[[r[axis] - r[f"{axis}_lo"]],
                                                  [r[f"{axis}_hi"] - r[axis]]],
                                fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2)
                ax.scatter(k, r[axis], marker=MARKERS[p], s=36, zorder=3,
                           facecolor="white" if low else COLORS[p], edgecolor=COLORS[p],
                           linewidth=1.2)
            if i == 0:
                ax.set_title(t, fontstyle="italic")
            if j == 0:
                ax.set_ylabel(lab)
            ax.set_xticks(range(len(PLATFORMS)))
            ax.set_xticklabels([PLABEL[p] for p in PLATFORMS], rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out / "contrast_plots.png", dpi=300)
    plt.close(fig)

    # ---- figure: post-level clouds
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), TERMS):
        ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.axvline(0, color="0.6", lw=0.6, ls="--")
        for p in PLATFORMS:
            g = d[(d.term == t) & (d.platform == p)]
            if len(g) == 0:
                continue
            gs = g.sample(min(len(g), 400), random_state=1)
            ax.scatter(gs.X, gs.Y, s=4, color=COLORS[p], alpha=0.15, linewidths=0)
        c = cells[cells.term == t].set_index("platform")
        for p in PLATFORMS:
            r = c.loc[p]
            if np.isnan(r.X):
                continue
            low = r.n_full < LOW_N
            ax.scatter(r.X, r.Y, marker=MARKERS[p], s=48, zorder=3,
                       facecolor="white" if low else COLORS[p], edgecolor="black", linewidth=0.8)
        ax.set_title(t, fontstyle="italic")
        ax.set_xlim(-0.12, 0.14)
        ax.set_ylim(-0.12, 0.12)
    fig.legend(handles=handles, loc="center right", frameon=False)
    fig.supxlabel("Ideology X (right is positive)")
    fig.supylabel("Framing Y (positive is up)")
    fig.subplots_adjust(right=0.86, wspace=0.08, hspace=0.18, left=0.09, bottom=0.1)
    fig.savefig(out / "context_clouds.png", dpi=300)
    plt.close(fig)

    # ---- LaTeX
    def fmt_p(p):
        if p < 1e-4:
            e = int(np.floor(np.log10(p))) if p > 0 else -300
            return f"$<10^{{{e + 1}}}$"
        return f"{p:.3f}"

    lines = ["% contrasts: term & axis & contrast & n & diff & 95% CI & Holm p"]
    for _, r in con.iterrows():
        lines.append(f"\\textit{{{r.term}}} & {r.axis} & {PLABEL[r.a]} vs.\\ {PLABEL[r.b]} & "
                     f"{r.n_a:,}/{r.n_b:,} & ${r['diff']:+.3f}$ & "
                     f"[${r.ci_lo:+.3f}$, ${r.ci_hi:+.3f}$] & {fmt_p(r.p_holm)} \\\\")
    lines.append("% cells: term & platform & n & X [CI] & Y [CI]")
    for _, r in cells.iterrows():
        ci = (lambda a, lo, hi: f"${a:+.3f}$ [${lo:+.3f}$, ${hi:+.3f}$]"
              if not np.isnan(lo) else (f"${a:+.3f}$" if not np.isnan(a) else "--"))
        lines.append(f"\\textit{{{r.term}}} & {PLABEL[r.platform]} & {int(r.n_full):,} & "
                     f"{ci(r.X, r.X_lo, r.X_hi)} & {ci(r.Y, r.Y_lo, r.Y_hi)} \\\\")
    (out / "tables.tex").write_text("\n".join(lines) + "\n")

    pd.set_option("display.width", 200)
    print(cells.round(4).to_string(index=False))
    print(con.round(4).to_string(index=False))
    print(omni.to_string(index=False))


if __name__ == "__main__":
    main()

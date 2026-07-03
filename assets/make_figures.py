"""Generate the README figures from the raw result JSONs in results/raw/.

Usage (from the repo root):
    python assets/make_figures.py

Outputs:
    assets/fig_main_results.png   -- main comparison (StrictPairAcc / FlipRate / Macro-F1)
    assets/fig_lambda_sensitivity.png -- lambda ablation (StrictPairAcc / ProbGap vs lambda)

Only matplotlib is required; the numbers are read directly from the shared
result files so the figures stay in sync with the tables in the README.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
OUT = ROOT / "assets"

# Chart chrome (light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#2a78d6"      # 3-seed mean
SEED = "#9ec5f4"      # per-seed observations


def load(name: str) -> dict:
    with open(RAW / name, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs)


def style_ax(ax, title, higher_better=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    suffix = "" if higher_better is None else ("  (higher = better)" if higher_better else "  (lower = better)")
    ax.set_title(title + suffix, fontsize=10.5, color=INK, pad=10, loc="left")


def main_results_figure():
    core = load("results_core_followup.json")
    mask = load("results_masking.json")
    rows = [
        ("Baseline", core["Baseline"]),
        ("Masking Cons Reg", mask["Masking Cons Reg"]),
        ("Naive Swap", core["Naive Swap"]),
        ("Strict-Gated", core["Strict-Gated"]),
        ("Strict-Matched", core["Strict-Matched"]),
    ]
    metrics = [
        ("strict_pair_accuracy", "StrictPairAcc ↑ (main metric)", None),
        ("flip_rate", "FlipRate ↓", None),
        ("f1", "Macro-F1 ↑ (guardrail)", None),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), facecolor=SURFACE)
    ypos = list(range(len(rows)))[::-1]
    for ax, (key, title, hb) in zip(axes, metrics):
        style_ax(ax, title, hb)
        for y, (name, r) in zip(ypos, rows):
            vals = r[key]
            ax.plot(vals, [y] * len(vals), "o", color=SEED, ms=6, zorder=2)
            ax.plot([mean(vals)], [y], "o", color=BLUE, ms=9, zorder=3)
        ax.set_yticks(ypos)
        ax.set_ylim(-0.6, len(rows) - 0.4)
    axes[0].set_yticklabels(
        [n for n, _ in rows], fontsize=9.5, color=INK_2,
        fontweight="normal",
    )
    axes[0].get_yticklabels()[-1].set_fontweight("bold")  # Strict-Matched (bottom row)
    for ax in axes[1:]:
        ax.set_yticklabels([])
    fig.legend(
        handles=[
            plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=9, label="3-seed mean"),
            plt.Line2D([], [], marker="o", ls="", color=SEED, ms=6, label="per-seed run (42 / 123 / 456)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, -0.06), labelcolor=INK_2,
    )
    fig.suptitle(
        "Validity-gated CCR: robustness improves while Macro-F1 stays flat"
        "  (↑ higher = better, ↓ lower = better)",
        fontsize=12, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT / "fig_main_results.png", dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def lambda_figure():
    core = load("results_core_followup.json")
    lam05 = load("results_strict_lam005.json")
    lam25 = load("results_strict_lam025.json")
    # (lambda, row) for the Strict-Gated family
    pts = [
        (0.05, lam05["Strict_lam=0.05"]),
        (0.10, core["Strict-Gated"]),
        (0.1297, core["Strict-Matched"]),
        (0.15, core["Strict_lam=0.15"]),
        (0.25, lam25["Strict_lam=0.25"]),
    ]
    metrics = [
        ("strict_pair_accuracy", "StrictPairAcc vs λ", True),
        ("prob_gap", "ProbGap vs λ", False),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    for ax, (key, title, hb) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        suffix = "  (higher = better)" if hb else "  (lower = better)"
        ax.set_title(title + suffix, fontsize=10.5, color=INK, pad=10, loc="left")

        xs = [l for l, _ in pts]
        means = [mean(r[key]) for _, r in pts]
        for l, r in pts:
            ax.plot([l] * len(r[key]), r[key], "o", color=SEED, ms=5, zorder=2)
        ax.plot(xs, means, "-o", color=BLUE, ms=7, lw=2, zorder=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(["0.05", "0.10", "0.13†", "0.15", "0.25"])
        ax.set_xlabel("λ (consistency weight)", fontsize=9.5, color=INK_2)
    fig.suptitle(
        "λ sensitivity (Strict-Gated family) — † = coverage-matched λ (Strict-Matched)",
        fontsize=11.5, color=INK, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "fig_lambda_sensitivity.png", dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    main_results_figure()
    lambda_figure()
    print("wrote", OUT / "fig_main_results.png")
    print("wrote", OUT / "fig_lambda_sensitivity.png")

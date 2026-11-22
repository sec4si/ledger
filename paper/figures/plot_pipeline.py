#!/usr/bin/env python3
"""Generate Ledger pipeline architecture diagram (fig:pipeline).

Output: paper/figures/pipeline.pdf (vector, single-column article width ~6.5 in).
Palette matches dirty_l2_sweep.py and per_kernel_ab.py (no red).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def rounded_box(
    ax, x: float, y: float, w: float, h: float,
    facecolor: str, edgecolor: str, lw: float = 1.0,
    radius: float = 0.08, linestyle: str = "-", zorder: int = 2,
):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax, x0: float, y0: float, x1: float, y1: float,
    color: str = "#4a5568", lw: float = 1.2, rad: float = 0.0, zorder: int = 4,
):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>,head_length=4.5,head_width=2.6",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(6.5, 2.75))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    # Colors (consistent with dirty_l2_sweep & per_kernel_ab, no red)
    BLUE = "#1f5a99"
    BLUE_BG = "#eef4fa"
    TEAL = "#2a9d8f"
    TEAL_DARK = "#1e7a6f"
    TEAL_BG = "#edf7f6"
    ORANGE = "#d95f02"
    ORANGE_DARK = "#b84d00"
    ORANGE_BG = "#fdf4ec"
    SLATE = "#4a5568"
    DARK = "#1a202c"
    GRAY_BG = "#f7f8fa"
    GRAY_EDGE = "#cbd5e0"

    # ------------------------------------------------------------------
    # Stage containers
    # ------------------------------------------------------------------
    # Stage 1: x in [0.06, 3.20] (center 1.63)
    rounded_box(ax, 0.06, 0.12, 3.14, 3.92, BLUE_BG, BLUE, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        1.63, 3.84, "1. Capture (Runtime)",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color=BLUE, zorder=3,
    )

    # Stage 2: x in [3.62, 6.62] (center 5.12)
    rounded_box(ax, 3.62, 0.12, 3.00, 3.92, TEAL_BG, TEAL, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        5.12, 3.84, "2. Attribution (analyze)",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color=TEAL_DARK, zorder=3,
    )

    # Stage 3: x in [6.98, 9.94] (center 8.46)
    rounded_box(ax, 6.98, 0.12, 2.96, 3.92, ORANGE_BG, ORANGE, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        8.46, 3.84, "3. Accounting Report",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color=ORANGE_DARK, zorder=3,
    )

    # ------------------------------------------------------------------
    # Stage 1 internals (inner x in [0.20, 3.06], w=2.86)
    # ------------------------------------------------------------------
    rounded_box(ax, 0.20, 2.20, 2.86, 1.42, "white", BLUE, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        1.63, 3.43, "Unmodified engine",
        ha="center", va="center", fontsize=7.6, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 3.18, "+ LD_PRELOAD libledger.so",
        ha="center", va="center", fontsize=6.6, family="monospace", color=BLUE, zorder=3,
    )
    ax.text(
        1.63, 2.81, "CUPTI callbacks: graph node\nparams & eager launches",
        ha="center", va="center", fontsize=6.6, color=SLATE, linespacing=1.22, zorder=3,
    )
    ax.text(
        1.63, 2.36, "CUPTI activity: kernel timestamps",
        ha="center", va="center", fontsize=6.6, color=SLATE, zorder=3,
    )

    # 4 JSONL traces
    rounded_box(ax, 0.20, 1.32, 2.86, 0.64, GRAY_BG, GRAY_EDGE, lw=0.9, radius=0.07, zorder=2)
    ax.text(
        1.63, 1.75, "nodes, clones, launches, kernels",
        ha="center", va="center", fontsize=6.0, family="monospace", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 1.48, "per-run JSONL traces",
        ha="center", va="center", fontsize=6.6, color=SLATE, zorder=3,
    )
    arrow(ax, 1.63, 2.20, 1.63, 1.96, color=BLUE, lw=1.1)

    # One-time regions.json export
    rounded_box(ax, 0.20, 0.28, 2.86, 0.82, "white", BLUE, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        1.63, 0.90, "regions.json (one-time export)",
        ha="center", va="center", fontsize=6.4, family="monospace", fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 0.54, "Address ranges: weights, KV cache,\n& recurrent-state tensors",
        ha="center", va="center", fontsize=6.5, color=SLATE, linespacing=1.22, zorder=3,
    )

    # ------------------------------------------------------------------
    # Stage 2 internals (inner x in [3.76, 6.48], w=2.72)
    # ------------------------------------------------------------------
    rounded_box(ax, 3.76, 2.66, 2.72, 0.96, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 3.40, "Join timings & params",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 2.98, "Match activity records via\ngraph node ID or correlation ID",
        ha="center", va="center", fontsize=6.5, color=SLATE, linespacing=1.25, zorder=3,
    )

    rounded_box(ax, 3.76, 1.46, 2.72, 0.96, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 2.20, "Resolve pointer arguments",
        ha="center", va="center", fontsize=7.3, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 1.78, "Scan 8-byte words in param bytes;\nbisect into memory regions",
        ha="center", va="center", fontsize=6.5, color=SLATE, linespacing=1.25, zorder=3,
    )

    rounded_box(ax, 3.76, 0.28, 2.72, 0.94, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 1.00, "Per-launch accounting",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 0.59, "Weight streamed, caches touched,\n& duration per launch",
        ha="center", va="center", fontsize=6.5, color=SLATE, linespacing=1.25, zorder=3,
    )

    arrow(ax, 5.12, 2.66, 5.12, 2.42, color=TEAL_DARK, lw=1.1)
    arrow(ax, 5.12, 1.46, 5.12, 1.22, color=TEAL_DARK, lw=1.1)

    # ------------------------------------------------------------------
    # Stage 3 internals (inner x in [7.12, 9.80], w=2.68)
    # ------------------------------------------------------------------
    items = [
        ("Step books", "Busy, idle, bytes, vs. BW ceiling"),
        ("Per-weight efficiency", "% ceiling & lost ms/step"),
        ("Same kernel/shape split", "Divergence + prior cache kernel"),
        ("Repeated reads", "Weights streamed >1x per step"),
    ]
    y_tops = [3.62, 2.78, 1.94, 1.10]
    box_h = 0.72
    for (title, desc), y_top in zip(items, y_tops):
        y0 = y_top - box_h
        rounded_box(ax, 7.12, y0, 2.68, box_h, "white", ORANGE, lw=0.85, radius=0.07, zorder=2)
        ax.text(
            7.24, y_top - 0.23, title,
            ha="left", va="center", fontsize=7.3, fontweight="bold", color=DARK, zorder=3,
        )
        ax.text(
            7.24, y_top - 0.51, desc,
            ha="left", va="center", fontsize=6.5, color=SLATE, zorder=3,
        )

    # ------------------------------------------------------------------
    # Inter-stage arrows (cleanly routed in the channel between stages)
    # ------------------------------------------------------------------
    arrow(ax, 3.06, 1.66, 3.76, 3.08, color=BLUE, lw=1.2, rad=-0.08)
    arrow(ax, 3.06, 0.70, 3.76, 1.90, color=BLUE, lw=1.2, rad=0.08)
    arrow(ax, 6.62, 2.08, 6.98, 2.08, color=TEAL_DARK, lw=1.5)

    out_dir = Path(__file__).resolve().parent
    fig.tight_layout(pad=0.1)
    fig.savefig(out_dir / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

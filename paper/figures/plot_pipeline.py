#!/usr/bin/env python3
"""Generate Ledger pipeline architecture diagram (fig:pipeline).

Shows both the lightweight steady-state accounting path (~0.3% overhead) and
the deep access-tracing & traffic path (~80x slower, run on a few steps only).
Output: paper/figures/pipeline.pdf (vector, single-column width 6.5 in, no red).
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
    color: str = "#4a5568", lw: float = 1.15, rad: float = 0.0,
    linestyle: str = "-", zorder: int = 4,
):
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle="-|>,head_length=4.2,head_width=2.4",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=lw, linestyle=linestyle, zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # Single-column width: 6.5 in x 4.05 in -> coordinate space [0, 10] x [0, 6.25]
    fig, ax = plt.subplots(figsize=(6.5, 4.05))
    ax.set_xlim(0, 10.0)
    ax.set_ylim(0, 6.25)
    ax.axis("off")

    # Palette (matches dirty_l2_sweep & per_kernel_ab; no red)
    BLUE = "#1f5a99"
    BLUE_BG = "#eef4fa"
    TEAL = "#2a9d8f"
    TEAL_DARK = "#1e7a6f"
    TEAL_BG = "#edf7f6"
    ORANGE = "#d95f02"
    ORANGE_DARK = "#b84d00"
    ORANGE_BG = "#fdf4ec"
    INDIGO = "#54658a"
    INDIGO_BG = "#f2f4f8"
    SLATE = "#4a5568"
    DARK = "#1a202c"
    GRAY_BG = "#f7f8fa"
    GRAY_EDGE = "#cbd5e0"

    # ==================================================================
    # TOP BAND: Lightweight per-launch accounting (~0.3% overhead)
    # y in [2.30, 6.18]
    # ==================================================================
    # Stage 1: Capture (Runtime)
    rounded_box(ax, 0.06, 2.34, 3.14, 3.82, BLUE_BG, BLUE, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        1.63, 5.96, "1. Capture (~0.3% overhead)",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color=BLUE, zorder=3,
    )

    # Stage 2: Attribution
    rounded_box(ax, 3.62, 2.34, 3.00, 3.82, TEAL_BG, TEAL, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        5.12, 5.96, "2. Attribution (analyze)",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color=TEAL_DARK, zorder=3,
    )

    # Stage 3: Accounting Report
    rounded_box(ax, 6.98, 2.34, 2.96, 3.82, ORANGE_BG, ORANGE, lw=1.1, radius=0.12, zorder=1)
    ax.text(
        8.46, 5.96, "3. Accounting Report",
        ha="center", va="center", fontsize=7.8, fontweight="bold", color=ORANGE_DARK, zorder=3,
    )

    # --- Stage 1 internals ---
    rounded_box(ax, 0.20, 4.38, 2.86, 1.38, "white", BLUE, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        1.63, 5.58, "Unmodified engine",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 5.33, "+ LD_PRELOAD libledger.so",
        ha="center", va="center", fontsize=6.4, family="monospace", color=BLUE, zorder=3,
    )
    ax.text(
        1.63, 4.97, "CUPTI callbacks: graph node\nparams & eager launches",
        ha="center", va="center", fontsize=6.4, color=SLATE, linespacing=1.22, zorder=3,
    )
    ax.text(
        1.63, 4.53, "CUPTI activity: kernel timestamps",
        ha="center", va="center", fontsize=6.4, color=SLATE, zorder=3,
    )

    rounded_box(ax, 0.20, 3.52, 2.86, 0.62, GRAY_BG, GRAY_EDGE, lw=0.9, radius=0.07, zorder=2)
    ax.text(
        1.63, 3.94, "nodes, clones, launches, kernels",
        ha="center", va="center", fontsize=5.9, family="monospace", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 3.68, "per-run JSONL traces",
        ha="center", va="center", fontsize=6.4, color=SLATE, zorder=3,
    )
    arrow(ax, 1.63, 4.38, 1.63, 4.14, color=BLUE, lw=1.1)

    rounded_box(ax, 0.20, 2.48, 2.86, 0.82, "white", BLUE, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        1.63, 3.10, "regions.json (one-time export)",
        ha="center", va="center", fontsize=6.3, family="monospace", fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 2.74, "Address ranges: weights, KV cache,\n& recurrent-state tensors",
        ha="center", va="center", fontsize=6.3, color=SLATE, linespacing=1.22, zorder=3,
    )

    # --- Stage 2 internals ---
    rounded_box(ax, 3.76, 4.82, 2.72, 0.94, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 5.55, "Join timings & params",
        ha="center", va="center", fontsize=7.2, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 5.14, "Match activity records via\ngraph node ID or correlation ID",
        ha="center", va="center", fontsize=6.3, color=SLATE, linespacing=1.24, zorder=3,
    )

    rounded_box(ax, 3.76, 3.64, 2.72, 0.94, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 4.37, "Resolve pointer arguments",
        ha="center", va="center", fontsize=7.2, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 3.96, "Scan 8-byte words in param bytes;\nbisect into memory regions",
        ha="center", va="center", fontsize=6.3, color=SLATE, linespacing=1.24, zorder=3,
    )

    rounded_box(ax, 3.76, 2.48, 2.72, 0.92, "white", TEAL, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        5.12, 3.19, "Per-launch accounting",
        ha="center", va="center", fontsize=7.2, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 2.79, "Weight streamed, caches touched,\n& duration per launch",
        ha="center", va="center", fontsize=6.3, color=SLATE, linespacing=1.24, zorder=3,
    )

    arrow(ax, 5.12, 4.82, 5.12, 4.58, color=TEAL_DARK, lw=1.1)
    arrow(ax, 5.12, 3.64, 5.12, 3.40, color=TEAL_DARK, lw=1.1)

    # --- Stage 3 internals ---
    items = [
        ("Step books", "Busy, idle, bytes, vs. BW ceiling"),
        ("Per-weight efficiency", "% ceiling & lost ms/step"),
        ("Same kernel/shape split", "Divergence + prior cache kernel"),
        ("Repeated reads", "Weights streamed >1x per step"),
    ]
    y_tops = [5.76, 4.94, 4.12, 3.30]
    box_h = 0.70
    for (title, desc), y_top in zip(items, y_tops):
        y0 = y_top - box_h
        rounded_box(ax, 7.12, y0, 2.68, box_h, "white", ORANGE, lw=0.85, radius=0.07, zorder=2)
        ax.text(
            7.24, y_top - 0.22, title,
            ha="left", va="center", fontsize=7.1, fontweight="bold", color=DARK, zorder=3,
        )
        ax.text(
            7.24, y_top - 0.50, desc,
            ha="left", va="center", fontsize=6.3, color=SLATE, zorder=3,
        )

    # Top inter-stage arrows
    arrow(ax, 3.06, 3.84, 3.76, 5.25, color=BLUE, lw=1.15, rad=-0.08)
    arrow(ax, 3.06, 2.90, 3.76, 4.08, color=BLUE, lw=1.15, rad=0.08)
    arrow(ax, 6.62, 4.24, 6.98, 4.24, color=TEAL_DARK, lw=1.4)

    # ==================================================================
    # BOTTOM BAND: Deep access-tracing & traffic branch (~80x slower)
    # y in [0.08, 2.08]
    # ==================================================================
    rounded_box(
        ax, 0.06, 0.08, 9.88, 2.02, INDIGO_BG, INDIGO,
        lw=1.0, radius=0.12, linestyle="--", zorder=1,
    )
    ax.text(
        5.35, 1.91, "Optional Deep Path: Access Tracing & Traffic (~80x slower; run on a few steps only)",
        ha="center", va="center", fontsize=7.5, fontweight="bold", color=INDIGO, zorder=3,
    )

    # Left: Compute Sanitizer access tracer
    rounded_box(ax, 0.20, 0.24, 2.86, 1.46, "white", INDIGO, lw=0.9, radius=0.08, zorder=2)
    ax.text(
        1.63, 1.50, "access/libledger_access.so",
        ha="center", va="center", fontsize=6.4, family="monospace", fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        1.63, 1.18, "Compute Sanitizer R/W granules",
        ha="center", va="center", fontsize=6.4, color=INDIGO, fontweight="bold", zorder=3,
    )
    ax.text(
        1.63, 0.67, "Logs 1 KiB R/W granules per launch;\nsplits CUDA graphs per node;\ndecodes TMA tensor maps",
        ha="center", va="center", fontsize=6.1, color=SLATE, linespacing=1.24, zorder=3,
    )

    # Middle-top: ledger.dataflow
    rounded_box(ax, 3.76, 1.02, 2.72, 0.68, "white", TEAL, lw=0.9, radius=0.07, zorder=2)
    ax.text(
        5.12, 1.49, "ledger.dataflow",
        ha="center", va="center", fontsize=6.8, family="monospace", fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 1.21, "Byte-level writer -> reader flow",
        ha="center", va="center", fontsize=6.2, color=SLATE, zorder=3,
    )

    # Middle-bottom: Optional Nsight Compute L2 fills -> ledger.traffic
    rounded_box(ax, 3.76, 0.24, 2.72, 0.68, "white", TEAL, lw=0.9, radius=0.07, linestyle="--", zorder=2)
    ax.text(
        5.12, 1.49 - 0.78, "ledger.traffic",
        ha="center", va="center", fontsize=6.8, family="monospace", fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        5.12, 0.43, "+ optional Nsight Compute L2 fills",
        ha="center", va="center", fontsize=6.1, color=SLATE, zorder=3,
    )

    # Right-top: Dead-write classification
    rounded_box(ax, 7.12, 1.02, 2.68, 0.68, "white", ORANGE, lw=0.85, radius=0.07, zorder=2)
    ax.text(
        7.24, 1.49, "Dead-write classification",
        ha="left", va="center", fontsize=7.0, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        7.24, 1.21, "fixed / moving / varies",
        ha="left", va="center", fontsize=6.2, color=SLATE, zorder=3,
    )

    # Right-bottom: Pulled vs. needed traffic
    rounded_box(ax, 7.12, 0.24, 2.68, 0.68, "white", ORANGE, lw=0.85, radius=0.07, zorder=2)
    ax.text(
        7.24, 0.71, "Pulled vs. needed traffic",
        ha="left", va="center", fontsize=7.0, fontweight="bold", color=DARK, zorder=3,
    )
    ax.text(
        7.24, 0.43, "Extra bytes vs. slow kernel",
        ha="left", va="center", fontsize=6.2, color=SLATE, zorder=3,
    )

    # Bottom arrows
    arrow(ax, 3.06, 1.28, 3.76, 1.36, color=INDIGO, lw=1.15)
    arrow(ax, 3.06, 0.68, 3.76, 0.58, color=INDIGO, lw=1.15, linestyle="--")
    arrow(ax, 6.48, 1.36, 7.12, 1.36, color=TEAL_DARK, lw=1.2)
    arrow(ax, 6.48, 0.58, 7.12, 0.58, color=TEAL_DARK, lw=1.2)

    # Vertical dashed link showing regions.json also feeds deep path (routed cleanly at x=0.55)
    arrow(ax, 0.55, 2.48, 0.55, 1.70, color=BLUE, lw=0.95, linestyle=":")

    out_dir = Path(__file__).resolve().parent
    fig.tight_layout(pad=0.1)
    fig.savefig(out_dir / "pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

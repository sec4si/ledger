#!/usr/bin/env python3
"""Plot isolated GEMM latency vs. dirty bytes written to L2 before the launch.

Data source: case-studies/gb10-qwen3.8-27b/results.md (gemm_isolation.py, GB10).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Raw measurements from case-studies/gb10-qwen3.8-27b/results.md (mb << 20 => MiB)
DIRTY_MIB = np.array([0, 2, 4, 8, 12, 16, 24, 32], dtype=float)
LATENCY_US = np.array([147.5, 158.7, 169.0, 187.4, 197.6, 201.7, 213.2, 215.0], dtype=float)

# In-engine steady-state medians from report-031.md (batch 1, 40 steps)
ENGINE_OPROJ_US = 147.3
ENGINE_OUTPROJ_US = 194.6
GDN_STATE_MIB = 12.0  # 4 positions * 48 heads * 128 * 128 * 4 B = 12,582,912 B = 12.0 MiB
L2_SIZE_MIB = 24.0


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    # Main sweep curve
    ax.plot(
        DIRTY_MIB, LATENCY_US,
        marker="o", markersize=5, linewidth=1.8,
        color="#1f5a99", zorder=4,
        label="Isolated GEMM after $N$ MiB dirty write",
    )

    # Vertical reference: GB10 L2 capacity
    ax.axvline(L2_SIZE_MIB, color="#666666", linestyle="--", linewidth=1.0, zorder=2)
    ax.text(
        L2_SIZE_MIB + 0.5, 224.0, "GB10 L2 (24 MiB)",
        fontsize=8, color="#444444", va="top", ha="left",
    )

    # Vertical reference: GDN 4-position state write (12 MiB)
    ax.axvline(GDN_STATE_MIB, color="#d95f02", linestyle=":", linewidth=1.1, zorder=2)
    ax.text(
        GDN_STATE_MIB + 0.5, 224.0, "GDN state\n(12 MiB)",
        fontsize=7.8, color="#b84d00", va="top", ha="left",
    )

    # In-engine observed medians as markers/reference points
    ax.scatter(
        [0.0], [ENGINE_OPROJ_US],
        marker="D", s=40, facecolor="white", edgecolor="#2a9d8f",
        linewidth=1.4, zorder=5,
        label="In-engine attn o_proj (147.3 $\\mu$s)",
    )
    ax.scatter(
        [GDN_STATE_MIB], [ENGINE_OUTPROJ_US],
        marker="s", s=44, facecolor="#d95f02", edgecolor="white",
        linewidth=0.8, zorder=5,
        label="In-engine GDN out_proj (194.6 $\\mu$s, 12 MiB)",
    )

    # Key annotations (carefully placed to avoid overlap)
    ax.annotate(
        "147.5 $\\mu$s\n(clean L2)",
        xy=(0, 147.5), xytext=(2.8, 144.0),
        fontsize=7.8, color="#1f5a99",
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
    )
    ax.annotate(
        "197.6 $\\mu$s (+34%)",
        xy=(12, 197.6), xytext=(3.8, 206.5),
        fontsize=7.8, color="#1f5a99",
        arrowprops=dict(arrowstyle="->", color="#1f5a99", lw=0.8),
    )
    ax.annotate(
        "213.2 $\\mu$s\n(+45%, saturates)",
        xy=(24, 213.2), xytext=(24.8, 191.0),
        fontsize=7.8, color="#1f5a99",
        arrowprops=dict(arrowstyle="->", color="#1f5a99", lw=0.8),
    )

    ax.set_xlabel("Dirty bytes written to L2 before GEMM (MiB)")
    ax.set_ylabel("CUTLASS FP8 GEMM median ($\\mu$s)")
    ax.set_xlim(-1.2, 34.0)
    ax.set_ylim(138, 226)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.grid(axis="y", linestyle="-", linewidth=0.4, color="#e0e0e0", zorder=0)

    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#dddddd")

    out_dir = Path(__file__).resolve().parent
    fig.tight_layout()
    fig.savefig(out_dir / "dirty_l2_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

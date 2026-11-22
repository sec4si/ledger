#!/usr/bin/env python3
"""Plot per-kernel A/B comparison: per-token stores vs. deferred stores (mode 2).

Data source: case-studies/gb10-qwen3.8-27b/results.md
(Deferred stores, locked 2000 MHz clock, 40 steps, mean across alternating runs).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Per-kernel medians (us), averaged over the valid alternating runs in results.md:
# Batch 1: per-token (r1, r2), deferred (r1, r2)
# Batch 4: per-token (r2; r1 fell back to eager), deferred (r1, r2)
KERNELS = ["GDN decode", "GDN out_proj", "Attn o_proj\n(control)"]

B1_BASE = np.array([
    np.mean([40.7, 40.7]),
    np.mean([193.5, 195.2]),
    np.mean([149.4, 148.9]),
])
B1_DEFER = np.array([
    np.mean([27.4, 27.4]),
    np.mean([162.2, 162.0]),
    np.mean([148.6, 148.8]),
])

B4_BASE = np.array([270.8, 200.0, 149.2])
B4_DEFER = np.array([
    np.mean([97.0, 97.5]),
    np.mean([193.2, 193.5]),
    np.mean([149.0, 150.0]),
])


def draw_panel(ax, base: np.ndarray, defer: np.ndarray, title: str, show_ylabel: bool) -> None:
    x = np.arange(len(KERNELS))
    width = 0.34

    bars_base = ax.bar(
        x - width / 2, base, width,
        color="#7f8c9a", edgecolor="white", linewidth=0.6,
        label="Per-token stores (vLLM 0.31)", zorder=3,
    )
    bars_defer = ax.bar(
        x + width / 2, defer, width,
        color="#1f5a99", edgecolor="white", linewidth=0.6,
        label="Deferred stores", zorder=3,
    )

    for b_bar, d_bar, b_val, d_val in zip(bars_base, bars_defer, base, defer):
        ax.text(
            b_bar.get_x() + b_bar.get_width() / 2, b_val + 4,
            f"{b_val:.1f}", ha="center", va="bottom", fontsize=7.6, color="#444444",
        )
        ax.text(
            d_bar.get_x() + d_bar.get_width() / 2, d_val + 4,
            f"{d_val:.1f}", ha="center", va="bottom", fontsize=7.6, color="#1f5a99", fontweight="bold",
        )
        pct = (d_val - b_val) / b_val * 100.0
        top = max(b_val, d_val) + 22
        if abs(pct) >= 1.0:
            speedup = b_val / d_val
            ax.text(
                (b_bar.get_x() + d_bar.get_x() + width) / 2, top,
                f"{speedup:.2f}$\\times$ faster", ha="center", va="bottom", fontsize=7.8,
                color="#2e8b57", fontweight="bold",
            )
        else:
            ax.text(
                (b_bar.get_x() + d_bar.get_x() + width) / 2, top,
                "~0%", ha="center", va="bottom", fontsize=7.5, color="#666666",
            )

    ax.set_title(title, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(KERNELS)
    ax.set_ylim(0, 328)
    ax.grid(axis="y", linestyle="-", linewidth=0.4, color="#e0e0e0", zorder=0)
    if show_ylabel:
        ax.set_ylabel("Kernel median latency ($\\mu$s)\n(lower is better)")


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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)

    draw_panel(ax1, B1_BASE, B1_DEFER, "Batch 1 (12 MiB/layer < 24 MiB L2)", show_ylabel=True)
    draw_panel(ax2, B4_BASE, B4_DEFER, "Batch 4 (48 MiB/layer = 2$\\times$ L2)", show_ylabel=False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )

    out_dir = Path(__file__).resolve().parent
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "per_kernel_ab.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

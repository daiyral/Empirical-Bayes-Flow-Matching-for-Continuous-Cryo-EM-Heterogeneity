import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryoflow import paths
from config import FORWARDS, METHODS, M_VALUES, cell_config, run_dir

OUT = paths.REPORTS
TITLES = {"linear": r"Linear: $\mathcal{A}(x_i) = A_i x_i$",
          "sin": r"Non-linear: $\mathcal{A}(x_i) = sin(A_i x_i)$"}
LIM = 3.0


def load_cell(method, forward, M):
    d = run_dir(method, forward, M)
    path = os.path.join(d, "em_history.csv")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}; run train_two_moons.py --method {method} "
                         f"--forward {forward} --M {M} first")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    outers = [int(r["outer"]) for r in rows]
    swd = np.array([float(r["swd_2d"]) for r in rows])

    cfg = cell_config(method, forward, M)
    shown = cfg.N_outer
    if method == "ebfm":
        # EB-FM row: saved checkpoint at (or just before) the lowest-SWD outer iteration
        every = cfg.log_scatter_every
        shown = max(every, outers[int(np.nanargmin(swd))] // every * every)
    points = np.load(os.path.join(d, "scatter.npz"))[f"outer{shown:03d}"]
    return points, shown, swd[outers.index(shown)]


def build_figure(path):
    n_rows, n_M = len(METHODS), len(M_VALUES)
    n_cols = 2 * n_M
    # thin invisible spacer column between the linear and non-linear blocks
    fig = plt.figure(figsize=(1.6 * n_cols + 1.2, 1.6 * n_rows + 0.6))
    gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1.0] * n_M + [0.18] + [1.0] * n_M,
                          wspace=0.0, hspace=0.0)

    axes = np.empty((n_rows, n_cols + 1), dtype=object)
    for i, (method, label) in enumerate(METHODS):
        for fi, forward in enumerate(FORWARDS):
            for j, M in enumerate(M_VALUES):
                col = j + fi * (n_M + 1)
                ax = axes[i, col] = fig.add_subplot(gs[i, col])
                points, shown, swd = load_cell(method, forward, M)
                print(f"{label:<13}{forward:<7} M={M:<4} outer {shown:>3}  swd {swd:.4f}")
                ax.scatter(points[:, 0], points[:, 1], s=1.5, alpha=0.5, color="C1", linewidths=0)
                ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_linewidth(0.8)
                    sp.set_edgecolor("black")
                if i == 0:
                    ax.set_title(f"M = {M}", fontsize=15, pad=4)
                if j == 0 and fi == 0:
                    ax.text(0.04, 0.96, label, transform=ax.transAxes, fontsize=14,
                            ha="left", va="top", fontweight="bold",
                            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.5))

    fig.subplots_adjust(top=0.82, bottom=0.01, left=0.02, right=0.995)
    fig.canvas.draw()
    top_y = axes[0, 0].get_position().y1
    for fi, forward in enumerate(FORWARDS):
        x0 = axes[0, fi * (n_M + 1)].get_position().x0
        x1 = axes[0, fi * (n_M + 1) + n_M - 1].get_position().x1
        fig.text((x0 + x1) / 2, top_y + 0.085, TITLES[forward], ha="center", fontsize=18)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"})
    os.makedirs(OUT, exist_ok=True)
    build_figure(os.path.join(OUT, "figure_two_moons.png"))
    build_figure(os.path.join(OUT, "figure_two_moons.pdf"))

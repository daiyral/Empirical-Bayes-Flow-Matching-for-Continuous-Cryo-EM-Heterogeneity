import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryoflow import paths
from utils import METHODS, OBS, metrics_path, run_name

OUT = paths.REPORTS

# (metrics.json key, column label); precision = fidelity, recall = coverage
COLUMNS = [("emd_psnr", "EMD"), ("precision_psnr", "Fidelity"), ("recall_psnr", "Coverage"),
           ("emd_ssim", "EMD"), ("precision_ssim", "Fidelity"), ("recall_ssim", "Coverage")]
KEYS = [k for k, _ in COLUMNS]
GROUPS = [("PSNR", 3), ("SSIM", 3)]

# Numbers reported in the paper table (tab:mnist_results), in COLUMNS order.
PAPER = {
    50: {"diffusion": (13.75, 14.08, 14.49, 0.59, 0.60, 0.62),
         "ebfm_nopl": (13.75, 14.33, 14.48, 0.56, 0.58, 0.62),
         "ebfm": (15.67, 16.20, 16.35, 0.71, 0.73, 0.74)},
    100: {"diffusion": (14.72, 15.36, 15.42, 0.68, 0.70, 0.71),
          "ebfm_nopl": (14.39, 15.04, 15.55, 0.68, 0.70, 0.76),
          "ebfm": (15.37, 16.06, 16.13, 0.71, 0.73, 0.74)},
}
PAPER = {m: {t: dict(zip(KEYS, v)) for t, v in rows.items()} for m, rows in PAPER.items()}

CAPTION = (r"MNIST: quality of the learned EB-FM prior under linear random-projection "
           r"observations ($M{=}50$ and $M{=}100$ measurements per $28{\times}28$ image). "
           r"Metrics are computed between 10K unconditional samples from the prior and 10K "
           r"held-out images: EMD (Hungarian assignment), Fidelity (each "
           r"generated$\rightarrow$nearest GT), Coverage (each GT$\rightarrow$nearest "
           r"generated), each in PSNR and SSIM. Higher is better.")

FS = 13
FONT = "DejaVu Serif"
PAD = 0.16     # side padding inside the rules (inches)
ROW_H = 0.34   # single-line row height


def fetch(suffix=""):
    data = {}
    for m in OBS:
        data[m] = {}
        for tag, _ in METHODS:
            run = run_name(tag, m) + suffix
            path = metrics_path(run)
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}; run eval_mnist.py --run_name {run} first")
            with open(path) as f:
                metrics = json.load(f)
            data[m][tag] = {k: float(metrics[k]) for k in KEYS}
    return data


def tex_table(data):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % CAPTION,
         r"\label{tab:mnist_results}", r"\vspace{-2pt}", r"\small{",
         r"\begin{tabular}{ll%s}" % ("c" * len(COLUMNS)), r"\hline",
         " &  & " + " & ".join(r"\multicolumn{%d}{c}{\textbf{%s} $\uparrow$}" % (n, g)
                               for g, n in GROUPS) + r" \\",
         " ".join(r"\cline{%d-%d}" % (3 + sum(k for _, k in GROUPS[:i]),
                                      2 + sum(k for _, k in GROUPS[:i + 1]))
                  for i in range(len(GROUPS))),
         r"\textbf{Obs.} & \textbf{Method} & "
         + " & ".join(r"\textbf{%s}" % lbl for _, lbl in COLUMNS) + r" \\",
         r"\hline", ""]
    for m in OBS:
        L.append(r"\multirow{%d}{*}{%d}" % (len(METHODS), m))
        for tag, name in METHODS:
            L.append("& %s & " % name + " & ".join("%.2f" % data[m][tag][k] for k in KEYS) + r" \\")
        L += [r"\hline", ""]
    L += [r"\end{tabular}", r"}", r"\vspace{-2pt}", r"\end{table}"]
    return "\n".join(L)


def md_table(data):
    head = ["Obs.", "Method"] + [f"{g} {lbl}" for g, n in GROUPS
                                 for _, lbl in COLUMNS[:n]]
    rows = [head, ["---"] * 2 + [":--:"] * len(COLUMNS)]
    for m in OBS:
        for i, (tag, name) in enumerate(METHODS):
            rows.append([f"**{m}**" if i == 0 else "", name]
                        + [f"{data[m][tag][k]:.2f}" for k in KEYS])
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def _txt(ax, x, y, s, ha="center", weight="normal", size=FS):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight,
            fontfamily=FONT, linespacing=1.35)


def _rule(ax, y, W, lw=1.0, x0=None, x1=None):
    ax.plot([PAD if x0 is None else x0, W - PAD if x1 is None else x1], [y, y], color="k", lw=lw)


def png_table(data, title, path):
    W, x_obs, x_meth, x_body = 11.0, 0.22, 1.05, 3.20
    cw = (W - PAD - x_body) / len(COLUMNS)
    cols = [x_body + cw * (i + 0.5) for i in range(len(COLUMNS))]
    title_h, h1, h2 = 0.52, 0.40, 0.42
    block_h = ROW_H * len(METHODS)
    H = title_h + h1 + h2 + block_h * len(OBS) + 0.30

    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

    _txt(ax, W / 2, H - title_h / 2, title, size=FS + 2)
    y = H - title_h
    _rule(ax, y, W, lw=1.5)

    y -= h1 / 2
    start = 0
    for g, n in GROUPS:
        x0, x1 = x_body + cw * start, x_body + cw * (start + n)
        _txt(ax, (x0 + x1) / 2, y, g + " ↑", weight="bold")
        _rule(ax, y - h1 / 2 + 0.02, W, lw=0.8, x0=x0 + 0.08, x1=x1 - 0.08)
        start += n
    y -= h1 / 2

    y -= h2 / 2
    _txt(ax, x_obs, y, "Obs.", ha="left", weight="bold")
    _txt(ax, x_meth, y, "Method", ha="left", weight="bold")
    for x, (_, lbl) in zip(cols, COLUMNS):
        _txt(ax, x, y, lbl, weight="bold")
    y -= h2 / 2
    _rule(ax, y, W, lw=1.0)

    for m in OBS:
        _txt(ax, x_obs, y - block_h / 2, str(m), ha="left")
        for tag, name in METHODS:
            yc = y - ROW_H / 2
            _txt(ax, x_meth, yc, name, ha="left")
            for x, k in zip(cols, KEYS):
                _txt(ax, x, yc, f"{data[m][tag][k]:.2f}")
            y -= ROW_H
        _rule(ax, y, W, lw=1.0)
    _rule(ax, y, W, lw=1.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def print_comparison(data):
    print(f"{'Obs.':<5}{'Method':<14}" + "".join(f"{g + ' ' + lbl:>24}" for g, n in GROUPS
                                                  for _, lbl in COLUMNS[:n]))
    for m in OBS:
        for tag, name in METHODS:
            cells = [f"{data[m][tag][k]:.2f} ({PAPER[m][tag][k]:.2f}, {data[m][tag][k] - PAPER[m][tag][k]:+.2f})"
                     for k in KEYS]
            print(f"{m:<5}{name:<14}" + "".join(f"{c:>24}" for c in cells))
    print("cells: reproduced (paper, difference)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true",
                    help="Render the numbers reported in the paper instead of the local eval results")
    ap.add_argument("--suffix", default="",
                    help="Use eval results of runs mnist_<method>_m<M><suffix>")
    args = ap.parse_args()

    data = PAPER if args.paper else fetch(args.suffix)
    name = "mnist_paper" if args.paper else f"mnist{args.suffix}"
    title = "MNIST: quality of the learned prior" + ("  —  reported" if args.paper else "")
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}_metrics.json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(OUT, f"table_{name}.tex"), "w") as f:
        f.write(tex_table(data) + "\n")
    with open(os.path.join(OUT, f"table_{name}.md"), "w") as f:
        f.write(md_table(data) + "\n")
    png_table(data, title, os.path.join(OUT, f"table_{name}.png"))
    if not args.paper:
        print_comparison(data)
    print("wrote", OUT)

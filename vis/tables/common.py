import os

from cryoflow import paths
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# "project" resolves against your wandb default entity; override with "entity/project"
# via WANDB_ENTITY_PROJECT if the runs live in a different team/entity.
ENTITY_PROJECT = os.environ.get("WANDB_ENTITY_PROJECT", "cryo-em-flow-eval")
PROTEINS = paths.PROTEINS   # column order for every table
OUT = paths.REPORTS

KEYS = {
    "pw": "start_predicted_mean_pairwise_rmsd",
    "pw_gt": "start_ref_mean_pairwise_rmsd",
    "rmsf": "start_predicted_rmsf",
    "rmsf_gt": "start_ref_rmsf",
    "rmsf_r": "start_global_rmsf r",
    "emd": "emd_rmsd",
    "cov": "recall_rmsd",     # mean over GT of min RMSD to generated
    "fid": "precision_rmsd",  # mean over generated of min RMSD to GT
}

FS = 13
FONT = "DejaVu Serif"
PAD = 0.16     # side padding inside the rules (inches)
ROW_H = 0.34   # single-line row height


def _txt(ax, x, y, s, ha="center", weight="normal", size=FS):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight,
            fontfamily=FONT, linespacing=1.35)


def _rule(ax, y, W, lw=1.0, x0=None, x1=None):
    ax.plot([PAD if x0 is None else x0, W - PAD if x1 is None else x1], [y, y], color="k", lw=lw)


def _new_fig(W, H):
    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    return fig, ax


# ------------------------------------------------ transposed table: metrics across, methods down

# (key, png label, latex label, ground-truth key)
COLUMNS = [("emd", "EMD\nRMSD Å", r"EMD RMSD \AA", None),
           ("cov", "Coverage\nRMSD Å", r"Coverage RMSD \AA", None),
           ("fid", "Fidelity\nRMSD Å", r"Fidelity RMSD \AA", None),
           ("pw", "Pairwise\nRMSD Å", r"Pairwise RMSD \AA", "pw_gt"),
           ("rmsf", "RMSF Å", r"RMSF \AA", "rmsf_gt"),
           ("rmsf_r", "RMSF r", r"RMSF $r$", None)]

GROUPS = [("Distributional accuracy", 3), ("Flexibility", 3)]


def best_tag(k, data_p, gt, tags):
    if k == "rmsf_r":
        score = lambda v: -v
    elif k in ("pw", "rmsf"):
        score = lambda v: abs(v - gt)
    else:
        score = lambda v: v
    scores = {t: score(data_p[t][k]) for t in tags}
    lo = min(scores.values())
    winners = [t for t, s in scores.items() if abs(s - lo) < 1e-9]
    return winners[0] if len(winners) == 1 else None


def _win(data, p, tag, k, gk, methods):
    return best_tag(k, data[p], data[p].get("gt", {}).get(gk), [t for t, _ in methods]) == tag


def tex_transposed(data, methods, caption, label, tex_label=lambda l: l):
    L = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption,
         r"\label{%s}" % label, r"\vspace{-2pt}", r"\footnotesize{",
         r"\begin{tabular}{ll%s}" % ("c" * len(COLUMNS)), r"\hline",
         " &  & " + " & ".join(r"\multicolumn{%d}{c}{\textbf{%s}}" % (n, g) for g, n in GROUPS)
         + r" \\",
         " ".join(r"\cline{%d-%d}" % (3 + sum(m for _, m in GROUPS[:i]),
                                      2 + sum(m for _, m in GROUPS[:i + 1]))
                  for i in range(len(GROUPS))),
         " &  & " + " & ".join(lbl for _, _, lbl, _ in COLUMNS) + r" \\",
         r"\hline", ""]
    for p in PROTEINS:
        L.append(r"\multirow{%d}{*}{\textbf{%s}}" % (len(methods) + 1, p.replace("_", r"\_")))
        gt = ["%.2f" % data[p]["gt"][gk] if gk else "--" for _, _, _, gk in COLUMNS]
        L += ["& Ground truth", "& " + " & ".join(gt) + r" \\"]
        for tag, name in methods:
            cells = [(r"\textbf{%.2f}" if _win(data, p, tag, k, gk, methods) else "%.2f")
                     % data[p][tag][k] for k, _, _, gk in COLUMNS]
            L += ["& " + tex_label(name), "& " + " & ".join(cells) + r" \\"]
        L += [r"\hline", ""]
    L += [r"\end{tabular}", r"}", r"\vspace{-2pt}", r"\end{table}"]
    return "\n".join(L)


def md_transposed(data, methods):
    head = ["Protein", "Method"] + [lbl.replace("\n", " ") for _, lbl, _, _ in COLUMNS]
    rows = [head, ["---"] * 2 + [":--:"] * len(COLUMNS)]
    for p in PROTEINS:
        rows.append([f"**{p}**", "Ground truth"]
                    + [f"{data[p]['gt'][gk]:.2f}" if gk else "--" for _, _, _, gk in COLUMNS])
        for tag, name in methods:
            cells = []
            for k, _, _, gk in COLUMNS:
                v = f"{data[p][tag][k]:.2f}"
                cells.append(f"**{v}**" if _win(data, p, tag, k, gk, methods) else v)
            rows.append(["", name] + cells)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


def png_transposed(data, methods, title, path):
    W, x_prot, x_meth, x_body = 13.0, 0.22, 1.55, 4.60
    cw = (W - PAD - x_body) / len(COLUMNS)
    cols = [x_body + cw * (i + 0.5) for i in range(len(COLUMNS))]
    title_h, h1, h2 = 0.52, 0.40, 0.62
    block_h = ROW_H * (len(methods) + 1)
    H = title_h + h1 + h2 + block_h * len(PROTEINS) + 0.30
    fig, ax = _new_fig(W, H)

    _txt(ax, W / 2, H - title_h / 2, title, size=FS + 2)
    y = H - title_h
    _rule(ax, y, W, lw=1.5)

    y -= h1 / 2
    start = 0
    for g, n in GROUPS:
        x0, x1 = x_body + cw * start, x_body + cw * (start + n)
        _txt(ax, (x0 + x1) / 2, y, g, size=FS)
        _rule(ax, y - h1 / 2 + 0.02, W, lw=0.8, x0=x0 + 0.08, x1=x1 - 0.08)
        start += n
    y -= h1 / 2

    y -= h2 / 2
    for x, (_, lbl, _, _) in zip(cols, COLUMNS):
        _txt(ax, x, y, lbl, size=FS - 1)
    y -= h2 / 2
    _rule(ax, y, W, lw=1.0)

    for p in PROTEINS:
        _txt(ax, x_prot, y - block_h / 2, p, ha="left", weight="bold", size=FS + 1)
        yc = y - ROW_H / 2
        _txt(ax, x_meth, yc, "Ground truth", ha="left", size=FS - 1)
        for x, (_, _, _, gk) in zip(cols, COLUMNS):
            _txt(ax, x, yc, f"{data[p]['gt'][gk]:.2f}" if gk else "--", size=FS - 1)
        y -= ROW_H
        for tag, name in methods:
            yc = y - ROW_H / 2
            _txt(ax, x_meth, yc, name, ha="left")
            for x, (k, _, _, gk) in zip(cols, COLUMNS):
                _txt(ax, x, yc, f"{data[p][tag][k]:.2f}",
                     weight="bold" if _win(data, p, tag, k, gk, methods) else "normal")
            y -= ROW_H
        _rule(ax, y, W, lw=1.0)
    _rule(ax, y, W, lw=1.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def write_transposed(data, methods, name, title, caption, label, tex_label=lambda l: l):
    import json
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}_metrics.json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(OUT, f"table_{name}.tex"), "w") as f:
        f.write(tex_transposed(data, methods, caption, label, tex_label) + "\n")
    with open(os.path.join(OUT, f"table_{name}.md"), "w") as f:
        f.write(md_transposed(data, methods) + "\n")
    png_transposed(data, methods, title, os.path.join(OUT, f"table_{name}.png"))
    print("wrote", OUT)

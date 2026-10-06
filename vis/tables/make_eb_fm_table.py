import argparse
import json
import os

import matplotlib.pyplot as plt

from common import (KEYS, PROTEINS, ENTITY_PROJECT, OUT, FS, PAD, ROW_H,
                    _txt, _rule, _new_fig)

ROWS = [("Flexibility", "Pairwise RMSD Å", r"Pairwise RMSD \AA", "pw", "pw_gt"),
        (None, "RMSF Å", r"RMSF \AA", "rmsf", "rmsf_gt"),
        (None, "RMSF r", r"RMSF $r$", "rmsf_r", None),
        ("Distributional\naccuracy", "EMD RMSD Å", r"EMD RMSD \AA", "emd", None),
        (None, "Coverage RMSD Å", r"Coverage RMSD \AA", "cov", None),
        (None, "Fidelity RMSD Å", r"Fidelity RMSD \AA", "fid", None)]


def fetch(suffix):
    import wandb
    api = wandb.Api()
    data = {}
    for p in PROTEINS:
        name = p + suffix
        runs = [r for r in api.runs(ENTITY_PROJECT, filters={"display_name": name},
                                    order="-created_at") if r.state == "finished"]
        if not runs:
            raise SystemExit(f"missing finished eval run: {name}")
        data[p] = {k: float(runs[0].summary[v]) for k, v in KEYS.items()}
        data[p]["run_id"] = runs[0].id
    return data


def tex_table(data):
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{EB-FM results on four ATLAS targets.}",
         r"\label{tab:cryo_em_results}", r"\vspace{-2pt}", r"\small{",
         r"\begin{tabular}{llcccc}", r"\hline",
         " &  & " + " & ".join(r"\textbf{%s}" % p.replace("_", r"\_") for p in PROTEINS) + r" \\",
         r"\hline", "",
         r"\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}Flexibility\end{tabular}}"]
    for _, _, label, k, gk in ROWS:
        if k == "emd":
            L += [r"\hline", "",
                  r"\multirow{3}{*}{\begin{tabular}[c]{@{}l@{}}Distributional\\ accuracy\end{tabular}}"]
        cells = [f"{data[p][k]:.2f}" + (r" \footnotesize{(gt: %.2f)}" % data[p][gk] if gk else "")
                 for p in PROTEINS]
        L += ["& " + label, "& " + " & ".join(cells) + r" \\", ""]
    L += [r"\hline", r"\end{tabular}", r"}", r"\vspace{-2pt}", r"\end{table}"]
    return "\n".join(L)


def png_table(data, path):
    W, x_grp, x_met, x_body = 11.0, 0.22, 1.78, 3.85
    cw = (W - PAD - x_body) / len(PROTEINS)
    centers = [x_body + cw * i + cw / 2 for i in range(len(PROTEINS))]
    title_h, hdr_h = 0.50, 0.44
    H = title_h + hdr_h + ROW_H * len(ROWS) + 0.40
    fig, ax = _new_fig(W, H)

    _txt(ax, W / 2, H - title_h / 2, "EB-FM results", size=FS + 2)
    y = H - title_h
    _rule(ax, y, W, lw=1.5)
    y -= hdr_h / 2
    for c, p in zip(centers, PROTEINS):
        _txt(ax, c, y, p, weight="bold", size=FS + 1)
    y -= hdr_h / 2
    _rule(ax, y, W, lw=1.0)

    for i, (grp, label, _, k, gk) in enumerate(ROWS):
        yc = y - ROW_H / 2
        if grp:
            _txt(ax, x_grp, yc - (ROW_H / 2 if "\n" in grp else 0), grp, ha="left")
        _txt(ax, x_met, yc, label, ha="left")
        for c, p in zip(centers, PROTEINS):
            b = data[p]
            if gk:
                _txt(ax, c - 0.05, yc, f"{b[k]:.2f}", ha="right")
                _txt(ax, c + 0.05, yc, f"(gt: {b[gk]:.2f})", ha="left", size=FS - 2)
            else:
                _txt(ax, c, yc, f"{b[k]:.2f}")
        y -= ROW_H
        if i == 2:
            _rule(ax, y, W, lw=1.0)
    _rule(ax, y, W, lw=1.5)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_baseline",
                    help="eval runs are named <protein><suffix> in wandb")
    args = ap.parse_args()

    data = fetch(args.suffix)
    name = "eb_fm" if args.suffix == "_baseline" else f"eb_fm{args.suffix}"
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}_metrics.json"), "w") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(OUT, f"table_{name}.tex"), "w") as f:
        f.write(tex_table(data) + "\n")
    png_table(data, os.path.join(OUT, f"table_{name}.png"))
    print("wrote", OUT)

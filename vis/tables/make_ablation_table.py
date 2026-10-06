from common import KEYS, PROTEINS, ENTITY_PROJECT, write_transposed

# (tag, run suffix, row label). Eval runs are named <protein><suffix> in wandb.
RUNS = [("base", "_baseline", "EB-FM"),
        ("small", "_ctfpose_small", "EB-FM small-error-all"),
        ("err10", "_ctfpose_err10", "EB-FM p=0.1 error"),
        ("err30", "_ctfpose_err30", "EB-FM p=0.3 error")]

METHODS = [(t, l) for t, _, l in RUNS]

CAPTION = (r"Ablation summary on four ATLAS targets. EB-FM is the unperturbed baseline; "
           r"small-error-all applies a small pose/CTF error to every particle; "
           r"$p{=}0.1$ and $p{=}0.3$ randomise the metadata of that fraction of particles. "
           r"Best per protein in bold; for Pairwise RMSD and RMSF ``best'' means closest to "
           r"the ground truth.")


def fetch():
    import wandb
    api = wandb.Api()
    runs = {}
    for r in api.runs(ENTITY_PROJECT):
        runs.setdefault(r.name, r)
    data = {}
    for p in PROTEINS:
        data[p] = {}
        for tag, suffix, _ in RUNS:
            name = p + suffix
            if name not in runs:
                raise SystemExit(f"missing run: {name}")
            s = runs[name].summary
            d = {k: float(s[v]) for k, v in KEYS.items()}
            d["run_id"] = runs[name].id
            data[p][tag] = d
        data[p]["gt"] = {"pw_gt": data[p]["base"]["pw_gt"], "rmsf_gt": data[p]["base"]["rmsf_gt"]}
    return data


if __name__ == "__main__":
    write_transposed(fetch(), METHODS, "ablations",
                     "EB-FM ablations  —  best per protein in bold", CAPTION,
                     "tab:cryo_em_ablations", tex_label=lambda l: l.replace("=", "$=$"))

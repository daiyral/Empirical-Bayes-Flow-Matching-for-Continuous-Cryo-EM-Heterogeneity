import json
import os

from cryoflow import paths
from common import ENTITY_PROJECT, KEYS, PROTEINS, write_transposed

METHODS = [("ebfm", "EB-FM"), ("vae", "VAE"), ("uaf", "AF3 MSA sub"),
           ("gaf", "cryoDRGN-guided-AF3")]

# AlphaFold3 (Protenix) ensembles scored by eval/eval_{un,}guided_alphafold.py.
AF_DIRS = {"uaf": "unguided", "gaf": "guided"}

# run_suite reports these unprefixed, unlike the wandb eval runs.
AF_KEYS = {"pw": "predicted_mean_pairwise_rmsd", "rmsf": "predicted_rmsf",
           "rmsf_r": "global_rmsf r", "emd": "emd_rmsd", "cov": "recall_rmsd",
           "fid": "precision_rmsd"}

METRICS = tuple(AF_KEYS)

CAPTION = (r"Four configurations on four ATLAS targets. AF3 MSA sub is AlphaFold3 with MSA "
           r"subsampling; cryoDRGN-guided-AF3 is AlphaFold3 guided by cryoDRGN. Best per "
           r"protein in bold; for Pairwise RMSD and RMSF ``best'' means closest to the "
           r"ground truth.")


def fetch_vae(api, p):
    # logged by vae/eval_vae.py
    name = f"{p}_vae"
    runs = [r for r in api.runs(ENTITY_PROJECT, filters={"display_name": name},
                                order="-created_at") if r.state == "finished"]
    if not runs:
        raise SystemExit(f"missing finished eval run: {name}; run vae/eval_vae.py --protein {p} first")
    d = {k: float(runs[0].summary[KEYS[k]]) for k in METRICS}
    d["run_id"] = runs[0].id
    return d


def fetch_ebfm(api, p):
    # the plain baseline run, logged by eval/eval.py --run_name <protein>_baseline
    name = f"{p}_baseline"
    runs = [r for r in api.runs(ENTITY_PROJECT, filters={"display_name": name},
                                order="-created_at") if r.state == "finished"]
    if not runs:
        raise SystemExit(f"missing finished eval run: {name}; run eval/eval.py --run_name {name} first")
    d = {k: float(runs[0].summary[v]) for k, v in KEYS.items()}
    d["run_id"] = runs[0].id
    return d


def fetch():
    import wandb
    api = wandb.Api()
    data = {}
    for p in PROTEINS:
        ebfm = fetch_ebfm(api, p)
        data[p] = {"ebfm": {k: ebfm[k] for k in METRICS},
                   "gt": {"pw_gt": ebfm["pw_gt"], "rmsf_gt": ebfm["rmsf_gt"]},
                   "vae": fetch_vae(api, p)}
        for tag, sub in AF_DIRS.items():
            path = os.path.join(paths.ALPHAFOLD, sub, f"{p}_{sub}_alphafold", "metrics.json")
            if not os.path.exists(path):
                raise SystemExit(f"missing {path}; run eval/eval_{sub}_alphafold.py first")
            with open(path) as f:
                m = json.load(f)
            data[p][tag] = {k: float(m[v]) for k, v in AF_KEYS.items()}
    return data


if __name__ == "__main__":
    write_transposed(fetch(), METHODS, "compare_models",
                     "EB-FM vs. VAE vs. AlphaFold3 (MSA-subsampled and cryoDRGN-guided)"
                     "  —  best per protein in bold", CAPTION,
                     "tab:cryo_em_compare_models")

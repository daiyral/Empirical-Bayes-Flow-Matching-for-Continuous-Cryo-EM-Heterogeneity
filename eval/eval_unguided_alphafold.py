import argparse
import glob
import json
import os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "64")

import numpy as np
import torch
import wandb

from cryoflow import paths, utils
from ensemble_suite import default_topology, load_gt, load_structure_dir, run_suite


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble_dir", required=True)
    p.add_argument("--protein", required=True, choices=paths.PROTEINS)
    p.add_argument("--topology", default=None,
                   help="default: <cryo-dataset>/<protein>_sim_output/frames/frame_00000.pdb")
    p.add_argument("--outdir", default=None)
    p.add_argument("--num_samples", type=int, default=10001)
    p.add_argument("--run_name", default=None, help="default: <protein>_unguided_alphafold")
    p.add_argument("--wandb_project", default="cryo-em-flow-eval")
    p.add_argument("--wandb_mode", default="online")
    # Same threshold as eval_guided_alphafold.py: AF3 samples come out either slightly
    # stretched (max CA-CA 4.0-4.9) or snapped (8-60), nothing between.
    p.add_argument("--ca_break", type=float, default=5.0)
    p.add_argument("--lddt_n", type=int, default=0)
    return p.parse_args()


def load_ensemble(ensemble_dir, top_pdb):
    files = sorted(glob.glob(os.path.join(ensemble_dir, "sub_*.npz")))
    if not files:   # plain .pdb/.cif structures: no per-depth breakdown
        coords, n_files = load_structure_dir(ensemble_dir, top_pdb)
        return coords, None, n_files
    coords, depth = [], []
    for f in files:
        d = np.load(f)
        coords.append(d["coords"])
        depth.append(np.full(len(d["coords"]), int(d["msa_depth"])))
    return np.concatenate(coords).astype(np.float32), np.concatenate(depth), len(files)


def unbroken(coords, ca_break):
    ca = coords[:, 1::4, :]                    # N, CA, C, O per residue
    d = np.linalg.norm(ca[:, 1:] - ca[:, :-1], axis=-1)
    return d.max(axis=1) < ca_break


def depth_breakdown(utils, gen, depth, GT, device, n_gt=500, n_per_depth=200):
    gt = torch.as_tensor(GT[np.linspace(0, len(GT) - 1, n_gt).astype(int)], device=device)
    out = {}
    for d in sorted(set(depth.tolist())):
        idx = np.nonzero(depth == d)[0]
        if len(idx) > n_per_depth:
            idx = idx[np.linspace(0, len(idx) - 1, n_per_depth).astype(int)]
        g = torch.as_tensor(gen[idx], device=device)
        m = utils._rmsd_matrix_batched(gt, g)
        pw = utils._rmsd_matrix_batched(g, g)
        iu = torch.triu_indices(len(idx), len(idx), 1)
        out[f"depth_{d}_fidelity"] = float(m.min(0).values.mean())
        out[f"depth_{d}_coverage"] = float(m.min(1).values.mean())
        out[f"depth_{d}_pairwise"] = float(pw[iu[0], iu[1]].mean())
        out[f"depth_{d}_n"] = int(len(np.nonzero(depth == d)[0]))
    return out


def main():
    args = parse_args()
    p = args.protein
    args.topology = args.topology or default_topology(p)
    args.run_name = args.run_name or f"{p}_unguided_alphafold"
    args.outdir = args.outdir or os.path.join(paths.ALPHAFOLD, "unguided", args.run_name)
    os.makedirs(args.outdir, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_all, depth_all, n_files = load_ensemble(args.ensemble_dir, args.topology)
    keep = unbroken(gen_all, args.ca_break)
    print(f"{len(gen_all)} structures from {n_files} files; "
          f"{(~keep).sum()} broken ({100 * (~keep).mean():.1f}%) dropped")
    gen_all = gen_all[keep]
    assert len(gen_all) >= args.num_samples, \
        f"only {len(gen_all)} unbroken structures, need {args.num_samples}"
    # Evenly spaced, not a prefix: the ensemble is ordered by subsample index, which
    # cycles through the MSA depths, so a prefix would over-represent the early depths.
    pick = np.linspace(0, len(gen_all) - 1, args.num_samples).astype(int)
    gen = gen_all[pick]
    depth = None if depth_all is None else depth_all[keep][pick]

    GT = load_gt(p)
    gt = GT[:args.num_samples]
    assert gt.shape[1] == gen.shape[1], f"{gt.shape} GT vs {gen.shape} generated"

    ens = os.path.join(args.outdir, f"ensemble_{len(gen)}.npz")
    np.savez(ens, coords=gen, **({} if depth is None else {"msa_depth": depth}))
    print(f"scored ensemble -> {ens}")

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.config.update({"n_subsamples": n_files, "n_generated": int(len(keep)),
                         "n_broken": int((~keep).sum()), "n_scored": len(gen),
                         "method": "unguided_msa_subsampling"})
    if depth is not None:
        wandb.config.update({"msa_depths": sorted(set(depth.tolist()))})

    res = run_suite(gen, gt, "unguided_alphafold",
                    os.path.join(args.outdir, "unguided_alphafold"),
                    args.topology, device, args.lddt_n)
    res["broken_fraction"] = float((~keep).mean())
    res["n_subsamples"] = n_files
    if depth is not None:
        res.update(depth_breakdown(utils, gen, depth, GT, device))

    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    for k, v in res.items():
        wandb.summary[k] = v

    print("\n" + "=" * 60)
    for k in sorted(res):
        print(f"{k:<46}{res[k]:>12.4f}")
    print("=" * 60)
    wandb.finish()


if __name__ == "__main__":
    main()

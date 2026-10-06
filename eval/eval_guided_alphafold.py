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
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--run_name", default=None, help="default: <protein>_guided_alphafold")
    p.add_argument("--wandb_project", default="cryo-em-flow-eval")
    p.add_argument("--wandb_mode", default="online")
    # AF3 samples come out either slightly stretched (max CA-CA 4.0-4.9) or snapped
    # (8-60), nothing between, so 5.0 splits the two modes.
    p.add_argument("--ca_break", type=float, default=5.0)
    p.add_argument("--lddt_n", type=int, default=0)
    return p.parse_args()


def load_ensemble(ensemble_dir, top_pdb):
    files = sorted(glob.glob(os.path.join(ensemble_dir, "map_*.npz")))
    if not files:   # plain .pdb/.cif structures: no per-map steering metrics
        coords, n_files = load_structure_dir(ensemble_dir, top_pdb)
        return coords, None, n_files
    coords, src = [], []
    for f in files:
        d = np.load(f)
        coords.append(d["coords"])
        src.append(np.full(len(d["coords"]), int(d["map_index"])))
    return np.concatenate(coords).astype(np.float32), np.concatenate(src), len(files)


def steering(utils, gen, src, GT, device, n_maps=400):
    from scipy.stats import pearsonr
    ids = np.unique(src)[:n_maps]
    means = torch.stack([torch.tensor(gen[src == i].mean(0)) for i in ids]).to(device)
    gts = torch.tensor(GT[ids]).to(device)
    dg = utils._rmsd_matrix_batched(gts, gts)
    dm = utils._rmsd_matrix_batched(means, means)
    iu = torch.triu_indices(len(ids), len(ids), 1)
    x, y = dg[iu[0], iu[1]].cpu().numpy(), dm[iu[0], iu[1]].cpu().numpy()
    paired = utils._rmsd_matrix_batched(gts, means).diagonal()
    ref = torch.tensor(GT.mean(0)).to(device)[None].expand(len(ids), -1, -1)
    base = utils._rmsd_matrix_batched(gts, ref).diagonal()
    return {"map_steering_r": float(pearsonr(x, y)[0]),
            "map_pair_dist_gt": float(x.mean()), "map_pair_dist_gen": float(y.mean()),
            "per_map_paired_rmsd": float(paired.mean()),
            "gt_mean_baseline_rmsd": float(base.mean())}


def unbroken(coords, ca_break):
    ca = coords[:, 1::4, :]                    # N, CA, C, O per residue
    d = np.linalg.norm(ca[:, 1:] - ca[:, :-1], axis=-1)
    return d.max(axis=1) < ca_break


def main():
    args = parse_args()
    args.topology = args.topology or default_topology(args.protein)
    args.run_name = args.run_name or f"{args.protein}_guided_alphafold"
    args.outdir = args.outdir or os.path.join(paths.ALPHAFOLD, "guided", args.run_name)
    os.makedirs(args.outdir, exist_ok=True)
    os.environ["WANDB_MODE"] = args.wandb_mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_all, src_all, n_maps = load_ensemble(args.ensemble_dir, args.topology)
    keep = unbroken(gen_all, args.ca_break)
    print(f"{len(gen_all)} structures from {n_maps} files; "
          f"{(~keep).sum()} broken ({100 * (~keep).mean():.1f}%) dropped")
    gen_all = gen_all[keep]
    assert len(gen_all) >= args.num_samples, \
        f"only {len(gen_all)} unbroken structures, need {args.num_samples}"
    # evenly spaced, not a prefix: the ensemble is ordered by map file name
    gen = gen_all[np.linspace(0, len(gen_all) - 1, args.num_samples).astype(int)]

    GT = load_gt(args.protein)
    gt = GT[:args.num_samples]
    assert gt.shape[1] == gen.shape[1], f"{gt.shape} GT vs {gen.shape} generated"

    ens = os.path.join(args.outdir, f"ensemble_{len(gen)}.npz")
    np.savez(ens, coords=gen)
    print(f"scored ensemble -> {ens}")

    wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))
    wandb.config.update({"n_maps": n_maps, "n_generated": int(len(keep)),
                         "n_broken": int((~keep).sum()), "n_scored": len(gen)})

    res = run_suite(gen, gt, "guided_alphafold",
                    os.path.join(args.outdir, "guided_alphafold"),
                    args.topology, device, args.lddt_n)
    res["broken_fraction"] = float((~keep).mean())
    res["n_maps"] = n_maps
    if src_all is not None:
        res.update(steering(utils, gen_all, src_all[keep], GT, device))

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

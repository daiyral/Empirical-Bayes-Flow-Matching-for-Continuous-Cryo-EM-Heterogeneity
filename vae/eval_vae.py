import argparse
import json
import os

import MDAnalysis as mda
import numpy as np
import torch
import wandb

from cryoflow import paths, utils
from frames import base_frames_from_universe
from train_vae import run_dir
from vae_model import DanVAE


def rmsf(coords, ref):
    # per-atom RMS fluctuation about the ensemble mean, after rigid alignment to ref
    al = utils.rmsdalign(ref.unsqueeze(0).expand_as(coords), coords)
    return (al - al.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean(0).sqrt()


@torch.no_grad()
def main(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdb_file, xtc_file = paths.md_files(args.protein)
    u = mda.Universe(pdb_file, xtc_file)
    bf = base_frames_from_universe(u, dev)

    ckpt = args.checkpoint_path or os.path.join(run_dir(args.protein), "best.pt")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    model = DanVAE(bf, image_sigma=a["image_sigma"], beta_conf=a["beta_conf"],
                   w_center=a["w_center"], w_backbone=a["w_backbone"],
                   w_enm=a["w_enm"], enm_tol=a["enm_tol"]).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()

    # ground-truth ensemble: frames evenly spaced over the whole trajectory
    bb = u.select_atoms("backbone")
    n_traj = len(u.trajectory)
    want = set(np.linspace(0, n_traj - 1, min(args.num_samples, n_traj)).round().astype(int).tolist())
    gt = torch.stack([torch.tensor(bb.positions.copy(), dtype=torch.float32)
                      for i, _ in enumerate(u.trajectory) if i in want]).to(dev)

    # unconditional samples from the prior
    gen = model.sample_structures(args.num_samples).to(dev).float()

    # distributional accuracy (keys as in eval/eval.py's wandb summary, read by vis/tables)
    M = utils._rmsd_matrix_batched(gt, gen)
    res = {"emd_rmsd": utils.emd_rmsd(rmsd_matrix=M),            # Hungarian matching
           "recall_rmsd": utils.recall_rmsd(rmsd_matrix=M),      # coverage: GT -> nearest sample
           "precision_rmsd": utils.precision_rmsd(rmsd_matrix=M)}  # fidelity: sample -> nearest GT

    # flexibility: ensemble spread (evenly spaced subsample) and per-residue CA RMSF
    sub = min(512, args.num_samples)
    sel_gt = torch.linspace(0, gt.shape[0] - 1, sub).round().long().to(dev)
    sel_gen = torch.linspace(0, gen.shape[0] - 1, sub).round().long().to(dev)
    res["start_predicted_mean_pairwise_rmsd"] = float(utils._rmsd_matrix_batched(gen[sel_gen], gen[sel_gen]).mean())
    res["start_ref_mean_pairwise_rmsd"] = float(utils._rmsd_matrix_batched(gt[sel_gt], gt[sel_gt]).mean())
    ca = np.nonzero(bb.names == "CA")[0]
    f_gen, f_gt = rmsf(gen, gt[0]).cpu().numpy()[ca], rmsf(gt, gt[0]).cpu().numpy()[ca]
    res["start_predicted_rmsf"] = float(np.median(f_gen))
    res["start_ref_rmsf"] = float(np.median(f_gt))
    res["start_global_rmsf r"] = float(np.corrcoef(f_gen, f_gt)[0, 1])

    # bond health: % of C-N peptide bonds more than 0.3 A off the base conformation
    cn = torch.as_tensor(bb.names[bf["bond_i"].cpu().numpy()] == "C", device=dev)
    d = (gen[:, bf["bond_i"][cn], :] - gen[:, bf["bond_j"][cn], :]).norm(dim=-1)
    res["pct_cn_broken"] = 100.0 * float((torch.abs(d - bf["bond_target"][cn]) > 0.3).float().mean())

    print(f"===== vae {args.protein} (epoch {ck['epoch']}, n={args.num_samples}) =====")
    print(f"  EMD RMSD        {res['emd_rmsd']:8.3f}")
    print(f"  Coverage RMSD   {res['recall_rmsd']:8.3f}")
    print(f"  Fidelity RMSD   {res['precision_rmsd']:8.3f}")
    print(f"  Pairwise RMSD   {res['start_predicted_mean_pairwise_rmsd']:8.3f}   (GT {res['start_ref_mean_pairwise_rmsd']:.3f})")
    print(f"  RMSF            {res['start_predicted_rmsf']:8.3f}   (GT {res['start_ref_rmsf']:.3f})")
    print(f"  RMSF r          {res['start_global_rmsf r']:8.3f}")
    print(f"  C-N broken      {res['pct_cn_broken']:8.1f} %")

    info = dict(epoch=ck["epoch"], checkpoint=ckpt, num_samples=args.num_samples)
    out = paths.results(f"vae_{args.protein}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({**res, **info}, f, indent=2)
    print("wrote", os.path.join(out, "metrics.json"))

    # <protein>_<tag> like the EB-FM eval runs; vae_<protein> there are older inferred-pose runs
    wandb.init(project="cryo-em-flow-eval", name=f"{args.protein}_vae", config={**a, **info})
    wandb.log(res)
    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protein", type=str, required=True, choices=paths.PROTEINS)
    p.add_argument("--checkpoint_path", type=str, default=None,
                   help="default: runs/checkpoints/vae_<protein>/best.pt")
    p.add_argument("--num_samples", type=int, default=10000)
    main(p.parse_args())

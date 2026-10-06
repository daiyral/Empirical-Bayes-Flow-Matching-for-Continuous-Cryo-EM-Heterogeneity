import argparse
import os

import MDAnalysis as mda
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from cryoflow import paths
from cryoflow.datasets.rendered import TemSimulatorDataset
from cryoflow.utils import _rmsd_matrix_batched, rmsdalign
from frames import base_frames_from_universe
from vae_model import DanVAE

# (full_epochs, log_every_n_epochs) of the runs behind the paper's VAE column
SCHEDULE = {"1a62_A": (76, 25), "7r79_C": (70, 10), "1egw_C": (70, 10), "3rjs_A": (70, 10)}


def run_dir(protein):
    return os.path.join(paths.CHECKPOINTS, f"vae_{protein}")


@torch.no_grad()
def reconstruction_rmsd(model, images_all, gt_coords_all, device, batch_size=256):
    rmsds = []
    for i in range(0, images_all.shape[0], batch_size):
        gt = gt_coords_all[i:i + batch_size].to(device)
        pred = rmsdalign(gt, model.reconstruct(images_all[i:i + batch_size].to(device)))
        rmsds.append(torch.sqrt(((pred - gt) ** 2).sum(-1).mean(-1)))
    rmsds = torch.cat(rmsds).cpu().numpy()
    return {"rmsd_all_mean": float(np.mean(rmsds)), "rmsd_all_median": float(np.median(rmsds))}


@torch.no_grad()
def diversity_stats(model, device, n=256):
    # spread of the unconditional prior ensemble + bond health
    was_training = model.training
    gen = model.sample_structures(n).to(device).float()
    sub = min(128, n)
    pw = float(_rmsd_matrix_batched(gen[:sub], gen[:sub]).mean())
    d = (gen[:, model.bond_i, :] - gen[:, model.bond_j, :]).norm(dim=-1)
    broken = 100.0 * float((torch.abs(d - model.bond_target.unsqueeze(0)) > 0.3).float().mean())
    if was_training:
        model.train()
    return {"gen_pairwise_rmsd": pw, "pct_bonds_broken": broken}


def train(model, loader, opt, device, sigma_render, epochs, conf_samples, log_every,
          images_all, gt_coords_all, accum, save_ckpt, grad_clip=1.0):
    best_spread = -1.0
    for ep in range(epochs):
        model.train()
        agg = {}; nmic = 0
        opt.zero_grad()
        for imgs, angs in loader:                    # loader batch = micro_batch_size
            total, logd = model.compute_losses(imgs.to(device), angs.to(device),
                                               sigma_render, conf_samples)
            (total / accum).backward()               # gradient accumulation -> effective batch
            for k, v in logd.items():
                agg[k] = agg.get(k, 0.0) + v
            nmic += 1
            if nmic % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step(); opt.zero_grad()
        if nmic % accum != 0:                        # flush leftover micro-batches
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step(); opt.zero_grad()

        print(f"[ep {ep + 1}/{epochs}] " + " ".join(f"{k}={v / max(nmic, 1):.4f}" for k, v in agg.items()),
              flush=True)
        if ep % log_every == 0 or ep == epochs - 1:
            metrics = {**reconstruction_rmsd(model, images_all, gt_coords_all, device),
                       **diversity_stats(model, device)}
            print(f"    [eval ep {ep}] recon_rmsd_mean={metrics['rmsd_all_mean']:.4f} "
                  f"median={metrics['rmsd_all_median']:.4f} | "
                  f"gen_pairwise_rmsd={metrics['gen_pairwise_rmsd']:.4f} "
                  f"bonds_broken={metrics['pct_bonds_broken']:.1f}%", flush=True)
            save_ckpt(f"ep{ep}.pt", ep, metrics)
            # Model selection without ground truth: the ensemble spread rises, then collapses
            # as the encoder overfits; keep the checkpoint at its peak.
            if metrics["gen_pairwise_rmsd"] > best_spread:
                best_spread = metrics["gen_pairwise_rmsd"]
                save_ckpt("best.pt", ep, metrics)
                print(f"      ^ new best spread {best_spread:.4f} -> saved best.pt", flush=True)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    full_epochs, log_every = SCHEDULE[args.protein]
    full_epochs = args.full_epochs or full_epochs
    log_every = args.log_every_n_epochs or log_every

    pdb_file, xtc_file = paths.md_files(args.protein)
    universe = mda.Universe(pdb_file, xtc_file)
    base_frames = base_frames_from_universe(universe, device)

    ds = TemSimulatorDataset(paths.sim_dataset(args.protein), universe=universe)
    n_frames = min(len(ds), args.limit_frames) if args.limit_frames > 0 else len(ds)
    images_all = torch.stack([ds.frame_images[i][0].float() for i in range(n_frames)])
    gt_coords_all = torch.stack([ds.coords_3d[i].float() for i in range(n_frames)])
    # true viewing angles (ZYZ-intrinsic degrees, as cryojax expects)
    angles_all = torch.stack([ds.frame_angles[i][0].float() for i in range(n_frames)])
    print(f"Loaded {n_frames} frames, image {tuple(images_all.shape[1:])}, "
          f"n_atoms={gt_coords_all.shape[1]}, n_res={base_frames['n_res']}")

    # gradient accumulation: the loader yields micro-batches, accumulated to batch_size
    micro = min(args.micro_batch_size, args.batch_size)
    accum = max(1, args.batch_size // micro)
    loader = DataLoader(TensorDataset(images_all, angles_all), batch_size=micro,
                        shuffle=True, drop_last=True)

    model = DanVAE(base_frames, image_sigma=args.image_sigma, beta_conf=args.beta_conf,
                   w_center=args.w_center, w_backbone=args.w_backbone,
                   w_enm=args.w_enm, enm_tol=args.enm_tol).to(device)
    print(f"VAE params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.95))

    out = run_dir(args.protein)
    os.makedirs(out, exist_ok=True)

    def save_ckpt(name, epoch, metrics):
        torch.save({"state_dict": model.state_dict(), "args": vars(args),
                    "epoch": epoch, "metrics": metrics}, os.path.join(out, name))

    print(f"=== {args.protein}: {full_epochs} epochs, {args.conf_samples} conf samples, poses given ===")
    train(model, loader, opt, device, args.sigma, full_epochs, args.conf_samples, log_every,
          images_all, gt_coords_all, accum, save_ckpt)
    print(f"Saved checkpoints -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--protein", type=str, required=True, choices=paths.PROTEINS)
    p.add_argument("--full_epochs", type=int, default=0, help="0 = per-protein SCHEDULE")
    p.add_argument("--log_every_n_epochs", type=int, default=0, help="0 = per-protein SCHEDULE")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=8)      # per-forward chunk (grad accum)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--conf_samples", type=int, default=64)
    p.add_argument("--beta_conf", type=float, default=1.0)
    p.add_argument("--w_center", type=float, default=0.1)
    p.add_argument("--w_backbone", type=float, default=3000.0)
    p.add_argument("--w_enm", type=float, default=100.0,
                   help="elastic-network prior on base-conformation CA-CA contacts")
    p.add_argument("--enm_tol", type=float, default=1.0,
                   help="flat-bottom tolerance (A) of the elastic-network restraint")
    p.add_argument("--image_sigma", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=1.0)             # renderer Gaussian sigma
    p.add_argument("--limit_frames", type=int, default=0, help="0=all; else first N frames (quick tests)")
    p.add_argument("--seed", type=int, default=1,
                   help="the spread peak is a near-tie between eval epochs, so the selected "
                        "checkpoint is seed-dependent (seed 0: ep25 on 1a62_A, ep40 on 3rjs_A)")
    main(p.parse_args())

import glob
import os
import time

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

import MDAnalysis as mda
import wandb

from cryoflow import paths, utils


def load_gt(protein):
    # backbone coords of every simulated frame, centred on the particle's bounding box
    ds = paths.sim_dataset(protein)
    names = sorted(f for f in os.listdir(ds) if f.endswith(".npz"))
    coords = np.stack([np.load(os.path.join(ds, n))["coords_3d"].astype(np.float32) for n in names])
    centers = ((coords.min(1) + coords.max(1)) / 2).astype(np.float32)
    return (coords - centers[:, None, :]).astype(np.float32)


def default_topology(protein):
    return os.path.join(paths.CRYO_DATASET, f"{protein}_sim_output", "frames", "frame_00000.pdb")


def load_structure_dir(ensemble_dir, top_pdb):
    # every model of every .pdb/.cif file, reduced to the backbone atoms of top_pdb in its order
    import gemmi
    top = mda.Universe(top_pdb)
    layout = [(i, a.name) for i, res in enumerate(top.residues) for a in res.atoms]
    files = sorted(f for f in glob.glob(os.path.join(ensemble_dir, "*"))
                   if f.lower().endswith((".pdb", ".cif", ".mmcif")))
    assert files, f"no .pdb/.cif files under {ensemble_dir}"
    coords = []
    for f in files:
        for model in gemmi.read_structure(f):
            residues = [r for chain in model for r in chain if r.find_atom("CA", "*")]
            assert len(residues) == len(top.residues), \
                f"{f}: {len(residues)} residues, topology has {len(top.residues)}"
            coords.append([residues[i].find_atom(name, "*").pos.tolist() for i, name in layout])
    return np.asarray(coords, dtype=np.float32), len(files)


class CoordDataset:
    # stand-in for MDTrajectoryDataset, enough for utils.calculate_lddt_metrics
    def __init__(self, coords):
        self.coords = torch.as_tensor(coords, dtype=torch.float32)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        return self.coords[i], None

    def reverse_transform(self, x):
        return x


def write_traj(coords, top_pdb, out_pdb, out_xtc):
    u = mda.Universe(top_pdb)
    with mda.coordinates.XTC.XTCWriter(out_xtc, u.atoms.n_atoms) as W:
        for c in coords:
            u.atoms.positions = np.asarray(c, dtype=np.float32)
            W.write(u.atoms)
    u.atoms.positions = np.asarray(coords[0], dtype=np.float32)
    u.atoms.write(out_pdb)


def ca_indices(top_pdb):
    u = mda.Universe(top_pdb)
    return np.array([a.index for a in u.atoms if a.name == "CA"])


def trace_gif(gt, gen, ca, path, n_frames=60, fps=8):
    step = max(1, len(gt) // n_frames)
    gt, gen = gt[::step][:n_frames], gen[::step][:n_frames]
    lim = np.abs(np.concatenate([gt[:, ca], gen[:, ca]])).max() * 1.05
    fig = plt.figure(figsize=(9, 4.5))
    axes = [fig.add_subplot(1, 2, k + 1, projection="3d") for k in range(2)]
    lines = []
    for ax, ttl, col in zip(axes, ["ground truth", "generated"], ["tab:blue", "tab:orange"]):
        ax.set_title(ttl)
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim), ax.set_zlim(-lim, lim)
        ax.set_axis_off()
        lines.append(ax.plot([], [], [], lw=1.6, color=col)[0])

    def upd(i):
        for ln, data in zip(lines, (gt, gen)):
            p = data[i][ca]
            ln.set_data(p[:, 0], p[:, 1])
            ln.set_3d_properties(p[:, 2])
        for ax in axes:
            ax.view_init(elev=20, azim=i * 360 / len(gt))
        return lines

    animation.FuncAnimation(fig, upd, frames=len(gt), blit=False).save(
        path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    return path


def rmsf_plot(ref_rmsf, pred_rmsf, path, tag):
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(ref_rmsf, label="ground truth", color="tab:blue")
    ax.plot(pred_rmsf, label="generated", color="tab:orange")
    ax.set_xlabel("backbone atom"), ax.set_ylabel("RMSF (Å)")
    ax.set_title(f"per-atom RMSF ({tag})"), ax.legend()
    fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)
    return path


def run_suite(gen, gt, tag, outdir, top_pdb, device, lddt_n=0):
    # eval.py's metric suite for an ensemble that comes as plain coordinates
    os.makedirs(outdir, exist_ok=True)
    res = {}
    gen_t = torch.as_tensor(gen, dtype=torch.float32, device=device)
    gt_t = torch.as_tensor(gt, dtype=torch.float32, device=device)

    # pairwise-distance / macrostate populations
    try:
        p = os.path.join(outdir, "pairwise_distance_distribution.png")
        fig, info = utils.plot_pairwise_distance_distributions(
            gen_coords=gen_t, gt_coords=gt_t, save_path=p)
        plt.close(fig)
        wandb.log({f"{tag}/pairwise_distance_distribution": wandb.Image(p)})
        for k, lbl in enumerate("ABC"):
            res[f"macrostate_{lbl}_gt"] = info["gt_fractions"][k]
            res[f"macrostate_{lbl}_gen"] = info["gen_fractions"][k]
    except Exception as e:
        print(f"{tag}: pairwise distance plot failed: {e}")

    # precision / recall / EMD over the full RMSD matrix
    print(f"{tag}: RMSD matrix {gt_t.shape[0]} GT x {gen_t.shape[0]} gen ...", flush=True)
    rmsd_full = utils._rmsd_matrix_batched(gt_t, gen_t)
    res["precision_rmsd"] = utils.precision_rmsd(rmsd_matrix=rmsd_full)
    res["recall_rmsd"] = utils.recall_rmsd(rmsd_matrix=rmsd_full)
    rmsd_cpu = rmsd_full.cpu().numpy()
    del rmsd_full
    torch.cuda.empty_cache()

    from scipy.optimize import linear_sum_assignment
    print(f"{tag}: Hungarian matching ...", flush=True)
    t0 = time.time()
    row, col = linear_sum_assignment(rmsd_cpu)
    res["emd_rmsd"] = float(rmsd_cpu[row, col].mean())
    print(f"{tag}: Hungarian done in {time.time() - t0:.0f}s", flush=True)

    order = np.argsort(row)
    winners = col[order]

    # matched trajectory + mdtraj analysis
    aligned = utils.rmsdalign(gt_t, gen_t[torch.as_tensor(winners, device=device)])
    gt_pdb = os.path.join(outdir, "gt_backbone.pdb")
    gt_xtc = os.path.join(outdir, "gt_trajectory.xtc")
    gen_pdb = os.path.join(outdir, "generated_backbone.pdb")
    gen_xtc = os.path.join(outdir, "generated_trajectory.xtc")
    write_traj(gt, top_pdb, gt_pdb, gt_xtc)
    write_traj(aligned.cpu().numpy(), top_pdb, gen_pdb, gen_xtc)
    utils.save_paths(gt_pdb, gen_pdb, gen_xtc, gt_xtc)
    art = wandb.Artifact(f"{tag}_trajectories", type="trajectory")
    for f in (gt_pdb, gt_xtc, gen_pdb, gen_xtc):
        art.add_file(f)
    wandb.log_artifact(art)

    traj_res = utils.analyze_trajectories(ref_pdb_path=gt_pdb, ref_xtc_path=gt_xtc,
                                          gen_pdb_path=gen_pdb, gen_xtc_path=gen_xtc,
                                          bb_only=True, label=tag, wandb=wandb)
    for k, v in traj_res.items():
        res[k] = float(v)

    # per-atom RMSF curve (analyze_trajectories only keeps the median)
    import mdtraj
    ref_aa = mdtraj.load(gt_pdb)
    tr = mdtraj.load(gt_xtc, top=gt_pdb)
    ge = mdtraj.load(gen_xtc, top=gen_pdb)
    tr.superpose(ref_aa), ge.superpose(ref_aa)
    p = rmsf_plot(mdtraj.rmsf(tr, ref_aa) * 10, mdtraj.rmsf(ge, ref_aa) * 10,
                  os.path.join(outdir, "rmsf.png"), tag)
    wandb.log({f"{tag}/rmsf_per_atom": wandb.Image(p)})

    if lddt_n > 0:
        n = min(lddt_n, len(gen), len(gt))
        rng = np.random.default_rng(0)
        gi = rng.choice(len(gen), n, replace=False)
        print(f"{tag}: lDDT on {n} generated x {n} GT ...", flush=True)
        rec, prec, dpred, dgt = utils.calculate_lddt_metrics(
            generated=torch.as_tensor(gen[gi], dtype=torch.float32),
            dataset=CoordDataset(gt), n_gt_samples=n, gen_chunk=16, gt_chunk=16)
        res.update(lddt_recall=rec, lddt_precision=prec,
                   lddt_diversity_pred=dpred, lddt_diversity_gt=dgt)

    try:
        gif = trace_gif(gt, aligned.cpu().numpy(), ca_indices(top_pdb),
                        os.path.join(outdir, "trajectory.gif"))
        wandb.log({f"{tag}/compare_trajectory": wandb.Video(gif, format="gif")})
    except Exception as e:
        print(f"{tag}: gif failed: {e}")

    wandb.log({f"{tag}/{k}": v for k, v in res.items()})
    return res

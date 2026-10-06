import argparse
import torch
import wandb
import MDAnalysis as mda
from cryoflow.datasets.md import MDTrajectoryDataset
from cryoflow.models.flow_matching import CryoEMFlowMatching
from cryoflow import paths, utils


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CryoEMFlowMatching.load_from_checkpoint(
        args.checkpoint_path,
        pdb_file=args.pdb_file,
        xtc_file=args.xtc_file,
        data_dir=args.data_dir,
        latent_init_npz=None,  # the checkpoint already holds the trained latents
    )
    model = model.to(device)
    model.eval()

    wandb.init(project="cryo-em-flow-eval", name=args.run_name)

    u = mda.Universe(args.pdb_file, args.xtc_file)
    md_trajectory_dataset = MDTrajectoryDataset(
        u,
        scale=True,
        frame_indices=model.frame_indices
    )
    backbone = u.select_atoms('backbone')
    u.trajectory[0]
    initial_coords = torch.tensor(backbone.positions.copy(), device=device, dtype=torch.float32)
    other_coords_list = []
    for ts in u.trajectory[1:]:
        other_coords_list.append(torch.tensor(backbone.positions.copy(),
                                              device=device, dtype=torch.float32))
    u.trajectory[0]
    if other_coords_list:
        other_coords = torch.stack(other_coords_list)
        ref_batch = initial_coords.unsqueeze(0).expand(other_coords.shape[0], -1, -1)
        aligned_others = utils.rmsdalign(ref_batch, other_coords)
        initial_rmsd = torch.sqrt(((aligned_others - ref_batch) ** 2).sum(dim=-1).mean(dim=-1))
        initial_rmsd_mean = initial_rmsd.mean().item()
        print(f"initial_latent_rmsd: mean={initial_rmsd_mean:.4f} Å (over {initial_rmsd.shape[0]} frames)")
        wandb.log({"initial_latent_rmsd": initial_rmsd_mean})

    # print("Running eval on latent parametric structures (no sampling)...")
    latent_samples = model.parametric_structures.detach().to(device).float()

    gt_list = []
    for i in range(len(md_trajectory_dataset)):
        gt_sample, _ = md_trajectory_dataset[i]
        gt_list.append(md_trajectory_dataset.reverse_transform(gt_sample).to(device).float())
    gt_traj = torch.stack(gt_list)
    n = min(latent_samples.shape[0], gt_traj.shape[0])
    aligned = utils.rmsdalign(gt_traj[:n], latent_samples[:n])
    per_frame_rmsd = torch.sqrt(((aligned - gt_traj[:n]) ** 2).sum(dim=-1).mean(dim=-1))
    mean_rmsd = per_frame_rmsd.mean().item()
    print(f"latent direct per-frame RMSD: mean={mean_rmsd:.4f} Å")
    wandb.log({"latent_direct_per_frame_rmsd_mean": mean_rmsd})

    # utils.run_analysis(None, md_trajectory_dataset, args.pdb_file, device=device,
    #                    samples=latent_samples, skip_matching=True, prefix="latent",
    #                    compute_lddt=args.compute_lddt)
    print("Running eval on unguided samples...")
    # Per-run output dir: 8 concurrent evals must not share PDB/XTC/gif filenames.
    output_dir = args.output_dir or paths.results(args.run_name)
    utils.run_analysis(model.flow_model, md_trajectory_dataset, args.pdb_file, device=device,
                       sample_scaler=model.coords_scaler, num_samples=args.num_samples,
                       N=args.N, compute_lddt=args.compute_lddt, output_dir=output_dir)
    # print("Running eval on guided samples...")
    # utils.run_guided_analysis(model, md_trajectory_dataset, device=device,
    #                           num_samples=args.num_samples, N=args.N,
    #                           compute_lddt=args.compute_lddt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--pdb_file', type=str, default=paths.md_files("1a62_A")[0])
    parser.add_argument('--xtc_file', type=str, default=paths.md_files("1a62_A")[1])
    parser.add_argument('--data_dir', type=str, default=paths.sim_dataset("1a62_A"))
    parser.add_argument('--run_name', type=str, default="eval")
    parser.add_argument('--compute_lddt', type=lambda x: str(x).lower() == 'true', default=False)
    parser.add_argument('--num_samples', type=int, default=10_000)
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where analysis PDB/XTC/gif artifacts go (default: runs/results/<run_name>)')
    args = parser.parse_args()
    main(args)


import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import DataLoader
from cryoflow.datasets.md import ParametricStructureDataset, MDTrajectoryDataset
from cryoflow.models.flow_matching import CryoEMFlowMatching
from cryoflow import paths, utils
import MDAnalysis as mda
torch.set_float32_matmul_precision('medium')
# torch>=2.9 extends 'medium' to CPU matmul (bf16), which corrupts the CPU-side cdist
# building PairwiseDistanceRegularizer.init_dists. torch 2.4 (original runs) did not.
torch.backends.mkldnn.matmul.fp32_precision = 'ieee'
def main(args):
    model = CryoEMFlowMatching(
        pdb_file=args.pdb_file,
        xtc_file=args.xtc_file,
        data_dir=args.data_dir,
        flow_model_path=args.flow_model_path,
        num_frames=args.num_frames,
        use_first_frame_initial=args.use_first_frame_initial,
        flow_weight=args.flow_weight,
        render_weight=args.render_weight,
        structure_lr=args.structure_lr,
        flow_lr=args.flow_lr,
        render_loss_type=args.render_loss_type,
        log_every_n_epochs=args.log_every_n_epochs,
        sigma=args.sigma,
        pairwise_distance_weight=args.pairwise_distance_weight,
        images_per_frame=args.images_per_frame,
        guided_N=args.guided_N,
        guided_N_min=args.guided_N_min,
        guided_N_rampup_retrains=args.guided_N_rampup_retrains,
        guided_warmup_steps=args.guided_warmup_steps,
        guided_strength=args.guided_strength,
        guided_n_candidates=args.guided_n_candidates,
        warmup_epochs=args.warmup_epochs,
        flow_warmup_epochs=args.flow_warmup_epochs,
        flow_train_epochs=args.flow_train_epochs,
        flow_retrain_every=args.flow_retrain_every,
        flow_train_batch_size=args.flow_train_batch_size,
        inverse_crime=args.inverse_crime,
        pose_error_frac=args.pose_error_frac,
        ctf_error_frac=args.ctf_error_frac,
        pose_error_deg=args.pose_error_deg,
        ctf_defocus_error=args.ctf_defocus_error,
        error_seed=args.error_seed,
        latent_init_npz=args.latent_init_npz,
        latent_init_recenter=args.latent_init_recenter,
    )
    
    run_id = args.run_name or f"run_{int(torch.randint(0, 100000, (1,)).item())}"
    ckpt_dir = os.path.join(paths.CHECKPOINTS, run_id)
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="cryo_em_flow-{epoch:02d}-{train_loss:.4f}",
            every_n_epochs=args.log_every_n_epochs,
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        # EarlyStopping(
        #     monitor="train_combined_loss",
        #     mode="min",
        #     patience=args.early_stopping_patience,
        #     min_delta=args.early_stopping_min_delta,
        #     verbose=False
        # )
    ]
    
    loggers = None
    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        import wandb

        os.makedirs(ckpt_dir, exist_ok=True)
        run_id_path = os.path.join(ckpt_dir, "wandb_run_id.txt")
        resuming = bool(args.resume_from_checkpoint) and os.path.exists(args.resume_from_checkpoint)
        wandb_id = None
        wandb_resume = None
        if resuming and args.wandb_run_id:
            wandb_id = args.wandb_run_id
            wandb_resume = "must"
            with open(run_id_path, "w") as f:
                f.write(wandb_id)
            print(f"Resuming wandb run id={wandb_id} (from --wandb_run_id)")
        elif resuming and os.path.exists(run_id_path):
            with open(run_id_path, "r") as f:
                wandb_id = f.read().strip()
            wandb_resume = "must"
            print(f"Resuming wandb run id={wandb_id}")
        else:
            wandb_id = wandb.util.generate_id()
            with open(run_id_path, "w") as f:
                f.write(wandb_id)

        wandb_logger = WandbLogger(
            project="cryo-em-final",
            name=args.run_name,
            log_model=True,
            id=wandb_id,
            resume=wandb_resume,
        )
        #wandb_logger.watch(model, log="all")
        loggers = wandb_logger

        import glob as _glob
        source_code_artifact = wandb.Artifact("source_code", type="code")
        for py_file in _glob.glob(os.path.join(paths.ROOT, "src", "**", "*.py"),
                                  recursive=True) + [os.path.abspath(__file__)]:
            # name= keeps the tree structure; several packages have an __init__.py
            source_code_artifact.add_file(py_file,
                                          name=os.path.relpath(py_file, paths.ROOT))
        wandb_logger.experiment.log_artifact(source_code_artifact)

        # wandb_logger.experiment.log_code(
        #     ".",
        #     include_fn=lambda path: path.endswith(".py"),
        #     exclude_fn=lambda path: "wandb/" in path or "cryoflowvenv/" in path,
        # )
        print("All Python source code files saved to wandb")

    pl_trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,  # Add wandb logger
        accelerator='auto',
        devices=1,
        log_every_n_steps=1, 
        gradient_clip_val=args.gradient_clip_val,
    )
    
    # Check for checkpoint resuming
    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        else:
            print(f"Checkpoint file not found: {args.resume_from_checkpoint}")
    else:
        print("Starting new training run")
    
    dataloader = DataLoader(
        model.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    pl_trainer.fit(model, dataloader, ckpt_path=args.resume_from_checkpoint)

    
    # Save the trained parametric structures
    print("Saving trained parametric structures...")
    save_dir = paths.LATENTS
    os.makedirs(save_dir, exist_ok=True)
    
    # Save all denormalized coordinates
    trained_coords_tensor = model.parametric_structures.detach().cpu()

    parametric_structures_path = os.path.join(save_dir, f'{args.run_name}_parametric_structures.pt')
    torch.save({
        'coords': trained_coords_tensor,
        'frame_indices': model.frame_indices,
        'pdb_file': args.pdb_file,
        'xtc_file': args.xtc_file,
        'n_residues': trained_coords_tensor.shape[1]
    }, parametric_structures_path)

    
    print(f"Saved {len(trained_coords_tensor)} parametric structures to {save_dir}")
    print(f"Structure shape: {trained_coords_tensor.shape}")

    # Compute final mean RMSD
    import numpy as np
    from cryoflow.utils import rmsdalign
    rmsds = []
    for i, frame_idx in enumerate(model.frame_indices):
        model.universe.trajectory[frame_idx]
        gt_coords = torch.tensor(model.dataset.backbone_atoms.positions, dtype=torch.float32)
        pred_coords = trained_coords_tensor[i]
        pred_aligned = rmsdalign(gt_coords.unsqueeze(0), pred_coords.unsqueeze(0)).squeeze(0)
        rmsd = torch.sqrt(torch.mean(torch.sum((pred_aligned - gt_coords) ** 2, dim=-1))).item()
        rmsds.append(rmsd)
    mean_rmsd = np.mean(rmsds)
    print(f"FINAL_MEAN_RMSD={mean_rmsd:.6f}")

    # print("Running post-training analysis...")
    if args.use_wandb and args.flow_weight > 0 and args.post_train_analysis:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            # print(f"Loading parametric structures from {parametric_structures_path}")
        
            # # Load saved parametric structures
            checkpoint = torch.load(parametric_structures_path)
            parametric_coords = checkpoint['coords']
            
            latent_coords = ParametricStructureDataset(
                parametric_coords,
                pdb_file=args.pdb_file,
                scale=False
            )
            u = mda.Universe(args.pdb_file, args.xtc_file)
        
            frame_indices = model.frame_indices
            md_trajectory_dataset = MDTrajectoryDataset(
                u,
                scale=False,
                frame_indices=frame_indices
            )
            utils.create_latent_gif(latent_coords, md_trajectory_dataset)

            u = mda.Universe(args.pdb_file, args.xtc_file)
        
            md_trajectory_dataset = MDTrajectoryDataset(
                u,
                scale=True,
                frame_indices=frame_indices
            )
            print("Running eval on unguided samples...")
            utils.run_analysis(model.flow_model, md_trajectory_dataset, args.pdb_file, device=device,
                               sample_scaler=model.coords_scaler,
                               output_dir=paths.results(args.run_name))
        except Exception as e:
            print(f"Warning: Post-training analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("Training done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a Cryo-EM Flow Matching model with parameterized structures')
    
    parser.add_argument('--pdb_file', type=str, default=paths.md_files("1egw_C")[0], help='Path to PDB file')
    parser.add_argument('--xtc_file', type=str, default=paths.md_files("1egw_C")[1], help='Path to XTC trajectory file')
    parser.add_argument('--flow_model_path', type=str, default=None, help='Path to pretrained flow model')
    parser.add_argument('--epochs', type=int, default=3000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--flow_lr', type=float, default=2e-4, help='Learning rate for flow model (scaled for bs=256 with sqrt rule)')
    parser.add_argument('--structure_lr', type=float, default=0.1, help='Learning rate for parametric structures')
    parser.add_argument('--num_frames', type=int, default=10000, help='Number of parameterized frames to optimize')
    parser.add_argument('--run_name', type=str, default='cryo_em_flow_parameterized', help='Name for the wandb run')
    parser.add_argument('--use_wandb', type=bool, default=True, help='Whether to use wandb for logging')
    parser.add_argument('--use_first_frame_initial', type=bool, default=True, help='Have the first frame as initial structure')
    parser.add_argument('--flow_weight', type=float, default=0.8, help='Weight for flow matching loss')
    parser.add_argument('--render_weight', type=float, default=0.8, help='Weight for rendering loss')
    parser.add_argument('--render_loss_type', type=str, default='mae',
                        choices=['mse', 'mae', 'ncc'],
                        help='Type of rendering loss function')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to checkpoint to resume training from')
    parser.add_argument('--wandb_run_id', type=str, default=None, help='Existing wandb run id to resume (overrides saved id). Find in wandb URL or wandb/run-*/ dir.')
    parser.add_argument('--log_every_n_epochs', type=int, default=20, help='Log visualizations and RMSD statistics every n epochs')
    parser.add_argument('--early_stopping_patience', type=int, default=5000, help='Early stopping patience - number of epochs to wait without improvement')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.003, help='Early stopping min delta - minimum change required to qualify as improvement')
    parser.add_argument('--sigma', type=float, default=1, help='Sigma for Gaussian projection in renderer')
    parser.add_argument('--pairwise_distance_weight', type=float, default=0.3, help='Weight for pairwise distance regularization (preserves initial structure shape)')
    parser.add_argument('--images_per_frame', type=int, default=1, help='Number of pre-rendered images per frame (only used if use_rendered_dataset=True)')
    parser.add_argument('--guided_N', type=int, default=10, help='Max number of guided sampling steps (after flow model matures)')
    parser.add_argument('--guided_N_min', type=int, default=3, help='Starting guided sampling steps when flow model is weak')
    parser.add_argument('--guided_N_rampup_retrains', type=int, default=5, help='Number of flow retrains to ramp guided_N from min to max')
    parser.add_argument('--guided_warmup_steps', type=int, default=3, help='Warmup steps before applying guidance')
    parser.add_argument('--guided_strength', type=float, default=0.5, help='Guidance strength')
    parser.add_argument('--guided_n_candidates', type=int, default=5, help='Number of candidate priors to try')
    parser.add_argument('--data_dir', type=str, default=paths.sim_dataset("1egw_C"), help='Path to TEM-simulator dataset dir (from tools/simulate_cryo_em.py)')
    parser.add_argument('--warmup_epochs', type=int, default=200, help='Structure-only epochs before any flow training')
    parser.add_argument('--flow_warmup_epochs', type=int, default=50, help='Flow model epochs on first (warmup) training')
    parser.add_argument('--flow_train_epochs', type=int, default=1, help='Flow model epochs for subsequent retrainings')
    parser.add_argument('--flow_retrain_every', type=int, default=1, help='Retrain flow model every N main-training epochs after warmup')
    parser.add_argument('--flow_train_batch_size', type=int, default=256, help='Batch size for flow model training on latents')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--inverse_crime', action='store_true', help='Use RenderedMDTrajectoryDataset instead of TemSimulatorDataset')
    parser.add_argument('--pose_error_frac', type=float, default=0.0, help='Fraction of particles with a misassigned orientation')
    parser.add_argument('--ctf_error_frac', type=float, default=0.0, help='Fraction of particles with a misestimated defocus')
    parser.add_argument('--pose_error_deg', type=float, default=0.0, help='Pose error magnitude in degrees (0 = uniformly random orientation)')
    parser.add_argument('--ctf_defocus_error', type=float, default=0.4, help='Max relative defocus error for corrupted particles')
    parser.add_argument('--error_seed', type=int, default=0, help='Seed for the pose/CTF corruption draw')
    parser.add_argument('--post_train_analysis', type=lambda x: str(x).lower() == 'true', default=True, help='Run the in-train post-training analysis (skip when eval.py runs separately)')
    parser.add_argument('--latent_init_npz', type=str, default=None, help='npz with coords [n_frames, n_atoms, 3] to initialize the latents from (e.g. cryoDRGN fits from thesis/vol_fit.py). Overrides --use_first_frame_initial')
    parser.add_argument('--latent_init_recenter', type=str, default='frame0', choices=['frame0', 'none'], help='Put the loaded latents back in the trajectory frame by matching frame 0s bbox midpoint')

    args = parser.parse_args()
    main(args) 
import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import wandb
from torch.utils.data import Dataset, DataLoader
import MDAnalysis as mda

from cryoflow.models.flow_matching import FlowModel
from cryoflow.datasets.md import MDTrajectoryDataset, ParametricStructureDataset
from cryoflow import paths
from cryoflow.utils import run_analysis

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CoordsOnlyDataset(Dataset):
    def __init__(self, md_dataset):
        self.md_dataset = md_dataset
    def __len__(self):
        return len(self.md_dataset)
    def __getitem__(self, idx):
        coords, _ = self.md_dataset[idx]
        return coords

def main(args):
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config=vars(args),
            save_code=True
        )

    # Check if we should load from parametric structures
    if args.use_parametric_structures and os.path.exists(args.parametric_structures_path):
        print(f"Loading parametric structures from {args.parametric_structures_path}")
        
        # Load saved parametric structures
        checkpoint = torch.load(args.parametric_structures_path)
        parametric_coords = checkpoint['coords']
        
        md_trajectory_dataset = ParametricStructureDataset(
            parametric_coords,
            pdb_file=args.pdb_file,
            scale=args.scale
        )
        
        n_residue = parametric_coords.shape[1]
        print(f"Loaded {len(parametric_coords)} parametric structures with {n_residue} residues each")
        
    else:
        print("Loading from MD trajectory...")
        u = mda.Universe(args.pdb_file, args.xtc_file)
        
        frame_indices = list(range(0, len(u.trajectory), 100))
        md_trajectory_dataset = MDTrajectoryDataset(
            u,
            scale=args.scale,
            frame_indices=frame_indices
        )
        n_residue = len(md_trajectory_dataset.backbone_atoms)
        
    
    coords_only_dataset = CoordsOnlyDataset(md_trajectory_dataset)
    
    dataloader = DataLoader(
        coords_only_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )


    # Initialize model with current arguments. This will be used if loading fails.
    model = FlowModel(
        n_residue=n_residue,
        channels_data=3,
        layers=args.flow_layers,
        dim=args.flow_dim,
        heads=args.flow_heads,
        lr=args.flow_lr 
    )


    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="flow-model-{epoch:02d}",
        save_top_k=-1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [checkpoint_callback, lr_monitor]

    if args.use_wandb:
        from pytorch_lightning.loggers import WandbLogger
        wandb_logger = WandbLogger(project="flow-matching-model", name=args.run_name, log_model=True)
        loggers = wandb_logger
    
    pl_trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        logger=loggers,
        accelerator='auto',
        devices=1,
        gradient_clip_val=args.gradient_clip_val,
    )
    
    print("Starting FlowModel training...")
    pl_trainer.fit(model, dataloader)
    print("FlowModel training done!")
    torch.save(model.state_dict(), args.checkpoint_dir + "/flow-model.pt")

    print("Running post-training analysis...")
    if args.use_wandb:
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            u = mda.Universe(args.pdb_file, args.xtc_file)
        
            frame_indices = list(range(0, len(u.trajectory), 100))
            md_trajectory_dataset = MDTrajectoryDataset(
                u,
                scale=args.scale,
                frame_indices=frame_indices
            )
            run_analysis(model, md_trajectory_dataset, args.pdb_file, device=device)
        except Exception as e:
            print(f"Warning: Post-training analysis failed: {e}")
            import traceback
            traceback.print_exc()
    

    #torch.save(model.state_dict(), args.checkpoint_dir + "/flow-model.pt")
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a FlowModel for protein structures')
    
    parser.add_argument('--pdb_file', type=str, default=paths.md_files("1a62_A")[0], help='Path to PDB template file')
    parser.add_argument('--xtc_file', type=str, default=paths.md_files("1a62_A")[1], help='Path to XTC trajectory file')
    
    # New arguments for parametric structures
    parser.add_argument('--use_parametric_structures', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='Use saved parametric structures instead of MD trajectory')
    parser.add_argument('--parametric_structures_path', type=str, default=os.path.join(paths.LATENTS, 'parametric_structures.pt'),
                        help='Path to saved parametric structures')
    
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--flow_lr', type=float, default=1e-4, help='Learning rate for FlowModel')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--flow_layers', type=int, default=3, help='Number of Transformer blocks in FlowModel')
    parser.add_argument('--flow_dim', type=int, default=256, help='Dimension of embeddings in FlowModel')
    parser.add_argument('--flow_heads', type=int, default=8, help='Number of attention heads in FlowModel')

    parser.add_argument('--checkpoint_dir', type=str, default="flow_model_checkpoints", help='Directory to save model checkpoints')

    parser.add_argument('--wandb_project', type=str, default='flow-matching-model', help='W&B project name')
    parser.add_argument('--run_name', type=str, default='flow_model_training_run', help='Name for the W&B run')
    parser.add_argument('--use_wandb', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to use W&B for logging')
    parser.add_argument('--scale', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to scale coordinates')

    args = parser.parse_args()
    main(args) 
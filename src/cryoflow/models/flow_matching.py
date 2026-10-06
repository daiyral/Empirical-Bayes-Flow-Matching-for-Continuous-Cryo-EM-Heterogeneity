import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
import MDAnalysis as mda
import wandb
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from cryoflow.datasets.md import HarmonicPrior, StandardScaler, ParametricStructureDataset
from cryoflow.datasets.rendered import RenderedMDTrajectoryDataset, TemSimulatorDataset, CorruptedTemSimulatorDataset
from cryoflow.datasets.md import MDTrajectoryDataset
from cryoflow.losses import  FlowMatchingLoss, RenderingLoss, PairwiseDistanceRegularizer
from cryoflow.rendering.cryojax import CryoJAXRenderer
import py3Dmol
from cryoflow.utils import log_py3Dmol_to_wandb , rmsdalign
from cryoflow import paths
import tempfile
import os
import plotly.graph_objects as go
import plotly.offline as pyo


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        
        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = Attention(dim, heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x



class FlowModel(pl.LightningModule):
    def __init__(self, n_residue, channels_data=3, layers=3, dim=256, heads=8, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.dim = dim

        self.pos_embedding = nn.Parameter(torch.randn(1, n_residue, dim))
        self.input_proj = nn.Linear(channels_data, dim)
        self.time_proj = nn.Linear(dim, dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads) for _ in range(layers)
        ])

        self.out_proj = nn.Linear(dim, channels_data)
        self.n_residue = n_residue
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.prior = HarmonicPrior(N=n_residue)
        self.t_eps = 0.05
        self.loss_fn = FlowMatchingLoss(
            w_mse=1.0,
            w_pairwise=0.0,
            w_fape=0.0
        )
        
    def gen_t_embedding(self, t, max_positions=10000):
        half_dim = self.dim // 2
        emb = math.log(max_positions) / (half_dim - 1)
        emb = torch.arange(half_dim, device=t.device).float().mul(-emb).exp()
        emb = t[:,:,None] * emb[None,:]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1), mode='constant')
        return emb
    
    def forward(self, x, t):
        h = self.input_proj(x)
        h = h + self.pos_embedding
        
        # Time embedding
        t_emb = self.gen_t_embedding(t)
        t_emb = self.time_proj(t_emb)
        h = h + t_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)
            
        return self.out_proj(h)
    
    def training_step(self, batch, batch_idx):
        x = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_size = x.shape[0]

        e = self.prior.sample(x.shape)
        e = rmsdalign(x, e)

        t = torch.rand(batch_size, device=self.device)
        t_ = t[:, None, None]

        z = t_ * x + (1 - t_) * e
        t_seq = t[:, None].expand(-1, x.shape[1])

        x_pred = self(z, t_seq)

        denom = (1 - t_).clamp(min=self.t_eps)
        v = (x - z) / denom
        v_pred = (x_pred - z) / denom
        loss = ((v - v_pred) ** 2).mean()

        self.log('train_flow_loss', loss.item(), prog_bar=True)
        return loss
    
    def inference(self, batch_size=1, n_residues=None, N=10, return_intermediates=False):
        n_residues = self.n_residue if n_residues is None else n_residues

        z = self.prior.sample((batch_size, n_residues, 3))

        intermediates = [z] if return_intermediates else None

        for n in range(N):
            t = n / N
            t_next = (n + 1) / N
            time = torch.full((batch_size,), t, device=self.device)
            time = time[:, None].expand(-1, n_residues)

            x_pred = self(z, time)

            if n == N - 1:
                if return_intermediates:
                    return x_pred, intermediates
                return x_pred

            z_aligned = rmsdalign(x_pred, z)
            denom = max(1.0 - t, self.t_eps)
            v_pred = (x_pred - z_aligned) / denom
            z = z_aligned + (t_next - t) * v_pred

            if return_intermediates:
                intermediates.append(z)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        return optimizer



class CryoEMFlowMatching(pl.LightningModule):
    def __init__(self, pdb_file, xtc_file, data_dir, flow_model_path=None, num_frames=None, use_first_frame_initial=False, flow_weight=0.5, render_weight=0.5, structure_lr=0.001, flow_lr=1e-4, render_loss_type='mse', log_every_n_epochs=50, sigma=5.0, consecutive_distance_weight=1.0, pairwise_distance_weight=0.0, images_per_frame=100, guided_N=10, guided_N_min=4, guided_N_rampup_retrains=5, guided_warmup_steps=3, guided_strength=0.01, guided_n_candidates=5, warmup_epochs=200, flow_warmup_epochs=100, flow_train_epochs=1, flow_retrain_every=1, flow_train_batch_size=256, inverse_crime=False, pose_error_frac=0.0, ctf_error_frac=0.0, pose_error_deg=0.0, ctf_defocus_error=0.4, error_seed=0, latent_init_npz=None, latent_init_recenter='frame0'):
        super().__init__()
        self.save_hyperparameters(ignore=['pdb_file', 'xtc_file'])

        # Store loss weights
        self.flow_weight = flow_weight
        self.render_weight = render_weight
        self.consecutive_distance_weight = consecutive_distance_weight
        self.pairwise_distance_weight = pairwise_distance_weight
        self.structure_lr = structure_lr
        self.flow_lr = flow_lr
        self.render_loss_type = render_loss_type
        self.log_every_n_epochs = log_every_n_epochs
        self.sigma = sigma
        self.images_per_frame = images_per_frame

        # Guided sampling params
        self.guided_N = guided_N
        self.guided_N_min = guided_N_min
        self.guided_N_rampup_retrains = guided_N_rampup_retrains
        self.guided_warmup_steps = guided_warmup_steps
        self.guided_strength = guided_strength
        self.guided_n_candidates = guided_n_candidates
        
        # Load MD data
        self.pdb_file = pdb_file
        self.xtc_file = xtc_file
        
        self.universe = mda.Universe(self.pdb_file, self.xtc_file)
        
        # Load dataset and frame indices
        import json
        with open(os.path.join(data_dir, "metadata.json")) as f:
            meta = json.load(f)
        self.frame_indices = meta["trajectory_frame_indices"]
        if inverse_crime:
            self.dataset = RenderedMDTrajectoryDataset(
                self.universe,
                frame_indices=self.frame_indices,
                images_per_frame=images_per_frame,
                sigma=sigma,
                batch_size_per_frame=5
            )
        elif pose_error_frac > 0 or ctf_error_frac > 0:
            self.dataset = CorruptedTemSimulatorDataset(
                data_dir,
                batch_size_per_frame=1,
                universe=self.universe,
                pose_error_frac=pose_error_frac,
                ctf_error_frac=ctf_error_frac,
                pose_error_deg=pose_error_deg,
                ctf_defocus_error=ctf_defocus_error,
                error_seed=error_seed,
            )
        else:
            self.dataset = TemSimulatorDataset(
                data_dir,
                batch_size_per_frame=1,
                universe=self.universe,
            )
        # Only CTF errors need plumbing; pose errors already live in dataset angles.
        self._use_defocus = ctf_error_frac > 0 and hasattr(self.dataset, "defocus")
        print(f"Using {len(self.frame_indices)} specific frames: {self.frame_indices}")

        self.universe.trajectory[0]  # Go to actual frame 0
        #self.dataset_original = MDTrajectoryDataset(self.universe, scale=True)
        #self.universe.trajectory[0]  # Go to actual frame 0
        # Create renderer
        self.renderer = CryoJAXRenderer()
        #self.renderer = Renderer()
        
        n_residue = len(self.dataset.backbone_atoms)
        
        self.flow_model = FlowModel(
            n_residue=n_residue,
            layers=3,
            dim=256,
            heads=8,
            lr=self.flow_lr
        )
        
        if flow_model_path and os.path.exists(flow_model_path):
            print(f"Loading pretrained model from {flow_model_path}")
            self.flow_model.load_state_dict(torch.load(flow_model_path))
            self._flow_model_pretrained = True
        else:
            self._flow_model_pretrained = False
        self._flow_model_frozen = False
        
        self.flow_loss_fn = FlowMatchingLoss(w_mse=1.0)
        
        # Initialize rendering loss
        self.render_loss_fn = RenderingLoss(loss_type=self.render_loss_type)

        
        self._latent_init_npz = latent_init_npz
        self._latent_init_recenter = latent_init_recenter
        self.parametric_structures = self._initialize_parametric_structures(len(self.frame_indices), use_first_frame_initial=use_first_frame_initial)

        # Initialize pairwise distance regularizer with initial structure (frame 0)
        if self.pairwise_distance_weight > 0:
            self.universe.trajectory[0]
            initial_coords = torch.FloatTensor(self.dataset.backbone_atoms.positions)
            self.pairwise_distance_loss_fn = PairwiseDistanceRegularizer(initial_coords)
        else:
            self.pairwise_distance_loss_fn = None

        # Initialize StandardScaler for coords_batch normalization
        self.coords_scaler = StandardScaler()
        # Fit scaler on all parametric structure coordinates
        all_coords_reshaped = self.parametric_structures.detach().reshape(-1, 3)
        self.coords_scaler.fit(all_coords_reshaped)
        #self.coords_scaler = self.dataset_original.scaler
        # Note: scaler tensors will be moved to correct device in on_fit_start()
        
        # Storage for creating optimization GIF
        self.optimization_coords_dict = {}  # Store coordinates of tracked structures at each epoch
        self.tracked_parametric_indices = {}  # Store which parametric indices we're tracking

        # Storage for RMSD comparison (latent vs final_prediction) per visualized frame
        self.rmsd_history = {}  # {frame_label: {'latent': [], 'final_prediction': [], 'epoch': []}}

        # Storage for loss-RMSD correlation analysis
        self.correlation_history = {'losses': [], 'rmsds': [], 'epochs': []}

        # Cache frame backbone coordinates on CPU to avoid repeated trajectory seeks.
        self._frame_backbone_cache = {}

        # Phased training params
        self._warmup_epochs = warmup_epochs
        self._flow_warmup_epochs = flow_warmup_epochs
        self._flow_train_epochs = flow_train_epochs
        self._flow_retrain_every = flow_retrain_every
        self._flow_train_batch_size = flow_train_batch_size
        self._flow_retrain_count = 0
        self._flow_model_ready = self._flow_model_pretrained

        # Initialize indices to visualize - shared across all visualization methods
        self._initialize_visualization_indices()
    
    def _initialize_visualization_indices(self):
        num_structures = self.parametric_structures.shape[0] if hasattr(self, 'parametric_structures') else 0
        
        if num_structures == 0:
            self.indices_to_visualize = []
        elif num_structures == 1:
            # Single structure: visualize it with frame label
            frame_label = str(self.frame_indices[0]) if hasattr(self, 'frame_indices') and self.frame_indices else "0"
            self.indices_to_visualize = [(0, frame_label)]
        else:
            # Multiple structures: visualize middle (and optionally last)
            mid_idx = (num_structures - 1) // 2
            mid_label = str(self.frame_indices[mid_idx]) if hasattr(self, 'frame_indices') else str(mid_idx)
            self.indices_to_visualize = [(mid_idx, mid_label)]
            
            # Add last index if different from middle
            last_idx = num_structures - 1
            if last_idx != mid_idx:
                last_label = str(self.frame_indices[last_idx]) if hasattr(self, 'frame_indices') else str(last_idx)
                self.indices_to_visualize.append((last_idx, last_label))

        # always visalize the first frame (index 0) if it exists and is not already included
        # if num_structures > 0 and (0, str(self.frame_indices[0]) if hasattr(self, 'frame_indices') else "0") not in self.indices_to_visualize:
        #     self.indices_to_visualize.insert(0, (0, str(self.frame_indices[0]) if hasattr(self, 'frame_indices') else "0"))
    
    def _load_latent_init(self, num_frames, n_residue):
        # cryoDRGN emits density, not coordinates; thesis/vol_fit.py flexibly fits a
        # backbone into each decoded volume. This loads that ensemble (one structure per
        # particle, so index i is the answer for frame i) as the latent initialisation.
        d = np.load(self._latent_init_npz)
        key = 'coords' if 'coords' in d.files else d.files[0]
        coords = torch.as_tensor(np.asarray(d[key]), dtype=torch.float32)
        if coords.shape[0] < num_frames:
            raise ValueError(f"{self._latent_init_npz}: {coords.shape[0]} structures < {num_frames} frames")
        if coords.shape[1] != n_residue:
            raise ValueError(f"{self._latent_init_npz}: {coords.shape[1]} atoms != {n_residue} backbone atoms")
        coords = coords[:num_frames].clone()

        # The fits live in the volume frame (origin = box centre); the renderer subtracts
        # each structure's bbox midpoint anyway, so this only restores the absolute frame.
        if self._latent_init_recenter == 'frame0':
            self.universe.trajectory[0]
            ref = torch.FloatTensor(self.dataset.backbone_atoms.positions)
            target_mid = 0.5 * (ref.amin(0) + ref.amax(0))
            mid = 0.5 * (coords.amin(1) + coords.amax(1))
            coords = coords - mid.unsqueeze(1) + target_mid
        elif self._latent_init_recenter != 'none':
            raise ValueError(f"unknown latent_init_recenter: {self._latent_init_recenter}")

        print(f"Initialized {num_frames} latents from {self._latent_init_npz} "
              f"(recenter={self._latent_init_recenter})")
        return coords.contiguous()

    def _initialize_parametric_structures(self, num_frames=None, use_first_frame_initial=False):
        n_residue = len(self.dataset.backbone_atoms)

        if getattr(self, '_latent_init_npz', None):
            if use_first_frame_initial:
                print("latent_init_npz overrides use_first_frame_initial")
            latent_init = self._load_latent_init(num_frames, n_residue)
            if hasattr(self, 'parametric_structures'):
                self._initialize_visualization_indices()
            return nn.Parameter(latent_init)

        if use_first_frame_initial:
            self.universe.trajectory[0]
            frame_0_coords = torch.FloatTensor(self.dataset.backbone_atoms.positions)
            base_coords = self.dataset.normalize_coords(frame_0_coords)

        all_coords = []
        for i in range(num_frames):
            if use_first_frame_initial:
                all_coords.append(base_coords.clone())
            else:
                all_coords.append(self.flow_model.prior.sample((1, n_residue, 3)).squeeze(0))

        if hasattr(self, 'parametric_structures'):
            self._initialize_visualization_indices()

        return nn.Parameter(torch.stack(all_coords))  # [n_frames, n_res, 3]
    
    
    

    def _defocus_for(self, frame_indices, num_images=1):
        if not self._use_defocus:
            return None
        d = self.dataset.get_defocus(frame_indices.detach().cpu())
        return d[:, :num_images].reshape(-1)

    def render_model(self, coords_batch, gt_images, angles_list, defocus=None):
        batch_size = coords_batch.shape[0]
        num_images = gt_images.shape[1]

        # Tile coords for each image, flatten to [batch*num_images, n_res, 3]
        coords_flat = coords_batch.repeat_interleave(num_images, dim=0)
        angles_flat = angles_list.reshape(batch_size * num_images, 3)

        pred_imgs = self.renderer.render_particle_batch(coords_flat, angles_flat, sigma=self.sigma, defocus=defocus)
        if isinstance(pred_imgs, tuple):
            pred_imgs = pred_imgs[0]

        gt_flat = gt_images.reshape(batch_size * num_images, *gt_images.shape[2:])
        if self.render_loss_type in ('mae', 'mse'):
            return self.render_loss_fn(pred_imgs, gt_flat)
        else:
            return torch.stack([
                self.render_loss_fn(pred_imgs[k], gt_flat[k])
                for k in range(batch_size * num_images)
            ]).mean()
    
    
    def _log_first_angle(self, angles_list):
        if not hasattr(self, '_angles_logged'):
            first_angle = angles_list[0][0].tolist()
            self._first_angle = tuple(first_angle)
            print(f"First angle in angles_list: {first_angle}")
            if self.logger and self.logger.experiment:
                self.logger.experiment.log({
                    "first_rendering_angle": str(first_angle)
                })
            self._angles_logged = True

    def _get_cached_backbone_positions(self, actual_frame_idx):
        if actual_frame_idx not in self._frame_backbone_cache:
            self.universe.trajectory[actual_frame_idx]
            self._frame_backbone_cache[actual_frame_idx] = torch.from_numpy(
                self.universe.select_atoms('backbone').positions.copy()
            ).float()
        return self._frame_backbone_cache[actual_frame_idx]
    
    def _compute_render_guidance_loss(self, prediction_coords, gt_images, angles_list, frame_indices):
        render_guidance_loss = 0.0
        batch_size = frame_indices.shape[0]
        defocus = self._defocus_for(frame_indices)
        for i in range(batch_size):
            pred_coords = prediction_coords[i]
            gt_image = gt_images[i][0]
            angles = tuple(angles_list[i][0].tolist())
            pred_img, _ = self.renderer.render_particle(
                pred_coords, angles, sigma=self.sigma,
                defocus=None if defocus is None else float(defocus[i]))
            img_loss = self.render_weight * self.render_loss_fn(pred_img, gt_image)
            render_guidance_loss += img_loss
        return render_guidance_loss / batch_size
    
    
    def _calculate_rmsd(self, prediction, gt_coords_denorm):
        final_prediction_denorm = self.coords_scaler.reverse_transform(prediction)
        pred_final_aligned = rmsdalign(gt_coords_denorm, final_prediction_denorm)
        diff = pred_final_aligned - gt_coords_denorm
        rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1))).item()
        return rmsd

    def _update_rmsd_history(self, latent_coords, final_prediction, all_candidates, all_candidate_losses, gt_coords, frame_indices):
        with torch.no_grad():
            frame_indices_list = frame_indices.tolist()
            for param_idx, frame_label in self.indices_to_visualize:
                if param_idx in frame_indices_list:
                    batch_i = frame_indices_list.index(param_idx)
                    rmsd_latent = self._calculate_rmsd(latent_coords[batch_i], gt_coords[batch_i])
                    rmsd_best = self._calculate_rmsd(final_prediction[batch_i], gt_coords[batch_i])
                    candidate_rmsds = [self._calculate_rmsd(cand, gt_coords[batch_i]) for cand in all_candidates[batch_i]]
                    if frame_label not in self.rmsd_history:
                        self.rmsd_history[frame_label] = {'latent': [], 'best': [], 'candidates': [], 'epoch': []}
                    self.rmsd_history[frame_label]['latent'].append(rmsd_latent)
                    self.rmsd_history[frame_label]['best'].append(rmsd_best)
                    self.rmsd_history[frame_label]['candidates'].append(candidate_rmsds)
                    self.rmsd_history[frame_label]['epoch'].append(self.current_epoch)
                    self.correlation_history['losses'].extend(all_candidate_losses[batch_i])
                    self.correlation_history['rmsds'].extend(candidate_rmsds)
                    self.correlation_history['epochs'].extend([self.current_epoch] * len(candidate_rmsds))
    
    def _train_flow_model_on_latents(self):
        if not self._flow_model_ready:
            n_residue = self.flow_model.n_residue
            self.flow_model = FlowModel(
                n_residue=n_residue, layers=3, dim=256, heads=8, lr=self.flow_lr
            ).to(self.device)
        else:
            for param in self.flow_model.parameters():
                param.requires_grad = True
            self.flow_model.train()

        all_coords = self.parametric_structures.detach()
        latent_scaler = StandardScaler()
        latent_scaler.fit(all_coords.reshape(-1, 3))
        normalized_coords = latent_scaler.transform(all_coords)
        latent_dataset = torch.utils.data.TensorDataset(normalized_coords)
        dataloader = torch.utils.data.DataLoader(latent_dataset, batch_size=self._flow_train_batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.flow_lr, betas=(0.9, 0.95), weight_decay=0.0001)
        self.flow_model.train()
        is_warmup = self._flow_retrain_count == 0
        epochs_since_warmup = self.current_epoch - self._warmup_epochs
        is_long_retrain = (not is_warmup) and epochs_since_warmup > 0 and (epochs_since_warmup % 300 == 0)
        if is_warmup:
            n_epochs = self._flow_warmup_epochs
        elif is_long_retrain:
            n_epochs = 50
        else:
            n_epochs = self._flow_train_epochs
        lr_warmup_steps = 100 if is_warmup else 0
        global_step = 0
        print(f"  Flow training: {n_epochs} epochs (retrain #{self._flow_retrain_count}, warmup={is_warmup}, long={is_long_retrain}, lr={self.flow_lr})")
        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch in dataloader:
                x1 = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(self.device)
                batch_size = x1.shape[0]
                x0 = self.flow_model.prior.sample(x1.shape)
                x0 = rmsdalign(x1, x0)
                t = torch.rand(batch_size, device=self.device)
                t_ = t[:, None, None]
                z = t_ * x1 + (1 - t_) * x0
                t_expanded = t[:, None].expand(-1, x1.shape[1])
                x_pred = self.flow_model(z, t_expanded)
                denom = (1 - t_).clamp(min=self.flow_model.t_eps)
                v = (x1 - z) / denom
                v_pred = (x_pred - z) / denom
                loss = ((v - v_pred) ** 2).mean()
                #loss = ((x1 - x_pred) ** 2).mean()
                if lr_warmup_steps > 0 and global_step < lr_warmup_steps:
                    lr_scale = (global_step + 1) / lr_warmup_steps
                    for pg in optimizer.param_groups:
                        pg['lr'] = self.flow_lr * lr_scale
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                global_step += 1
            if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                print(f"  Flow epoch {epoch+1}/{n_epochs} loss={total_loss/len(dataloader):.4f}")

        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False

        self.coords_scaler = latent_scaler
        self._flow_model_ready = True
        self._flow_retrain_count += 1
        print(f"Flow model trained on {len(all_coords)} structures (total retrains: {self._flow_retrain_count})")
        #self._log_flow_samples()

    def _log_flow_samples(self):
        try:
            template_universe = mda.Universe(self.pdb_file)
            backbone = template_universe.select_atoms('backbone')

            with torch.no_grad():
                unguided = self.flow_model.inference(batch_size=1, N=self.guided_N)
            unguided_abs = self.coords_scaler.reverse_transform(unguided[0]).detach().cpu().numpy()

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                pdb_path = tmp.name
            backbone.positions = unguided_abs
            backbone.write(pdb_path)
            with open(pdb_path, 'r') as f:
                pdb_str = f.read()
            os.remove(pdb_path)

            viewer = py3Dmol.view(width=800, height=400)
            viewer.addModel(pdb_str, "pdb")
            viewer.setStyle({"cartoon": {"color": "cyan"}})
            viewer.zoomTo()
            log_py3Dmol_to_wandb(viewer, name="unguided_sample_post_train")

            param_idx = self.frame_indices.index(5000) if 5000 in self.frame_indices else len(self.frame_indices) // 2
            gt_images_frame, _, angles_frame, _ = self.dataset[param_idx]
            gt_image0 = gt_images_frame[0].to(self.device)
            angles0 = tuple(angles_frame[0].tolist())

            guided = self.guided_sampling_single(
                self.coords_scaler.transform(
                    self.parametric_structures[param_idx].unsqueeze(0)
                ).squeeze(0),
                gt_image0, angles0, N=self.guided_N,
                warmup_steps=self.guided_warmup_steps,
                guidance_strength=self.guided_strength,
                defocus=(float(self.dataset.defocus[param_idx, 0]) if self._use_defocus else None),
            )
            guided_abs = self.coords_scaler.reverse_transform(guided)

            actual_frame = self.frame_indices[param_idx]
            self.universe.trajectory[actual_frame]
            gt_positions = self.universe.select_atoms('backbone').positions
            gt_tensor = torch.from_numpy(gt_positions).float().to(guided_abs.device)
            guided_aligned = rmsdalign(gt_tensor, guided_abs)
            guided_np = guided_aligned.detach().cpu().numpy()
            rmsd = np.sqrt(np.mean(np.sum((guided_np - gt_positions) ** 2, axis=-1)))

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                gt_path = tmp.name
            backbone.positions = gt_positions
            backbone.write(gt_path)
            with open(gt_path, 'r') as f:
                gt_str = f.read()
            os.remove(gt_path)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
                g_path = tmp.name
            backbone.positions = guided_np
            backbone.write(g_path)
            with open(g_path, 'r') as f:
                g_str = f.read()
            os.remove(g_path)

            viewer = py3Dmol.view(width=800, height=400, viewergrid=(1, 2), linked=True)
            viewer.addModel(gt_str, "pdb", viewer=(0, 0))
            viewer.setStyle({"cartoon": {"color": "lightgrey"}}, viewer=(0, 0))
            viewer.addModel(g_str, "pdb", viewer=(0, 1))
            viewer.setStyle({"cartoon": {"color": "lightgreen"}}, viewer=(0, 1))
            viewer.zoomTo()
            log_py3Dmol_to_wandb(viewer, name="guided_sample_post_train")

            if self.logger and hasattr(self.logger, 'experiment'):
                self.logger.experiment.log({"guided_sample_rmsd_post_train": rmsd})
            print(f"Flow sample RMSD (guided, frame {actual_frame}): {rmsd:.3f} Å")
        except Exception as e:
            print(f"Flow sample logging error: {e}")
            import traceback
            traceback.print_exc()


    def training_step(self, batch, batch_idx):
        gt_images, frame_indices, angles_list, gt_coords = batch
        self._log_first_angle(angles_list)
        batch_size = frame_indices.shape[0]

        guidance_loss = torch.tensor(0.0, device=self.device)
        predicted_coords_abs = self.parametric_structures[frame_indices]

        past_warmup = self.current_epoch >= self._warmup_epochs
        flow_active = self.flow_weight > 0 and (self._flow_model_pretrained or past_warmup)
        use_flow = flow_active and self._flow_model_ready

        # periodic flow training on latents (skip if pretrained)
        if flow_active and not self._flow_model_pretrained and batch_idx == 0:
            if (self.current_epoch - self._warmup_epochs) % self._flow_retrain_every == 0:
                print(f"Epoch {self.current_epoch}: Training flow model on latent coords...")
                self._train_flow_model_on_latents()

        if use_flow:
            latent_coords = self.coords_scaler.transform(predicted_coords_abs)
            # Schedule guided_N: ramp from guided_N_min -> guided_N as flow model matures (measured by retrain count)
            if self._flow_model_pretrained:
                current_guided_N = self.guided_N
            else:
                rampup = max(1, self.guided_N_rampup_retrains)
                progress = min(1.0, max(0, self._flow_retrain_count - 1) / rampup)
                current_guided_N = int(round(self.guided_N_min + (self.guided_N - self.guided_N_min) * progress))
                current_guided_N = max(self.guided_N_min, min(self.guided_N, current_guided_N))
            current_warmup_steps = min(self.guided_warmup_steps, max(0, current_guided_N - 1))
            self.log('train_guided_N', float(current_guided_N), prog_bar=True)
            all_candidates, all_candidate_losses, final_prediction, best_render_loss = self.guided_sampling_select_best(
                latent_coords, gt_images, angles_list, frame_indices,
                n_candidates=self.guided_n_candidates, N=current_guided_N,
                warmup_steps=current_warmup_steps, guidance_strength=self.guided_strength
            )
            guidance_loss = torch.functional.F.l1_loss(final_prediction, latent_coords)
            self.log('train_best_prior_render_loss', best_render_loss, prog_bar=True)
            self.log('train_guidance_loss', guidance_loss.item(), prog_bar=True)
            if self.current_epoch % self.log_every_n_epochs == 0:
                self._update_rmsd_history(latent_coords, final_prediction, all_candidates, all_candidate_losses, gt_coords, frame_indices)
        else:
            self.log('train_guidance_loss', 0.0, prog_bar=True)
            self.log('train_best_prior_render_loss', 0.0, prog_bar=True)


        pairwise_loss = torch.tensor(0.0, device=self.device)
        if self.pairwise_distance_weight > 0 and self.pairwise_distance_loss_fn is not None:
            pairwise_loss = self.pairwise_distance_loss_fn(predicted_coords_abs)
        self.log('train_pairwise_loss', pairwise_loss.item(), prog_bar=True)

        render_loss = self.render_model(
            predicted_coords_abs, gt_images, angles_list,
            defocus=self._defocus_for(frame_indices, num_images=gt_images.shape[1]))
        self.log('train_render_loss', render_loss.item(), prog_bar=True)

        combined_loss = (self.flow_weight * guidance_loss +
                        self.render_weight * render_loss +
                        self.pairwise_distance_weight * pairwise_loss)
        self.log('train_combined_loss', combined_loss.item(), prog_bar=True)
        return combined_loss
    

    def guided_sampling_single(self, x_t, gt_image, angles, N=10, warmup_steps=3, guidance_strength=0.1, defocus=None):
        x_t = x_t.unsqueeze(0).to(self.device)
        time = torch.empty((1, x_t.shape[1]), device=self.device)
        dps_scale = torch.tensor(guidance_strength, device=self.device)
        for step in range(N):
            t = step / N
            s = t + 1/N
            time.fill_(t)

            if step >= warmup_steps:
                with torch.no_grad():
                    pred_coords = self.flow_model(x_t, time)
                pred_coords = pred_coords.detach().requires_grad_(True)
                pred_denorm = self.coords_scaler.reverse_transform(pred_coords.squeeze(0))
                pred_img, _ = self.renderer.render_particle(pred_denorm, angles, sigma=self.sigma, defocus=defocus)
                loss = self.render_weight * self.render_loss_fn(pred_img, gt_image)
                if self.pairwise_distance_weight > 0 and self.pairwise_distance_loss_fn is not None:
                    loss = loss + self.pairwise_distance_weight * self.pairwise_distance_loss_fn(pred_denorm)
                residual = pred_img - gt_image
                res_norm = torch.norm(residual.reshape(-1)).clamp_min(1e-8)
                dps_scale = guidance_strength / res_norm
                grad = torch.autograd.grad(loss, pred_coords)[0]
                pred_coords = pred_coords.detach()
            else:
                with torch.no_grad():
                    pred_coords = self.flow_model(x_t, time)

            if step == N - 1:
                return pred_coords.squeeze(0)

            with torch.no_grad():
                z_aligned = rmsdalign(pred_coords, x_t)
                # schedule: strong early, decays to 0 at t=1
                guided_pred = pred_coords - dps_scale * grad if step >= warmup_steps else pred_coords
                denom = max(1.0 - t, self.flow_model.t_eps)
                v_pred = (guided_pred - z_aligned) / denom
                x_t = z_aligned + (s - t) * v_pred
        return x_t.squeeze(0)

    def guided_sampling_batched(self, x_t_batch, gt_images_batch, angles_tensor, N=10, warmup_steps=3, guidance_strength=0.1, defocus=None):
        # x_t_batch: [B, n_res, 3], gt_images_batch: [B, H, W], angles_tensor: [B, 3]
        B = x_t_batch.shape[0]
        x_t = x_t_batch.to(self.device)
        grads = torch.zeros_like(x_t)
        dps_scale = torch.full((B, 1, 1), guidance_strength, device=self.device)
        time = torch.empty((B, x_t.shape[1]), device=self.device)

        # Precompute NCC modules once — avoids recreating B modules every step
        if self.render_loss_type == 'ncc':
            from cryoflow.losses import NCC
            ncc_modules = [NCC(gt_images_batch[j].unsqueeze(0).unsqueeze(0), keep_channels=False) for j in range(B)]

        def _render_loss(pred_imgs):
            if self.render_loss_type in ('mae', 'mse'):
                return self.render_loss_fn(pred_imgs, gt_images_batch) * B
            return torch.stack([
                1.0 - ncc_modules[j](pred_imgs[j].unsqueeze(0).unsqueeze(0)).max() for j in range(B)
            ]).sum()

        for step in range(N):
            t = step / N
            s = t + 1/N
            time.fill_(t)

            if step >= warmup_steps:
                with torch.no_grad():
                    pred_coords = self.flow_model(x_t, time)
                pred_coords = pred_coords.detach().requires_grad_(True)
                pred_denorm = self.coords_scaler.reverse_transform(pred_coords)
                pred_imgs = self.renderer.render_particle_batch(pred_denorm, angles_tensor, sigma=self.sigma, defocus=defocus)
                if isinstance(pred_imgs, tuple):
                    pred_imgs = pred_imgs[0]
                total_loss = self.render_weight * _render_loss(pred_imgs)
                if self.pairwise_distance_weight > 0 and self.pairwise_distance_loss_fn is not None:
                    total_loss = total_loss + self.pairwise_distance_weight * B * self.pairwise_distance_loss_fn(pred_denorm)
                residual = pred_imgs - gt_images_batch
                res_norm = torch.norm(residual.reshape(B, -1), dim=1).clamp_min(1e-8)
                dps_scale = (guidance_strength / res_norm).view(B, 1, 1)
                # backbone continuity: consecutive CA-CA ≈ 3.8 Å
                grads = torch.autograd.grad(total_loss, pred_coords)[0]
                pred_coords = pred_coords.detach()
            else:
                with torch.no_grad():
                    pred_coords = self.flow_model(x_t, time)

            if step == N - 1:
                return pred_coords

            with torch.no_grad():
                z_aligned = rmsdalign(pred_coords, x_t)
                # schedule: strong early, decays to 0 at t=1
                guided_pred = pred_coords - dps_scale * grads if step >= warmup_steps else pred_coords
                denom = max(1.0 - t, self.flow_model.t_eps)
                v_pred = (guided_pred - z_aligned) / denom
                x_t = z_aligned + (s - t) * v_pred

        return x_t

    def guided_sampling_select_best(self, latent_coords, gt_images, angles_list, frame_indices, n_candidates=5, N=10, warmup_steps=3, guidance_strength=0.01):
        batch_size = latent_coords.shape[0]
        n_residues = latent_coords.shape[1]
        gt_images_batch = gt_images[:, 0]
        angles_batch = angles_list[:, 0, :].to(self.device)
        defocus_batch = self._defocus_for(frame_indices)

        # Run all candidate inference in a single batched call
        with torch.no_grad():
            all_preds, all_priors_list = self.flow_model.inference(
                batch_size=batch_size * n_candidates, n_residues=n_residues, N=warmup_steps, return_intermediates=True
            )
            all_preds = all_preds.reshape(batch_size, n_candidates, n_residues, 3)
            all_priors = all_priors_list[0].reshape(batch_size, n_candidates, n_residues, 3)

        with torch.no_grad():
            pred_denorm_all = self.coords_scaler.reverse_transform(
                all_preds.reshape(batch_size * n_candidates, n_residues, 3)
            )
            angles_expanded = angles_batch.unsqueeze(1).expand(-1, n_candidates, -1).reshape(batch_size * n_candidates, 3)
            defocus_expanded = (None if defocus_batch is None
                                else defocus_batch.repeat_interleave(n_candidates))
            pred_imgs_all = self.renderer.render_particle_batch(pred_denorm_all, angles_expanded, sigma=self.sigma, defocus=defocus_expanded)
            pred_imgs_all = pred_imgs_all.reshape(batch_size, n_candidates, *pred_imgs_all.shape[1:])

            if self.render_loss_type == 'mae':
                gt_expanded = gt_images_batch.unsqueeze(1).expand(-1, n_candidates, -1, -1)
                candidate_losses_all = F.l1_loss(pred_imgs_all, gt_expanded, reduction='none').mean(dim=(-2, -1))
            elif self.render_loss_type == 'mse':
                gt_expanded = gt_images_batch.unsqueeze(1).expand(-1, n_candidates, -1, -1)
                candidate_losses_all = F.mse_loss(pred_imgs_all, gt_expanded, reduction='none').mean(dim=(-2, -1))
            else:
                from cryoflow.losses import NCC
                ncc_modules = [
                    NCC(gt_images_batch[i].unsqueeze(0).unsqueeze(0), keep_channels=False)
                    for i in range(batch_size)
                ]
                candidate_losses = []
                for i in range(batch_size):
                    candidate_losses.append(torch.stack([
                        1.0 - ncc_modules[i](pred_imgs_all[i, j].unsqueeze(0).unsqueeze(0)).max()
                        for j in range(n_candidates)
                    ]))
                candidate_losses_all = torch.stack(candidate_losses, dim=0)

            best_indices = candidate_losses_all.argmin(dim=1)
            all_candidate_losses = candidate_losses_all.tolist()

        batch_arange = torch.arange(batch_size, device=best_indices.device)
        best_x_ts = all_priors[batch_arange, best_indices]
        all_candidates = [[all_preds[i, j] for j in range(n_candidates)] for i in range(batch_size)]

        # Run guided sampling for all batch elements in one batched call
        x_t_batch = best_x_ts
        final_predictions = self.guided_sampling_batched(x_t_batch, gt_images_batch, angles_batch, N, 0, guidance_strength, defocus=defocus_batch)

        with torch.no_grad():
            pred_denorm_final = self.coords_scaler.reverse_transform(final_predictions)
            pred_imgs_final = self.renderer.render_particle_batch(pred_denorm_final, angles_batch, sigma=self.sigma, defocus=defocus_batch)
            if self.render_loss_type == 'mae':
                best_loss_tensors = F.l1_loss(pred_imgs_final, gt_images_batch, reduction='none').mean(dim=(-2, -1))
            elif self.render_loss_type == 'mse':
                best_loss_tensors = F.mse_loss(pred_imgs_final, gt_images_batch, reduction='none').mean(dim=(-2, -1))
            else:
                best_loss_tensors = torch.stack([
                    self.render_loss_fn(pred_imgs_final[i], gt_images_batch[i]) for i in range(batch_size)
                ])
        best_render_loss = best_loss_tensors.mean().item()

        return all_candidates, all_candidate_losses, final_predictions, best_render_loss
    
    def on_train_epoch_end(self):
        if self.logger:
            if self.current_epoch % self.log_every_n_epochs == 0 or self.current_epoch == self.trainer.max_epochs - 1:
                #self._log_parametric_structure_py3dmol()
                #self._log_example_visualizations()
                #self._log_3d_structure_comparison()
                self._log_all_structure_rmsd()
                #self._log_per_axis_rmsd()
                #self._save_rmsd_comparison_plot()
                #self._save_loss_rmsd_correlation_plot()

    
    def _log_example_visualizations(self):
        try:
            if not self.indices_to_visualize:
                return
            
            for frame_idx, frame_label in self.indices_to_visualize:
                actual_frame_idx = self.frame_indices[frame_idx]

                try:
                    gt_images, _, angles_list, _ = self.dataset[frame_idx]
                    gt_img_tensor = gt_images[0].to(self.device)
                    angles = tuple(angles_list[0].tolist())

                    optimized_coords_tensor = self.parametric_structures[frame_idx]
                    pred_img_tensor, _ = self.renderer.render_particle(
                        optimized_coords_tensor, angles, sigma=self.sigma,
                        defocus=(float(self.dataset.defocus[frame_idx, 0]) if self._use_defocus else None))

                    loss = self.render_loss_fn(pred_img_tensor, gt_img_tensor)

                    # Convert to numpy arrays for visualization
                    gt_img_np = gt_img_tensor.detach().cpu().numpy()
                    pred_img_np = pred_img_tensor.detach().cpu().numpy()

                    fig, axes = plt.subplots(1, 2, figsize=(10, 5)) 

                    title = f"Frame {frame_label} - Loss: {loss.item():.4f} | epoch: {self.current_epoch}"
                    plt.suptitle(title, fontsize=14)
                    
                    axes[0].imshow(gt_img_np, cmap='gray')
                    axes[0].set_title(f"Frame {actual_frame_idx} GT")
                    axes[0].axis('off')
                    
                    axes[1].imshow(pred_img_np, cmap='gray')
                    axes[1].set_title(f"Frame {actual_frame_idx} Pred")
                    axes[1].axis('off')
                    
                    plt.tight_layout(pad=1.5) 
                    
                    log_key = f"rendered_cryo_em_{frame_label}"

                    if hasattr(self, 'logger') and self.logger is not None and self.logger.experiment is not None:
                        self.logger.experiment.log({
                            log_key: wandb.Image(fig), 
                            f"loss_frame_{frame_label}": loss.item()
                        })
                    plt.close(fig) 

                except Exception as e:
                    print(f"Viz Error (Frame {actual_frame_idx}): {e}")
                
        except Exception as e:
            print(f"Viz Setup Error: {e}")


    def _log_parametric_structure_py3dmol(self):
        if not hasattr(self, 'logger') or self.logger is None or self.logger.experiment is None:
            return

        try:
            if not self.indices_to_visualize:
                return

            template_universe = mda.Universe(self.pdb_file)
            backbone_selection = template_universe.select_atoms('backbone')
            
            for parametric_idx, frame_label in self.indices_to_visualize:
                if not hasattr(self, 'tracked_parametric_indices'):
                    self.tracked_parametric_indices = {}
                self.tracked_parametric_indices[frame_label] = parametric_idx

                actual_frame_idx = self.frame_indices[parametric_idx]

                gt_positions = self._get_cached_backbone_positions(actual_frame_idx).numpy()

                optimized_positions_tensor = self.parametric_structures[parametric_idx]

                gt_positions_tensor = self._get_cached_backbone_positions(actual_frame_idx).to(self.device)
                
                optimized_positions_tensor = rmsdalign(gt_positions_tensor, optimized_positions_tensor)

                optimized_positions_np = optimized_positions_tensor.detach().cpu().numpy()

                # Store coordinates for optimization GIF (for tracked parametric structures)
                if not hasattr(self, 'optimization_coords_dict'):
                    self.optimization_coords_dict = {}
                if frame_label not in self.optimization_coords_dict:
                    self.optimization_coords_dict[frame_label] = []
                self.optimization_coords_dict[frame_label].append(optimized_positions_np.copy())

                # Log detailed visualizations at configurable intervals or on the last epoch
                if self.current_epoch % self.log_every_n_epochs == 0 or self.current_epoch == self.trainer.max_epochs - 1:
                    rmsd = np.sqrt(np.mean(np.sum((optimized_positions_np - gt_positions) ** 2, axis=-1)))

                    # Log RMSD to wandb for tracking over time
                    self.logger.experiment.log({
                        f"rmsd_frame_{frame_label}": rmsd,
                    })
                    
                    gt_pdb_string = ""
                    opt_pdb_string = ""
                    
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb_file_gt:
                        gt_pdb_filename = tmp_pdb_file_gt.name
                    try:
                        backbone_selection.positions = gt_positions
                        backbone_selection.write(gt_pdb_filename)
                        with open(gt_pdb_filename, 'r') as f_read:
                            gt_pdb_string = f_read.read()
                    finally:
                        os.remove(gt_pdb_filename)

                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp_pdb_file_opt:
                        opt_pdb_filename = tmp_pdb_file_opt.name
                    try:
                        backbone_selection.positions = optimized_positions_np
                        backbone_selection.write(opt_pdb_filename)
                        with open(opt_pdb_filename, 'r') as f_read:
                            opt_pdb_string = f_read.read()
                    finally:
                        os.remove(opt_pdb_filename)

                    log_key_sidebyside = f"optimized_structure_{frame_label}"
                    viewer_sidebyside = py3Dmol.view(width=800, height=400, viewergrid=(1,2), linked=True)
                    viewer_sidebyside.addModel(gt_pdb_string, "pdb", viewer=(0,0))
                    viewer_sidebyside.setStyle({"cartoon": {"color": "lightgrey"}}, viewer=(0,0))
                    viewer_sidebyside.addModel(opt_pdb_string, "pdb", viewer=(0,1))
                    viewer_sidebyside.setStyle({"cartoon": {"color": "lightgreen"}}, viewer=(0,1))
                    
                    viewer_sidebyside.zoomTo()

                    viewer_sidebyside.addLabel(f"RMSD = {rmsd:.3f} Å | frame = {actual_frame_idx} | Green-Generated, Gray-GT | epoch: {self.current_epoch}", {
                        "fontSize": 16,
                        "fontColor": "black",
                        "backgroundColor": "white",
                        "position": {"x": 0, "y": 0, "z": 0}
                    })
                    
                    log_py3Dmol_to_wandb(viewer_sidebyside, name=log_key_sidebyside)

        except Exception as e:
            print(f"Py3DMol Error: {e}")


    def _log_all_structure_rmsd(self):
        if not hasattr(self, 'logger') or self.logger is None or self.logger.experiment is None:
            return

        try:
            num_structures = self.parametric_structures.shape[0]
            rmsds = []
            rmsds_z = []
            rmsds_x = []
            rmsds_y = []

            gt_positions_batch = torch.stack([
                self._get_cached_backbone_positions(self.frame_indices[structure_idx])
                for structure_idx in range(num_structures)
            ]).to(self.device)

            for structure_idx in range(num_structures):
                gt_positions_tensor = gt_positions_batch[structure_idx]
                optimized_positions_tensor = self.parametric_structures[structure_idx]
                
                # Align structures for fair RMSD calculation
                optimized_positions_tensor = rmsdalign(gt_positions_tensor, optimized_positions_tensor)
                diffs = optimized_positions_tensor - gt_positions_tensor
                
                # Calculate RMSD
                rmsd = torch.sqrt(torch.mean(torch.sum(diffs ** 2, dim=-1))).item()
                rmsds.append(rmsd)

                # Calculate Z-axis RMSD
                rmsd_z = torch.sqrt(torch.mean(diffs[:, 2] ** 2)).item()
                rmsds_z.append(rmsd_z)

                # Calculate X-axis RMSD
                rmsd_x = torch.sqrt(torch.mean(diffs[:, 0] ** 2)).item()
                rmsds_x.append(rmsd_x)

                # Calculate Y-axis RMSD
                rmsd_y = torch.sqrt(torch.mean(diffs[:, 1] ** 2)).item()
                rmsds_y.append(rmsd_y)

            
            # Calculate summary statistics only
            rmsds_array = np.array(rmsds)
            rmsds_z_array = np.array(rmsds_z)
            summary_stats = {
                "rmsd_all_mean": np.mean(rmsds_array),
                "rmsd_z_mean": np.mean(rmsds_z_array),
                "rmsd_x_mean": np.mean(rmsds_x),
                "rmsd_y_mean": np.mean(rmsds_y),
                "rmsd_all_std": np.std(rmsds_array),
                "rmsd_all_min": np.min(rmsds_array),
                "rmsd_all_max": np.max(rmsds_array),
                "rmsd_all_median": np.median(rmsds_array),
            }
            
            # Log only summary statistics to wandb
            self.logger.experiment.log(summary_stats)
            
        except Exception as e:
            print(f"Error calculating all structure RMSD: {e}")

    def _log_3d_structure_comparison(self):
        try:
            if not hasattr(self, 'logger') or self.logger is None or self.logger.experiment is None:
                return
            
            if not self.indices_to_visualize:
                return
            
            for frame_idx, frame_label in self.indices_to_visualize:
                actual_frame_idx = self.frame_indices[frame_idx]

                try:
                    _, _, _, gt_coords = self.dataset[frame_idx]
                    gt_coords_abs = self.dataset.reverse_transform(gt_coords).to(self.device)

                    pred_coords_abs = self.parametric_structures[frame_idx]

                    # Align predicted coordinates to ground truth for fair comparison
                    pred_coords_aligned = rmsdalign(gt_coords_abs, pred_coords_abs)
                    
                    # Convert to numpy arrays
                    gt_coords_np = gt_coords_abs.detach().cpu().numpy()
                    pred_coords_np = pred_coords_aligned.detach().cpu().numpy()
                    
                    # Calculate RMSD
                    rmsd = np.sqrt(np.mean(np.sum((gt_coords_np - pred_coords_np) ** 2, axis=-1)))
                    
                    # Calculate combined bounds for consistent axis scaling
                    all_coords = np.concatenate([gt_coords_np, pred_coords_np], axis=0)
                    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
                    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
                    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
                    
                    # Add some padding to the bounds
                    padding = 5.0  # Angstroms
                    x_range = [x_min - padding, x_max + padding]
                    y_range = [y_min - padding, y_max + padding]
                    z_range = [z_min - padding, z_max + padding]
                    
                    # Create interactive 3D plot with Plotly
                    fig = go.Figure()
                    
                    # Add ground truth structure
                    fig.add_trace(go.Scatter3d(
                        x=gt_coords_np[:, 0],
                        y=gt_coords_np[:, 1],
                        z=gt_coords_np[:, 2],
                        mode='markers',
                        marker={
                            'size': 8,
                            'color': 'rgba(255, 0, 0, 0.8)',
                            'line': {'width': 0}
                        },
                        name='Ground Truth',
                        text=[f'Atom {i}' for i in range(len(gt_coords_np))],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                    ))
                    
                    # Add predicted structure
                    fig.add_trace(go.Scatter3d(
                        x=pred_coords_np[:, 0],
                        y=pred_coords_np[:, 1],
                        z=pred_coords_np[:, 2],
                        mode='markers',
                        marker={
                            'size': 8,
                            'color': 'rgba(0, 0, 255, 0.8)',
                            'line': {'width': 0}
                        },
                        name='Predicted (aligned)',
                        text=[f'Atom {i}' for i in range(len(pred_coords_np))],
                        hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                    ))
                    
                    # Add lines connecting corresponding atoms
                    for i in range(len(gt_coords_np)):
                        fig.add_trace(go.Scatter3d(
                            x=[gt_coords_np[i, 0], pred_coords_np[i, 0]],
                            y=[gt_coords_np[i, 1], pred_coords_np[i, 1]],
                            z=[gt_coords_np[i, 2], pred_coords_np[i, 2]],
                            mode='lines',
                            line=dict(
                                color='black',
                                width=2,
                                dash='dash'
                            ),
                            opacity=0.5,
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Update layout with fixed axis ranges
                    fig.update_layout(
                        title=f'3D Structure Comparison - Frame {frame_label} (Epoch {self.current_epoch})<br>RMSD: {rmsd:.3f} Å',
                        scene=dict(
                            xaxis=dict(
                                title='X (Å)',
                                range=x_range
                            ),
                            yaxis=dict(
                                title='Y (Å)',
                                range=y_range
                            ),
                            zaxis=dict(
                                title='Z (Å)',
                                range=z_range
                            ),
                            aspectmode='cube'  # Forces equal aspect ratios
                        ),
                        width=800,
                        height=600,
                        margin=dict(l=0, r=0, b=0, t=50)
                    )
                    
                    # Convert to HTML string with explicit encoding
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
                        html_path = tmp_file.name
                        fig.write_html(html_path)
                    
                    # Log to wandb as HTML
                    log_key = f"3d_structure_comparison_{frame_label}"
                    self.logger.experiment.log({
                        log_key: wandb.Html(html_path),
                    })
                    
                except Exception as e:
                    print(f"3D Viz Error (Frame {actual_frame_idx}): {e}")
                    
        except Exception as e:
            print(f"3D Viz Setup Error: {e}")


    def _create_optimization_gif(self):
        if not hasattr(self, 'optimization_coords_dict') or not self.optimization_coords_dict:
            print("No optimization coordinates tracked, cannot create optimization GIFs")
            return
            
        if not hasattr(self, 'tracked_parametric_indices') or not self.tracked_parametric_indices:
            print("No parametric indices tracked, cannot create optimization GIFs")
            return
            
        try:
            from cryoflow.utils import save_gif
            import MDAnalysis as mda
            
            # Create output directory
            output_dir = os.path.join(paths.ANALYSIS, "optimization_trajectory")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create template universe for writing trajectories
            template_universe = mda.Universe(self.pdb_file)
            backbone_selection = template_universe.select_atoms('backbone')
            
            # Create GIFs for each tracked structure
            for frame_label in self.tracked_parametric_indices.keys():
                if frame_label not in self.optimization_coords_dict:
                    print(f"No optimization coordinates for frame {frame_label}")
                    continue
                    
                optimization_coords = self.optimization_coords_dict[frame_label]
                if len(optimization_coords) < 2:
                    print(f"Not enough epochs for frame {frame_label} to create optimization GIF")
                    continue
                
                # Get the tracked dataset index and actual frame index
                tracked_dataset_idx = self.tracked_parametric_indices[frame_label]
                tracked_actual_frame_idx = self.frame_indices[tracked_dataset_idx]
                
                print(f"Creating GIF for frame {frame_label} (dataset idx {tracked_dataset_idx}, actual frame {tracked_actual_frame_idx})")
                
                # Get ground truth coordinates for the tracked frame
                self.universe.trajectory[tracked_actual_frame_idx]
                gt_positions = self.universe.select_atoms('backbone').positions
                
                # Create ground truth PDB (single frame)
                gt_pdb_path = os.path.join(output_dir, f"gt_frame_{frame_label}.pdb")
                backbone_selection.positions = gt_positions
                backbone_selection.write(gt_pdb_path)
                
                # Create ground truth XTC (single frame, repeated for comparison)
                gt_xtc_path = os.path.join(output_dir, f"gt_frame_{frame_label}.xtc")
                with mda.coordinates.XTC.XTCWriter(gt_xtc_path, backbone_selection.n_atoms) as writer:
                    for _ in range(len(optimization_coords)):
                        backbone_selection.positions = gt_positions
                        writer.write(backbone_selection)
                
                # Create optimized trajectory XTC
                opt_xtc_path = os.path.join(output_dir, f"optimization_trajectory_{frame_label}.xtc")
                with mda.coordinates.XTC.XTCWriter(opt_xtc_path, backbone_selection.n_atoms) as writer:
                    for coords in optimization_coords:
                        backbone_selection.positions = coords
                        writer.write(backbone_selection)
                
                # Create optimized PDB (first frame)
                opt_pdb_path = os.path.join(output_dir, f"optimization_start_{frame_label}.pdb")
                backbone_selection.positions = optimization_coords[0]
                backbone_selection.write(opt_pdb_path)
                
                # Create the GIF
                gif_path = os.path.join(output_dir, f"optimization_progress_{frame_label}.gif")
                # Calculate fps to make gif 30 seconds long
                fps = len(optimization_coords) / 30.0
                gif_file = save_gif(
                    gt_pdb=gt_pdb_path,
                    gt_xtc=gt_xtc_path,
                    gen_pdb=opt_pdb_path,
                    gen_xtc=opt_xtc_path,
                    gif_path=gif_path,
                    fps=fps
                )
                
                # Log to wandb
                if hasattr(self, 'logger') and self.logger is not None and self.logger.experiment is not None:
                    # Log the GIF
                    self.logger.experiment.log({
                        f"optimization_progress_gif_{frame_label}": wandb.Image(str(gif_file))
                    })
                    
                    # Log trajectory files as artifacts
                    gt_pdb_artifact = wandb.Artifact(f"gt_pdb_{frame_label}", type="pdb")
                    gt_pdb_artifact.add_file(gt_pdb_path)
                    self.logger.experiment.log_artifact(gt_pdb_artifact)
                    
                    gt_xtc_artifact = wandb.Artifact(f"gt_xtc_{frame_label}", type="xtc")
                    gt_xtc_artifact.add_file(gt_xtc_path)
                    self.logger.experiment.log_artifact(gt_xtc_artifact)
                    
                    opt_pdb_artifact = wandb.Artifact(f"opt_pdb_{frame_label}", type="pdb")
                    opt_pdb_artifact.add_file(opt_pdb_path)
                    self.logger.experiment.log_artifact(opt_pdb_artifact)
                    
                    opt_xtc_artifact = wandb.Artifact(f"opt_xtc_{frame_label}", type="xtc")
                    opt_xtc_artifact.add_file(opt_xtc_path)
                    self.logger.experiment.log_artifact(opt_xtc_artifact)
                
                print(f"Optimization GIF for frame {frame_label} saved to: {gif_file}")
            
            # Log all Python source code files once at the end
            if hasattr(self, 'logger') and self.logger is not None and self.logger.experiment is not None:
                import glob
                source_code_artifact = wandb.Artifact("source_code", type="code")
                
                # Add all .py files in the current directory
                py_files = glob.glob("*.py")
                for py_file in py_files:
                    source_code_artifact.add_file(py_file)
                
                self.logger.experiment.log_artifact(source_code_artifact)
                print("All Python source code files saved to wandb")
            
        except Exception as e:
            print(f"Error creating optimization GIFs: {e}")
            import traceback
            traceback.print_exc()

    def _save_rmsd_comparison_plot(self):
        if not self.rmsd_history:
            print("No RMSD history to plot")
            return
        os.makedirs("guided_sampling_progress", exist_ok=True)
        for frame_label, data in self.rmsd_history.items():
            if not data['epoch']:
                continue
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = data['epoch']
            # plot all candidates (orange, medium)
            for i, (ep, cands) in enumerate(zip(epochs, data['candidates'])):
                for j, rmsd in enumerate(cands):
                    label = 'Candidates' if i == 0 and j == 0 else None
                    ax.scatter(ep, rmsd, c='orange', marker='o', s=40, alpha=0.7, label=label)
            # plot best (green, star)
            ax.scatter(epochs, data['best'], label='Best (selected)', c='green', marker='*', s=100, zorder=5)
            # plot latent (blue, circle)
            ax.scatter(epochs, data['latent'], label='Latent', c='blue', marker='o', s=50, zorder=4)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('RMSD (Å)')
            ax.set_title(f'Frame {frame_label}: Latent vs Flow Candidates RMSD')
            ax.legend()
            ax.grid(True, alpha=0.3)
            # Calculate average RMSD of all candidates
            all_candidate_rmsds = [rmsd for cands in data['candidates'] for rmsd in cands]
            avg_candidate_rmsd = np.mean(all_candidate_rmsds)
            ax.text(0.02, 0.98, f'Avg candidate RMSD: {avg_candidate_rmsd:.3f} Å',
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            save_path = f"guided_sampling_progress/rmsd_latent_vs_final_frame_{frame_label}.png"
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            if self.logger and self.logger.experiment:
                self.logger.experiment.log({f"rmsd_latent_vs_candidates_frame_{frame_label}": wandb.Image(save_path)})
            plt.close(fig)
            print(f"RMSD comparison plot saved to {save_path}")

    def _save_loss_rmsd_correlation_plot(self):
        if not self.correlation_history['losses']:
            print("No correlation data to plot")
            return
        os.makedirs("guided_sampling_progress", exist_ok=True)

        losses = np.array(self.correlation_history['losses'])
        rmsds = np.array(self.correlation_history['rmsds'])
        epochs = np.array(self.correlation_history['epochs'])

        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(losses, rmsds, c=epochs, cmap='viridis', alpha=0.6, s=40)
        plt.colorbar(scatter, ax=ax, label='Epoch')

        # correlation coefficient
        corr = np.corrcoef(losses, rmsds)[0, 1]

        ax.set_xlabel('Render Loss')
        ax.set_ylabel('RMSD (Å)')
        ax.set_title(f'Loss vs RMSD Correlation (r={corr:.3f})')
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.98, f'Correlation: {corr:.3f}\nn={len(losses)} candidates',
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        save_path = "guided_sampling_progress/loss_rmsd_correlation.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if self.logger and self.logger.experiment:
            self.logger.experiment.log({"loss_rmsd_correlation": wandb.Image(save_path)})
        plt.close(fig)
        print(f"Loss-RMSD correlation plot saved to {save_path}")

    def _generate_flow_samples_plot(self, n_samples=1000, N=5):
        os.makedirs("guided_sampling_progress", exist_ok=True)
        print(f"Generating {n_samples} flow samples with N={N}...")
        with torch.no_grad():
            samples = self.flow_model.inference(batch_size=n_samples, N=N)
            samples_denorm = self.coords_scaler.reverse_transform(samples)

        fig, ax = plt.subplots(figsize=(10, 6))
        x_positions = []
        all_rmsds = []
        labels = []

        for param_idx, frame_label in self.indices_to_visualize:
            actual_frame_idx = self.frame_indices[param_idx]
            self.universe.trajectory[actual_frame_idx]
            gt_coords = torch.FloatTensor(self.universe.select_atoms('backbone').positions).to(self.device)

            rmsds = []
            for i in range(n_samples):
                sample_aligned = rmsdalign(gt_coords, samples_denorm[i])
                diff = sample_aligned - gt_coords
                rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1))).item()
                rmsds.append(rmsd)

            x_pos = len(x_positions)
            x_jitter = np.random.normal(0, 0.1, n_samples)
            ax.scatter(x_pos + x_jitter, rmsds, c='orange', marker='o', s=10, alpha=0.3)
            x_positions.append(x_pos)
            all_rmsds.append(rmsds)
            labels.append(f"Frame {frame_label}")

            # add latent rmsd for comparison
            latent_coords = self.parametric_structures[param_idx]
            latent_aligned = rmsdalign(gt_coords, latent_coords)
            latent_rmsd = torch.sqrt(torch.mean(torch.sum((latent_aligned - gt_coords) ** 2, dim=-1))).item()
            ax.scatter(x_pos, latent_rmsd, c='blue', marker='*', s=200, zorder=5, label='Latent' if x_pos == 0 else None)

        global_mean_rmsd = np.mean([v for rmsds in all_rmsds for v in rmsds])

        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels)
        ax.set_ylabel('RMSD (Å)')
        ax.set_title(
            f'Flow Model Samples (N={N}, {n_samples} samples) vs Ground Truth\n'
            f'Global mean RMSD = {global_mean_rmsd:.3f} Å'
        )
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        for i, rmsds in enumerate(all_rmsds):
            mean_rmsd = np.mean(rmsds)
            ax.axhline(y=mean_rmsd, xmin=(i-0.3+0.5)/len(x_positions), xmax=(i+0.3+0.5)/len(x_positions),
                      c='red', linestyle='--', linewidth=2)

        save_path = "guided_sampling_progress/flow_samples_rmsd.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        if self.logger and self.logger.experiment:
            self.logger.experiment.log({"flow_samples_rmsd": wandb.Image(save_path)})
        plt.close(fig)
        print(f"Flow samples RMSD plot saved to {save_path}")

    def on_train_end(self):
        print("Training ended. Training final flow model on latents (50 epochs)...")
        if self.flow_weight > 0 and not self._flow_model_pretrained:
            orig = self._flow_train_epochs
            self._flow_train_epochs = 50
            self._train_flow_model_on_latents()
            self._flow_train_epochs = orig
        self._create_optimization_gif()

    def on_before_optimizer_step(self, optimizer, optimizer_idx=None):
        # Freeze frame 0 — it's initialized from GT and loss steers it slightly wrong
        if self.parametric_structures.grad is not None:
            self.parametric_structures.grad[0].zero_()

        # Only log gradients for indices_to_visualize
        if not self.indices_to_visualize:
            return
        
        # Extract just the indices (first element of each tuple)
        structure_indices = [idx for idx, _ in self.indices_to_visualize]

        for i in structure_indices:
            grad = self.parametric_structures.grad
            if grad is not None:
                grad_i = grad[i]
                self.log(f'gradients/structure_{i}_norm', grad_i.norm().item())
                self.log(f'gradients/structure_{i}_mean', grad_i.abs().mean().item())
                if grad_i.shape[0] > 0:
                    self.log(f'gradients/structure_{i}_x', grad_i[0, 0].abs().item())
                    self.log(f'gradients/structure_{i}_y', grad_i[0, 1].abs().item())
                    self.log(f'gradients/structure_{i}_z', grad_i[0, 2].abs().item())
            else:
                self.log(f'gradients/structure_{i}_norm', 0.0)

    

    def configure_optimizers(self):
        params = []
        if not getattr(self, "_flow_model_frozen", False):
            params.append({'params': self.flow_model.parameters(), 'lr': self.flow_lr})
        params.append({'params': [self.parametric_structures], 'lr': self.structure_lr})
        optimizer = torch.optim.Adam(params)
        
        # Add gradient clipping to prevent vanishing gradients
        return {
            "optimizer": optimizer,
            "gradient_clip_val": 1.0,  # Clip gradients at 1.0
            "gradient_clip_algorithm": "norm"
        }
        
        # ReduceLROnPlateau scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, 
        #     mode='min',           # assuming you're monitoring loss
        #     patience=10,           # reduce LR after 10 epochs of no improvement
        #     factor=0.5,          # reduce LR by half
        #     min_lr=1e-6,         # don't go below this LR
        #     verbose=False         # print when LR changes
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "train_combined_loss",  # change to "val_loss" if you have validation
        #         "frequency": 1,
        #         "interval": "epoch"
        #     }
        # }
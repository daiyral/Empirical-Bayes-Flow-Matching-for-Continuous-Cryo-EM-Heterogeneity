import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import MDAnalysis as mda
import tempfile
import os
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandardScaler(nn.Module):
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.register_buffer('epsilon', torch.tensor(epsilon))

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.register_buffer('mean', torch.mean(values, dim=dims).detach())
        self.register_buffer('std', torch.std(values, dim=dims).detach())
        if self.epsilon.device != values.device:
            self.register_buffer('epsilon', self.epsilon.to(values.device))

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)
    
    def reverse_transform(self, values):
        return values * (self.std.to(values.device) + self.epsilon.to(values.device)) + self.mean.to(values.device)
    

class MDTrajectoryDataset(Dataset):
    def __init__(self, universe, scale=True, frame_indices=None):
        self.universe = universe
        self.backbone_atoms = universe.select_atoms('backbone')
        self.scale = scale
        
        # Use specific frame indices if provided, otherwise use all frames
        if frame_indices is not None:
            self.frame_indices = frame_indices
            self.n_frames = len(frame_indices)
        else:
            self.frame_indices = list(range(len(universe.trajectory)))
            self.n_frames = len(universe.trajectory)
        
        self.scaler = StandardScaler()
        self.scaler.fit(torch.FloatTensor(self.backbone_atoms.positions))

    def normalize_coords(self, coords):
        if self.scale:
            return self.scaler.transform(coords)
        return coords
    
    def reverse_transform(self, coords):
        if self.scale:
            return self.scaler.reverse_transform(coords)
        return coords

    def __len__(self):
        return self.n_frames

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.n_frames))
            return [self[i] for i in indices]
        
        # Map dataset index to actual trajectory frame index
        actual_frame_idx = self.frame_indices[idx]
        self.universe.trajectory[actual_frame_idx]
        backbone_coords = torch.FloatTensor(self.backbone_atoms.positions)
        backbone_coords = self.normalize_coords(backbone_coords)
        
        return backbone_coords, idx  # Return coordinates and dataset index (not actual frame index)


class HarmonicPrior:
    def __init__(self, N=256, a=3/(3.8**2), num_samples=1000, device=None, scale=True):
        device = device or DEVICE
        J = torch.zeros(N, N, device=device)
        for i, j in zip(np.arange(N-1), np.arange(1, N)):
            J[i, i] += a
            J[j, j] += a
            J[i, j] = J[j, i] = -a

        D, P = torch.linalg.eigh(J)
        D_inv = 1 / D
        D_inv[0] = 0

        self.P, self.D_inv = P, D_inv
        self.N = N
        self.device = device
        self.scale = scale

        self.scaler = StandardScaler()

        samples = self._generate_prior_samples(num_samples)
        self.scaler.fit(samples)

    def _generate_prior_samples(self, num_samples):
        samples = self.P @ (torch.sqrt(self.D_inv)[:, None] * 
                            torch.randn(num_samples, self.N, 3, device=self.device))
        return samples  # (num_samples, N, 3)

    def to(self, device):
        self.P = self.P.to(device)
        self.D_inv = self.D_inv.to(device)
        return self

    def sample(self, shape):
        batch_size = shape[0]
        seq_length = shape[1]
        samples = self.P @ (torch.sqrt(self.D_inv)[:, None] *
                            torch.randn(batch_size, self.N, 3, device=self.device))

        if self.scale:
            samples = self.scaler.transform(samples)

        return samples[:, :seq_length]

    def reverse_transform(self, normalized_samples):
        return self.scaler.reverse_transform(normalized_samples)


class ParametricStructureDataset(Dataset):
    def __init__(self, parametric_coords, pdb_file, scale=True):
        self.coords = parametric_coords
        self.n_frames = len(parametric_coords)
        self.scale = scale
        
        # Convert parametric_coords to numpy if it's a tensor
        if isinstance(parametric_coords, torch.Tensor):
            coords_np = parametric_coords.cpu().numpy()
        else:
            coords_np = parametric_coords
        
        # Create a universe from the PDB template
        template_universe = mda.Universe(pdb_file)
        template_backbone = template_universe.select_atoms('backbone')
        
        # Create temporary PDB and XTC files for backbone only
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp_pdb:
            tmp_pdb_path = tmp_pdb.name
        with tempfile.NamedTemporaryFile(suffix='.xtc', delete=False) as tmp_xtc:
            tmp_xtc_path = tmp_xtc.name
        
        try:
            # Write backbone-only PDB
            template_backbone.write(tmp_pdb_path)
            
            # Create a universe with backbone-only PDB to write XTC
            backbone_universe = mda.Universe(tmp_pdb_path)
            
            # Write all parametric structures to the temporary XTC file
            with mda.coordinates.XTC.XTCWriter(tmp_xtc_path, backbone_universe.atoms.n_atoms, dt=1.0) as W:
                for coords in coords_np:
                    backbone_universe.atoms.positions = coords
                    W.write(backbone_universe.atoms)
            
            # Load the universe with the XTC trajectory
            self.universe = mda.Universe(tmp_pdb_path, tmp_xtc_path)
            self.backbone_atoms = self.universe.select_atoms('backbone')
        finally:
            # Clean up the temporary files
            for tmp_file in [tmp_pdb_path, tmp_xtc_path]:
                try:
                    if os.path.exists(tmp_file):
                        os.unlink(tmp_file)
                except PermissionError:
                    pass  # File might still be in use, will be cleaned up later
        
        if scale:
            self.scaler = StandardScaler()
            # Fit on all coordinates
            all_coords = parametric_coords.reshape(-1, 3)
            self.scaler.fit(all_coords)
        else:
            self.scaler = None
    
    def normalize_coords(self, coords):
        if self.scale and self.scaler is not None:
            return self.scaler.transform(coords)
        return coords
    
    def reverse_transform(self, coords):
        if self.scale and self.scaler is not None:
            return self.scaler.reverse_transform(coords)
        return coords
    
    def __len__(self):
        return self.n_frames
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            indices = range(*idx.indices(self.n_frames))
            return [self[i] for i in indices]
        
        coords = self.coords[idx]
        if self.scale:
            coords = self.normalize_coords(coords)
        return coords, idx
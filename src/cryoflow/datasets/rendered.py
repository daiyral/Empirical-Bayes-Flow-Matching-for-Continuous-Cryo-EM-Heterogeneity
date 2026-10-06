import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from cryoflow.datasets.md import StandardScaler
from tqdm import tqdm


import torch
import numpy as np
from torch.utils.data import Dataset
from cryoflow.rendering.gaussian import Renderer
from cryoflow.datasets.md import StandardScaler
from tqdm import tqdm
from cryoflow.rendering.cryojax import CryoJAXRenderer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RenderedMDTrajectoryDataset(Dataset):
    def __init__(self, universe, frame_indices=None, images_per_frame=100, sigma=5.0, seed=42, scale=False, batch_size_per_frame=None, noisy=True):
        self.universe = universe
        self.backbone_atoms = universe.select_atoms('backbone')
        self.images_per_frame = images_per_frame
        self.sigma = sigma
        self.scale = scale
        self.batch_size_per_frame = batch_size_per_frame if batch_size_per_frame is not None else images_per_frame
        
        # Use specific frame indices if provided, otherwise use all frames
        if frame_indices is not None:
            self.frame_indices = frame_indices
            self.n_frames = len(frame_indices)
        else:
            self.frame_indices = list(range(len(universe.trajectory)))
            self.n_frames = len(universe.trajectory)
        
        # Initialize scaler for 3D coordinates
        #self.scaler = StandardScaler()
        #self.scaler.fit(torch.FloatTensor(self.backbone_atoms.positions))
        
        # Initialize renderer
        self.renderer = CryoJAXRenderer()
        self.noisy = noisy

        # Store data organized by frame for O(1) access
        self.frame_images = {}  # Maps frame_idx -> stacked tensor of images [images_per_frame, H, W]
        self.frame_angles = {}  # Maps frame_idx -> list of angle tuples
        self.coords_3d = {}     # Maps frame_idx -> coordinates tensor

        np.random.seed(seed)
        torch.manual_seed(seed)

        noise_seed = 0
        for frame_idx in tqdm(range(self.n_frames), desc="Pre-rendering images"):
            actual_frame_idx = self.frame_indices[frame_idx]
            self.universe.trajectory[actual_frame_idx]

            # Get and store 3D coordinates
            backbone_coords = torch.FloatTensor(self.backbone_atoms.positions)

            # Lists to accumulate images and angles for this frame
            frame_image_list = []
            frame_angle_list = []

            # Render from multiple angles
            for img_idx in range(images_per_frame):
                # Generate random angles
                angles = (
                    float(torch.rand(1) * 360),
                    float(torch.rand(1) * 360),
                    float(torch.rand(1) * 360)
                )

                if noisy:
                    rendered_img, _ = self.renderer.render_particle_noisy(
                        backbone_coords, angles, sigma=self.sigma, seed=noise_seed
                    )
                    noise_seed += 1
                else:
                    rendered_img, _ = self.renderer.render_particle(
                        backbone_coords, angles, sigma=self.sigma
                    )
                
                # Accumulate for this frame
                frame_image_list.append(rendered_img.cpu())
                frame_angle_list.append(angles)
            
            # Stack all images for this frame and store in dict
            self.frame_images[frame_idx] = torch.stack(frame_image_list)  # [images_per_frame, H, W]
            # Convert angles list to tensor of shape [images_per_frame, 3]
            self.frame_angles[frame_idx] = torch.tensor(frame_angle_list, dtype=torch.float32)
            self.coords_3d[frame_idx] = backbone_coords
        
        total_images = self.n_frames * images_per_frame
        print(f"Pre-rendering complete! Total images: {total_images}")
        
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
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]
        
        # Randomly sample a subset of images for this frame (SGD mini-batch)
        num_available = self.frame_images[idx].shape[0]  # images_per_frame
        
        if self.batch_size_per_frame >= num_available:
            # Use all images if batch size >= total images
            sampled_images = self.frame_images[idx]
            sampled_angles = self.frame_angles[idx]
        else:
            # Randomly sample batch_size_per_frame images
            sample_indices = torch.randperm(num_available)[:self.batch_size_per_frame]
            sampled_images = self.frame_images[idx][sample_indices]
            sampled_angles = self.frame_angles[idx][sample_indices]
        
        return sampled_images, idx, sampled_angles, self.coords_3d[idx]
    
    def get_frame_coords(self, frame_idx):
        return self.coords_3d[frame_idx]
    
    def get_images_for_frame(self, frame_idx):
        return self.frame_images[frame_idx], self.frame_angles[frame_idx]



class _BackboneAtomsMock:
    def __init__(self, positions):
        self.positions = positions
    def __len__(self):
        return self.positions.shape[0]


class TemSimulatorDataset(Dataset):
    def __init__(self, data_dir, batch_size_per_frame=None, scale=False, universe=None):
        self.data_dir = data_dir
        self.scale = scale

        with open(os.path.join(data_dir, "metadata.json")) as f:
            self.metadata = json.load(f)

        self.n_frames = self.metadata["n_frames"]
        self.frame_files = self.metadata["frame_files"]

        self.frame_images = {}
        self.frame_angles = {}
        self.coords_3d = {}

        for i, fname in enumerate(tqdm(self.frame_files, desc="Loading pre-rendered frames")):
            data = np.load(os.path.join(data_dir, fname))
            self.frame_images[i] = torch.from_numpy(data["images"])
            self.frame_angles[i] = torch.from_numpy(data["angles"])
            self.coords_3d[i] = torch.from_numpy(data["coords_3d"])

        self.images_per_frame = self.frame_images[0].shape[0]
        self.batch_size_per_frame = batch_size_per_frame if batch_size_per_frame is not None else self.images_per_frame

        if universe is not None:
            self.backbone_atoms = universe.select_atoms('backbone')
        else:
            self.backbone_atoms = _BackboneAtomsMock(self.coords_3d[0].numpy())

        if scale:
            self.scaler = StandardScaler()
            self.scaler.fit(self.coords_3d[0])

        total_images = sum(self.frame_images[i].shape[0] for i in range(self.n_frames))
        print(f"Loaded {self.n_frames} frames, {total_images} total images from {data_dir}")

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
            indices = range(*idx.indices(len(self)))
            return [self[i] for i in indices]

        num_available = self.frame_images[idx].shape[0]

        if self.batch_size_per_frame >= num_available:
            sampled_images = self.frame_images[idx]
            sampled_angles = self.frame_angles[idx]
        else:
            sample_indices = torch.randperm(num_available)[:self.batch_size_per_frame]
            sampled_images = self.frame_images[idx][sample_indices]
            sampled_angles = self.frame_angles[idx][sample_indices]

        return sampled_images, idx, sampled_angles, self.coords_3d[idx]

    def get_frame_coords(self, frame_idx):
        return self.coords_3d[frame_idx]

    def get_images_for_frame(self, frame_idx):
        return self.frame_images[frame_idx], self.frame_angles[frame_idx]


def _random_so3_zyz_degrees(rng, n):
    phi = rng.uniform(-180.0, 180.0, size=n)
    theta = np.degrees(np.arccos(rng.uniform(-1.0, 1.0, size=n)))
    psi = rng.uniform(-180.0, 180.0, size=n)
    return np.stack([phi, theta, psi], axis=-1).astype(np.float32)


def _perturbed_zyz_degrees(rng, angles, max_deg):
    from scipy.spatial.transform import Rotation as R
    n = angles.shape[0]
    axis = rng.normal(size=(n, 3))
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    mag = rng.uniform(0.5 * max_deg, max_deg, size=(n, 1))
    perturb = R.from_rotvec(np.radians(axis * mag))
    base = R.from_euler('ZYZ', angles, degrees=True)
    return (perturb * base).as_euler('ZYZ', degrees=True).astype(np.float32)


class CorruptedTemSimulatorDataset(TemSimulatorDataset):
    def __init__(self, data_dir, batch_size_per_frame=None, scale=False, universe=None,
                 pose_error_frac=0.0, ctf_error_frac=0.0, pose_error_deg=0.0,
                 ctf_defocus_error=0.4, base_defocus=5000.0, error_seed=0):
        super().__init__(data_dir, batch_size_per_frame=batch_size_per_frame,
                         scale=scale, universe=universe)

        self.pose_error_frac = pose_error_frac
        self.ctf_error_frac = ctf_error_frac
        self.base_defocus = base_defocus

        n, k = self.n_frames, self.images_per_frame
        rng = np.random.default_rng(error_seed)

        # Independent draws: p is a per-particle corruption rate for each channel.
        pose_mask = rng.random((n, k)) < pose_error_frac
        ctf_mask = rng.random((n, k)) < ctf_error_frac

        if pose_mask.any():
            flat = pose_mask.reshape(-1)
            angles = np.stack([self.frame_angles[i].numpy() for i in range(n)]).reshape(-1, 3)
            if pose_error_deg > 0:
                angles[flat] = _perturbed_zyz_degrees(rng, angles[flat], pose_error_deg)
            else:
                angles[flat] = _random_so3_zyz_degrees(rng, int(flat.sum()))
            angles = angles.reshape(n, k, 3)
            for i in range(n):
                self.frame_angles[i] = torch.from_numpy(angles[i].copy())

        defocus = np.full((n, k), base_defocus, dtype=np.float32)
        if ctf_mask.any():
            mag = rng.uniform(0.5 * ctf_defocus_error, ctf_defocus_error, size=(n, k))
            sign = rng.choice([-1.0, 1.0], size=(n, k))
            defocus = np.where(ctf_mask, base_defocus * (1.0 + sign * mag), defocus).astype(np.float32)
        self.defocus = torch.from_numpy(defocus)
        self.pose_mask = torch.from_numpy(pose_mask)
        self.ctf_mask = torch.from_numpy(ctf_mask)

        print(f"Corrupted metadata: {int(pose_mask.sum())}/{n * k} poses "
              f"({'random' if pose_error_deg <= 0 else f'{pose_error_deg} deg'}), "
              f"{int(ctf_mask.sum())}/{n * k} defocus values "
              f"(±{ctf_defocus_error:.0%} of {base_defocus:.0f} A)")

    def get_defocus(self, frame_indices):
        return self.defocus[frame_indices]

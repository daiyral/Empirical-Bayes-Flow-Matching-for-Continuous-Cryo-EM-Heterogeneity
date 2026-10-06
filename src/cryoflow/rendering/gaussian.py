import numpy as np
import torch


class Renderer:
    def __init__(self):
        self.image_size = (64, 64)       # 64 px × 64 px (particle size)
        self.pixel_size = 1.2            # 1.2 Å
        self.voltage = 300               # 300 keV
        self.defocus = 5000              # 5,000 Å
        self.cs = 2.1                    # 2.1 nm (spherical aberration) CS in nm, converted to Å in ctf (cs*10)
        self.amplitude_contrast = 0.06   # Amplitude contrast ratio
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create coordinate grids for x and y (centered at image center)
        y_coords_1d = (np.arange(self.image_size[0]) - self.image_size[0]/2) * self.pixel_size
        x_coords_1d = (np.arange(self.image_size[1]) - self.image_size[1]/2) * self.pixel_size
        
        x_grid_np, y_grid_np = np.meshgrid(x_coords_1d, y_coords_1d)
        
        self.x_grid = torch.tensor(x_grid_np, dtype=torch.float32).to(self.device)
        self.y_grid = torch.tensor(y_grid_np, dtype=torch.float32).to(self.device)
    
    def gaussian_projection(self, positions, sigma=1.0):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device)
            
        projected_image = torch.zeros(self.image_size, dtype=torch.float32, device=self.device)
        
        atom_x_coords = positions[:, 0].view(-1, 1, 1) # (N_atoms, 1, 1)
        atom_y_coords = positions[:, 1].view(-1, 1, 1) # (N_atoms, 1, 1)
        
        grid_x = self.x_grid.unsqueeze(0) # (1, H, W)
        grid_y = self.y_grid.unsqueeze(0) # (1, H, W)
        
        r_squared = (grid_x - atom_x_coords)**2 + (grid_y - atom_y_coords)**2
        
        pi_tensor = torch.tensor(torch.pi, dtype=torch.float32, device=self.device)
        all_atom_projections = torch.exp(-0.5 * r_squared / sigma**2)
        
        projected_image = torch.sum(all_atom_projections, dim=0)
            
        return projected_image

    def center_protein(self, positions):
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device)
            
        com = torch.mean(positions, dim=0, keepdim=True)
        centered = positions - com
        
        if positions.requires_grad:
            centered = centered + (positions.sum() * 0.0) # Ensure gradient link
            
        return centered

    def apply_ctf(self, image):
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=self.device)
        
        requires_grad_input = image.requires_grad
        
        fy = torch.fft.fftfreq(self.image_size[0], d=self.pixel_size, device=self.device)
        fx = torch.fft.fftfreq(self.image_size[1], d=self.pixel_size, device=self.device)
        
        fy_grid = fy.view(-1, 1).expand(self.image_size)
        fx_grid = fx.view(1, -1).expand(self.image_size)
        f_squared = fx_grid**2 + fy_grid**2
        
        wavelength = 12.26 / torch.sqrt(torch.tensor(self.voltage * (1.0 + self.voltage * 0.978466e-6), 
                                                    dtype=torch.float32, device=self.device))
        
        chi = torch.pi * wavelength * f_squared * (0.5 * self.defocus - 0.25 * wavelength**2 * f_squared * self.cs * 10)
        
        amplitude_contrast_tensor = torch.tensor(self.amplitude_contrast, dtype=torch.float32, device=self.device)
        ctf_val = -torch.sqrt(1 - amplitude_contrast_tensor**2) * torch.sin(chi) - amplitude_contrast_tensor * torch.cos(chi)
        
        img_fft = torch.fft.fft2(image)
        ctf_applied_fft = img_fft * ctf_val
        ctf_result = torch.real(torch.fft.ifft2(ctf_applied_fft))
        
        if requires_grad_input:
            ctf_result = ctf_result + (image.sum() * 0.0) # Ensure gradient link
        
        return ctf_result

    def torch_rotation_matrix(self, angles):
        angles_rad = torch.tensor(angles, dtype=torch.float32, device=self.device) * (torch.pi / 180.0)
        alpha, beta, gamma = angles_rad[0], angles_rad[1], angles_rad[2]
        
        cos_a, sin_a = torch.cos(alpha), torch.sin(alpha)
        R_x = torch.zeros((3, 3), device=self.device)
        R_x[0, 0] = 1.0
        R_x[1, 1] = cos_a; R_x[1, 2] = -sin_a
        R_x[2, 1] = sin_a; R_x[2, 2] = cos_a
        
        cos_b, sin_b = torch.cos(beta), torch.sin(beta)
        R_y = torch.zeros((3, 3), device=self.device)
        R_y[0, 0] = cos_b; R_y[0, 2] = sin_b
        R_y[1, 1] = 1.0
        R_y[2, 0] = -sin_b; R_y[2, 2] = cos_b
        
        cos_g, sin_g = torch.cos(gamma), torch.sin(gamma)
        R_z = torch.zeros((3, 3), device=self.device)
        R_z[0, 0] = cos_g; R_z[0, 1] = -sin_g
        R_z[1, 0] = sin_g; R_z[1, 1] = cos_g
        R_z[2, 2] = 1.0
        
        # Combined rotation matrix: R = R_z @ R_y @ R_x
        R = torch.matmul(R_z, torch.matmul(R_y, R_x))
        return R
        
    def render_particle(self, atom_positions, angles, sigma=1.0):

        atom_positions = atom_positions.to(self.device)
        #center the protein
        centered_positions = self.center_protein(atom_positions)
        
        # apply rotation
        rotation_matrix = self.torch_rotation_matrix(angles) # angles is tuple (degrees)
        rotated_positions = torch.matmul(centered_positions, rotation_matrix.t())
        
        # 2d projection
        projected_coords = rotated_positions[:, :2]
        
        # 2d gaussians summed 
        projected_image = self.gaussian_projection(projected_coords, sigma=sigma)

        # ctf
        #ctf_applied_image = self.apply_ctf(projected_image)
        
        # normalize
        #min_val = torch.min(ctf_applied_image)
        #max_val = torch.max(ctf_applied_image)
        #normalized_image = (ctf_applied_image - min_val) / (max_val - min_val + 1e-8)

        result_image = projected_image
        #result_image = ctf_applied_image

        return result_image, angles

    def _batch_rotation_matrix(self, angles):
        # angles: (B, 3) in degrees, intrinsic XYZ-style consistent with torch_rotation_matrix
        angles_rad = angles.to(self.device).float() * (torch.pi / 180.0)
        a, b, g = angles_rad[:, 0], angles_rad[:, 1], angles_rad[:, 2]
        B = angles_rad.shape[0]
        zero = torch.zeros(B, device=self.device)
        one = torch.ones(B, device=self.device)

        ca, sa = torch.cos(a), torch.sin(a)
        cb, sb = torch.cos(b), torch.sin(b)
        cg, sg = torch.cos(g), torch.sin(g)

        R_x = torch.stack([
            torch.stack([one, zero, zero], dim=-1),
            torch.stack([zero, ca, -sa], dim=-1),
            torch.stack([zero, sa, ca], dim=-1),
        ], dim=-2)

        R_y = torch.stack([
            torch.stack([cb, zero, sb], dim=-1),
            torch.stack([zero, one, zero], dim=-1),
            torch.stack([-sb, zero, cb], dim=-1),
        ], dim=-2)

        R_z = torch.stack([
            torch.stack([cg, -sg, zero], dim=-1),
            torch.stack([sg, cg, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1),
        ], dim=-2)

        return R_z @ R_y @ R_x  # (B, 3, 3)

    def gaussian_projection_batch(self, positions, sigma=1.0):
        # positions: (B, N, 2)
        positions = positions.to(self.device)
        atom_x = positions[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        atom_y = positions[..., 1].unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)

        grid_x = self.x_grid.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        grid_y = self.y_grid.unsqueeze(0).unsqueeze(0)

        r_squared = (grid_x - atom_x) ** 2 + (grid_y - atom_y) ** 2  # (B, N, H, W)
        return torch.exp(-0.5 * r_squared / sigma ** 2).sum(dim=1)   # (B, H, W)

    def render_particle_batch(self, atom_positions, angles, sigma=1.0):
        # atom_positions: (B, N, 3), angles: (B, 3) in degrees
        atom_positions = atom_positions.to(self.device)
        angles = angles.to(self.device) if isinstance(angles, torch.Tensor) else torch.as_tensor(angles, device=self.device)

        centered = atom_positions - atom_positions.mean(dim=1, keepdim=True)
        R = self._batch_rotation_matrix(angles)              # (B, 3, 3)
        rotated = torch.matmul(centered, R.transpose(-1, -2))  # (B, N, 3)
        projected = self.gaussian_projection_batch(rotated[..., :2], sigma=sigma)
        return projected, angles

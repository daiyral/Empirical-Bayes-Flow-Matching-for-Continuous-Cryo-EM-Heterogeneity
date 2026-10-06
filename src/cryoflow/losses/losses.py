import torch
import torch.nn as nn
import torch.nn.functional as F


def patch_mean(images, patch_shape):
    channels, *patch_size = patch_shape
    dimensions = len(patch_size)
    padding = tuple(side // 2 for side in patch_size)

    conv = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]

    patch_elements = torch.Tensor(patch_size).prod().item()
    weights = torch.full((channels, channels, *patch_size), fill_value=1 / patch_elements)
    weights = weights.to(images.device)

    channel_selector = torch.eye(channels).bool()
    weights[~channel_selector] = 0

    result = conv(images, weights, padding=padding, bias=None)
    return result


def patch_std(image, patch_shape):
    return (patch_mean(image**2, patch_shape) - patch_mean(image, patch_shape)**2).sqrt()


def channel_normalize(template):
    reshaped_template = template.clone().view(template.shape[0], -1)
    reshaped_template.sub_(reshaped_template.mean(dim=-1, keepdim=True))
    reshaped_template.div_(reshaped_template.std(dim=-1, keepdim=True, unbiased=False))
    return reshaped_template.view_as(template)

class FlowMatchingLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_pairwise=0.0, w_fape=0.0):
        super().__init__()
        self.w_mse = w_mse  # MSE weight
        self.w_pairwise = w_pairwise  # Pairwise atomic distance loss weight
        self.w_fape = w_fape  # FAPE loss weight
    
    def mse_loss(self, pred, target):
        return F.mse_loss(pred, target)
    
    def compute_fape(
        self,
        pred_positions: torch.Tensor,
        target_positions: torch.Tensor,
        length_scale: float = 10.0,
        l1_clamp_distance: float = 10.0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        
        # Calculate pairwise distances between all positions
        delta_pred = pred_positions[:, :, None] - pred_positions[:, None, :]
        delta_target = target_positions[:, :, None] - target_positions[:, None, :]
        
        # Calculate error distances
        error_dist = torch.sqrt(
            torch.sum((delta_pred - delta_target) ** 2, dim=-1) + eps
        )
        
        if l1_clamp_distance is not None:
            error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)
        
        # Normalize error
        normed_error = error_dist / length_scale
        
        # Average over all pairs
        loss = torch.mean(normed_error)
        
        return loss ** 2
    
    def pairwise_distance_loss(self, pred, target, threshold=0.8):
        true_distances = torch.cdist(target, target, p=2)  # True pairwise distances
        pred_distances = torch.cdist(pred, pred, p=2)  # Predicted pairwise distances
        mask = (true_distances < threshold)  # Only consider close atom pairs
        
        loss = torch.mean((true_distances[mask] - pred_distances[mask]) ** 2) if torch.any(mask) else torch.tensor(0.0, device=pred.device)
        return loss
    
    def forward(self, pred_positions, target_positions):
        mse_loss = self.mse_loss(pred_positions, target_positions)
        pairwise_loss = self.pairwise_distance_loss(pred_positions, target_positions) if self.w_pairwise > 0 else torch.tensor(0.0, device=pred_positions.device)
        fape_loss = self.compute_fape(pred_positions, target_positions) if self.w_fape > 0 else torch.tensor(0.0, device=pred_positions.device)
        
        total_loss = (
            self.w_mse * mse_loss +
            self.w_pairwise * pairwise_loss +
            self.w_fape * fape_loss
        )
        
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'pairwise_loss': pairwise_loss.item(),
            'fape_loss': fape_loss.item()
        }


class NCC(nn.Module):
    def __init__(self, template, keep_channels=False):
        super().__init__()
        self.keep_channels = keep_channels

        channels, *template_shape = template.shape
        dimensions = len(template_shape)
        self.padding = tuple(side // 2 for side in template_shape)

        self.conv_f = (F.conv1d, F.conv2d, F.conv3d)[dimensions - 1]
        self.normalized_template = channel_normalize(template)
        ones = template.dim() * (1, )
        self.normalized_template = self.normalized_template.repeat(channels, *ones)

        channel_selector = torch.eye(channels).bool()
        self.normalized_template[~channel_selector] = 0

        patch_elements = torch.Tensor(template_shape).prod().item()
        self.normalized_template.div_(patch_elements)

    def forward(self, image):
        result = self.conv_f(image, self.normalized_template, padding=self.padding, bias=None)
        std = patch_std(image, self.normalized_template.shape[1:])
        result.div_(std)
        if not self.keep_channels:
            result = result.mean(dim=1)
        return result


class PatchNCCLoss(nn.Module):
    def forward(self, pred, target):
        # Add batch and channel dims if needed
        if pred.ndim == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        # Use target as template, compute NCC response on pred
        ncc_module = NCC(target, keep_channels=False).to(pred.device)
        ncc_response = ncc_module(pred)

        # Return 1 - max(NCC) as loss
        return 1.0 - ncc_response.max()


class RenderingLoss(nn.Module):
    def __init__(self, loss_type='mae'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'ncc':
            self.loss_fn = PatchNCCLoss()

    def forward(self, pred, target):
        if self.loss_type == 'mse':
            return F.mse_loss(pred, target)
        elif self.loss_type == 'mae':
            return F.l1_loss(pred, target)
        elif self.loss_type == 'ncc':
            return self.loss_fn(pred, target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        

class PairwiseDistanceRegularizer(nn.Module):
    def __init__(self, initial_coords, max_dist=6.0):
        super().__init__()
        if not isinstance(initial_coords, torch.Tensor):
            initial_coords = torch.tensor(initial_coords, dtype=torch.float32)
        dists = torch.cdist(initial_coords.unsqueeze(0), initial_coords.unsqueeze(0)).squeeze(0)
        mask = (dists < max_dist) & (dists > 0.1)
        self.register_buffer('init_dists', dists)
        self.register_buffer('mask', mask)

    def forward(self, positions):
        if positions.ndim == 2:
            pred_dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
            return F.mse_loss(pred_dists[self.mask], self.init_dists[self.mask])
        else:
            pred_dists = torch.cdist(positions, positions)  # [B, N, N]
            target = self.init_dists[self.mask].unsqueeze(0).expand(positions.shape[0], -1)
            return F.mse_loss(pred_dists[:, self.mask], target)
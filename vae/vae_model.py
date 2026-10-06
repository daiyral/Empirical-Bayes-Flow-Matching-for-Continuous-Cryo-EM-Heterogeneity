import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import frames as fr
from renderer import GMMRenderer

# Rosenbaum et al. 2021 (arXiv:2106.14108), Figure 10
D1 = 512    # encoder representation dim
D2 = 32     # conformation latent dim
D3 = 512    # conformation representation dim


def softplus(x):
    return F.softplus(x) + 1e-4


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x):
        return x + self.lin2(F.leaky_relu(self.lin1(self.norm(x))))


class DanVAE(nn.Module):
    # Known-pose variant: the per-image viewing angles are given to the renderer, so the
    # only latent is the conformation; the decoder predicts a rigid delta frame per residue.
    def __init__(self, base_frames, image_size=64, image_sigma=1.0, beta_conf=1.0,
                 w_center=0.1, w_backbone=3000.0, w_enm=100.0, enm_tol=1.0,
                 resnet_width=256, resnet_blocks=5):
        super().__init__()
        self.image_size = image_size
        self.n_pix = image_size * image_size
        self.image_sigma = image_sigma
        self.beta_conf = beta_conf
        self.w_center = w_center
        self.w_backbone = w_backbone
        self.w_enm = w_enm
        self.enm_tol = enm_tol

        self.n_res = base_frames["n_res"]
        self.n_atoms = base_frames["n_atoms"]
        self.register_buffer("R_base", base_frames["R_base"])
        self.register_buffer("t_base", base_frames["t_base"])
        self.register_buffer("local_atoms", base_frames["local_atoms"])
        self.register_buffer("atom2res", base_frames["atom2res"])
        for k in ("bond_i", "bond_j", "bond_target", "enm_i", "enm_j", "enm_target"):
            self.register_buffer(k, base_frames[k], persistent=False)

        # encoder
        self.img_norm = nn.LayerNorm(self.n_pix)
        enc_sizes = [self.n_pix, 2048, 1024, 512, D1]
        enc_layers = []
        for i in range(len(enc_sizes) - 1):
            enc_layers.append(nn.Linear(enc_sizes[i], enc_sizes[i + 1]))
            enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        # conformation posterior
        self.conf_norm = nn.LayerNorm(D1)
        self.conf_mu = nn.Linear(D1, D2)
        self.conf_sigma = nn.Linear(D1, D2)

        # conformation decoder
        self.conf_in = nn.Linear(D2, resnet_width)
        self.resblocks = nn.ModuleList([ResBlock(resnet_width) for _ in range(resnet_blocks)])
        self.conf_out_norm = nn.LayerNorm(resnet_width)
        self.conf_repr = nn.Linear(resnet_width, D3)
        self.frame_out = nn.Linear(D3, 9 * self.n_res)

        self._init_weights()
        self.renderer = GMMRenderer()

    def _init_weights(self):
        # truncated-normal init scaled by 1/sqrt(fan_in); zero-init residual updates
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=1.0 / math.sqrt(m.weight.shape[1]))
                nn.init.zeros_(m.bias)
        for blk in self.resblocks:
            nn.init.zeros_(blk.lin2.weight)
            nn.init.zeros_(blk.lin2.bias)
        nn.init.zeros_(self.frame_out.weight)   # start at the base conformation
        nn.init.zeros_(self.frame_out.bias)

    def encode(self, images):
        x = self.img_norm(images.reshape(images.shape[0], self.n_pix))
        h = self.conf_norm(self.encoder(x))
        return self.conf_mu(h), softplus(self.conf_sigma(h))

    def decode_conformation(self, z_conf):
        # z_conf [B, CS, D2] -> centred coords [B, CS, n_atoms, 3]
        B, CS, _ = z_conf.shape
        h = self.conf_in(z_conf.reshape(B * CS, D2))
        for blk in self.resblocks:
            h = blk(h)
        conf_repr = self.conf_repr(self.conf_out_norm(h))
        delta9 = self.frame_out(conf_repr).reshape(B * CS, self.n_res, 9)
        coords = fr.apply_delta_frames(delta9, self.R_base, self.t_base,
                                       self.local_atoms, self.atom2res)
        coords = coords.reshape(B, CS, self.n_atoms, 3)
        return coords - coords.mean(dim=2, keepdim=True)

    @staticmethod
    def reparam(mu, sigma, n_samples):
        shape = mu.shape[:-1] + (n_samples, mu.shape[-1])
        eps = torch.randn(shape, device=mu.device)
        return mu.unsqueeze(-2) + sigma.unsqueeze(-2) * eps

    @staticmethod
    def kl_normal(mu, sigma):
        return 0.5 * (mu.pow(2) + sigma.pow(2) - 2 * torch.log(sigma) - 1).sum(-1)

    def image_nll(self, pred, target):
        # Gaussian negative log-likelihood summed over pixels
        se = (pred - target).pow(2).sum(dim=(-2, -1))
        return se / (2 * self.image_sigma ** 2)

    def compute_losses(self, images, angles, sigma_render, conf_samples):
        # images [B, H, W]; angles [B, 3] ZYZ-intrinsic degrees (true viewing angles)
        B, CS = images.shape[0], conf_samples
        conf_mu, conf_sigma = self.encode(images)
        z_conf = self.reparam(conf_mu, conf_sigma, CS)                    # [B, CS, D2]
        conf_coords = self.decode_conformation(z_conf)                   # [B, CS, n_atoms, 3]

        # each sample is rendered at the image's true angles (cryojax centres, then rotates)
        coords_flat = conf_coords.reshape(B * CS, self.n_atoms, 3)
        angles_flat = angles.unsqueeze(1).expand(B, CS, 3).reshape(B * CS, 3)
        pred_imgs = self.renderer.render_particle_batch(coords_flat, angles_flat, sigma=sigma_render)
        pred_imgs = pred_imgs.reshape(B, CS, self.image_size, self.image_size)

        # IWAE reconstruction: -log( 1/CS sum_n P(y|z_n) )
        nll = self.image_nll(pred_imgs, images.unsqueeze(1))             # [B, CS]
        recon = -(torch.logsumexp(-nll, dim=1) - math.log(CS))
        recon = recon.mean()

        kl_conf = self.kl_normal(conf_mu, conf_sigma).mean()

        # structure priors
        center_loss = conf_coords.mean(dim=2).pow(2).sum(-1).mean()
        bl = (conf_coords[..., self.bond_i, :] - conf_coords[..., self.bond_j, :]).norm(dim=-1)
        backbone_loss = F.mse_loss(bl, self.bond_target.expand_as(bl))
        # flat-bottomed elastic network: no penalty within enm_tol of the base contact distance
        ed = (conf_coords[..., self.enm_i, :] - conf_coords[..., self.enm_j, :]).norm(dim=-1)
        dev = (ed - self.enm_target).abs()
        if self.enm_tol > 0:
            dev = F.relu(dev - self.enm_tol)
        enm_loss = dev.pow(2).mean()

        total = (recon
                 + self.beta_conf * kl_conf
                 + self.w_center * center_loss
                 + self.w_backbone * backbone_loss
                 + self.w_enm * enm_loss)

        logdict = {
            "train_recon": recon.item(),
            "train_kl_conf": kl_conf.item(),
            "train_backbone_loss": backbone_loss.item(),
            "train_enm_loss": enm_loss.item(),
            "train_combined_loss": total.item(),
        }
        return total, logdict

    @torch.no_grad()
    def sample_structures(self, n, batch_size=256):
        # unconditional prior samples -> coords [n, n_atoms, 3]
        self.eval()
        out = []
        for i in range(0, n, batch_size):
            b = min(batch_size, n - i)
            z = torch.randn(b, 1, D2, device=self.R_base.device)
            out.append(self.decode_conformation(z)[:, 0].cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def reconstruct(self, images):
        # posterior-mean reconstruction -> coords [B, n_atoms, 3]
        self.eval()
        conf_mu, _ = self.encode(images)
        return self.decode_conformation(conf_mu.unsqueeze(1))[:, 0]

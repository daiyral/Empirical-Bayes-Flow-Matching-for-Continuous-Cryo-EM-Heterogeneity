import time

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion import create_diffusion
from utils import DATA_ROOT, load_mnist


class FourierEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, scale=20.0):
        super().__init__()
        self.register_buffer('B', torch.randn(in_channels, out_channels // 2) * scale)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, groups=8):
        super().__init__()
        g1 = min(groups, in_ch)
        g2 = min(groups, out_ch)
        self.norm1 = nn.GroupNorm(g1, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.norm2 = nn.GroupNorm(g2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttentionGate(nn.Module):
    def __init__(self, skip_ch, gate_ch, inter_ch):
        super().__init__()
        self.W_skip = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.W_gate = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, skip, gate):
        a = F.silu(self.W_skip(skip) + self.W_gate(gate))
        return skip * torch.sigmoid(self.psi(a))


class FullImageFlowModel(nn.Module):
    def __init__(self, in_channels=1, base_ch=16, time_dim=64, t_eps=0.05):
        super().__init__()
        self.t_eps = t_eps
        self.time_emb = FourierEmbedding(1, 64, scale=20.0)
        self.time_mlp = nn.Sequential(
            nn.Linear(64, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.in_conv = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, time_dim)
        self.down2 = ResBlock(c1, c2, time_dim)
        self.pool = nn.AvgPool2d(2)
        self.mid1 = ResBlock(c2, c3, time_dim)
        self.mid2 = ResBlock(c3, c3, time_dim)
        self.att1 = AttentionGate(skip_ch=c2, gate_ch=c3, inter_ch=c2 // 2)
        self.att2 = AttentionGate(skip_ch=c1, gate_ch=c2, inter_ch=c1 // 2)
        self.up1 = ResBlock(c3 + c2, c2, time_dim)
        self.up2 = ResBlock(c2 + c1, c1, time_dim)
        self.out_conv = nn.Conv2d(c1, in_channels, 3, padding=1)

    def forward(self, x_t, t):
        if t.dim() == 1:
            t = t.float().view(-1, 1)
        t_emb = self.time_mlp(self.time_emb(t))
        h0 = self.in_conv(x_t)
        h1 = self.down1(h0, t_emb)
        h2 = self.down2(self.pool(h1), t_emb)
        m = self.mid2(self.mid1(self.pool(h2), t_emb), t_emb)
        u1 = F.interpolate(m, scale_factor=2, mode='nearest')
        h2_g = self.att1(h2, u1)
        u1 = self.up1(torch.cat([u1, h2_g], dim=1), t_emb)
        u2 = F.interpolate(u1, scale_factor=2, mode='nearest')
        h1_g = self.att2(h1, u2)
        u2 = self.up2(torch.cat([u2, h1_g], dim=1), t_emb)
        return self.out_conv(u2)


def make_A(idx, total_pixels, n):
    gen = torch.Generator(device='cpu').manual_seed(int(idx))
    A = torch.randn(total_pixels, n, generator=gen, device='cpu')
    return A / torch.norm(A, dim=0, keepdim=True).clamp_min(1e-12)


class ImageFlow(pl.LightningModule):
    def __init__(self, data_root=DATA_ROOT, n_measurements=100, noise_std=1e-3,
                 flow_lr=1e-4, latent_lr=0.01, flow_weight=1.0, render_weight=1.0,
                 guided_N=20, guided_eta=1.0, num_images=10_000,
                 batch_size=64, fm_steps=20, flow_train_epochs=5, warmup_epochs=0,
                 flow_refine_every=1, use_persistent_latents=True,
                 use_gaussian_smoothing=False, model_type='flow',
                 guided_batch_size=5000, log_every_n_epochs=50):
        super().__init__()
        self.save_hyperparameters()
        self.use_persistent_latents = use_persistent_latents
        self.use_gaussian_smoothing = use_gaussian_smoothing
        self.model_type = model_type
        self.flow_lr = flow_lr
        self.latent_lr = latent_lr
        self.flow_weight = flow_weight
        self.render_weight = render_weight
        self.guided_N = guided_N
        self.guided_eta = guided_eta
        self.batch_size = batch_size
        self.fm_steps = fm_steps
        self.guided_batch_size = guided_batch_size
        self.log_every_n_epochs = log_every_n_epochs
        self._flow_train_epochs = flow_train_epochs
        self._warmup_epochs = warmup_epochs
        self._flow_refine_every = max(1, int(flow_refine_every))
        self._flow_model_ready = False
        self._guided = None

        all_images = load_mnist(data_root, num_images)
        self.num_images, self.channels, self.img_size = all_images.shape[:3]

        # one fixed random projection per image, y_i = A_i^T x_i + noise
        D = self.channels * self.img_size * self.img_size
        A = torch.stack([make_A(i, D, n_measurements) for i in range(self.num_images)])
        observations = torch.bmm(all_images.reshape(self.num_images, 1, D), A).squeeze(1)
        if noise_std > 0:
            observations = observations + noise_std * torch.randn_like(observations)
        self.register_buffer('A', A, persistent=False)
        self.register_buffer('gt_observations', observations)
        self.register_buffer('gt_images', all_images)

        latents = torch.randn(self.num_images, self.channels, self.img_size, self.img_size)
        if self.use_persistent_latents:
            self.latent_images = nn.Parameter(latents)
        else:
            self.register_buffer('latent_images', latents)

        if self.model_type == 'diffusion':
            self.diffusion = create_diffusion(timestep_respacing="")
        self.flow_model = FullImageFlowModel(in_channels=self.channels)

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(torch.arange(self.num_images))
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def configure_optimizers(self):
        params = []
        if self.use_persistent_latents:
            params.append({'params': [self.latent_images], 'lr': self.latent_lr})
        # the prior is only ever updated by _train_flow_model_on_latents (no grads reach it here)
        params.append({'params': self.flow_model.parameters(), 'lr': self.flow_lr})
        return torch.optim.Adam(params)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm='norm')

    def _flow_loss(self, x1):
        B = x1.shape[0]
        if self.model_type == 'flow':
            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=self.device)
            t_ = t.view(B, 1, 1, 1)
            z = t_ * x1 + (1.0 - t_) * x0

            x_pred = self.flow_model(z, t.unsqueeze(1))
            denom = (1.0 - t_).clamp(min=self.flow_model.t_eps)
            v = (x1 - z) / denom
            v_pred = (x_pred - z) / denom
            return ((v - v_pred) ** 2).mean()

        t = torch.randint(0, self.diffusion.num_timesteps, (B, 1), device=self.device)
        return self.diffusion.training_losses(self.flow_model, x1, t, model_kwargs={})["loss"].mean()

    @torch.enable_grad()
    def _train_flow_model_on_latents(self, epochs=None):
        for param in self.flow_model.parameters():
            param.requires_grad = True
        self.flow_model.train()

        x1_all = self.latent_images.detach()
        betas = (0.999, 0.999) if self.use_gaussian_smoothing else (0.9, 0.95)
        optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=self.flow_lr, betas=betas, weight_decay=1e-5)

        for _ in range(epochs or self._flow_train_epochs):
            batches = torch.randperm(self.num_images, device=self.device).split(64)
            total_loss = 0.0
            for idx in batches:
                optimizer.zero_grad()
                loss = self._flow_loss(x1_all[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        self.flow_model.eval()
        for param in self.flow_model.parameters():
            param.requires_grad = False
        self._flow_model_ready = True
        return total_loss / len(batches)

    def _guided_sampling_batch(self, gt_obs, A_batch):
        self.flow_model.eval()
        B = gt_obs.shape[0]
        x = torch.randn(B, self.channels, self.img_size, self.img_size, device=self.device)

        if self.model_type == 'flow':
            dt = 1.0 / self.guided_N
            for step in range(self.guided_N):
                t_val = step * dt
                t_c = torch.full((B, 1), t_val, device=self.device)

                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    x1_hat = self.flow_model(x, t_c)
                    denom = max(1.0 - t_val, self.flow_model.t_eps)
                    v_pred = (x1_hat - x) / denom
                    x1_flat = x1_hat.view(B, -1)
                    y_hat = torch.bmm(x1_flat.unsqueeze(1), A_batch).squeeze(1)
                    residual = y_hat - gt_obs
                    g_loss = (residual ** 2).sum()
                    grad = torch.autograd.grad(g_loss, x)[0]

                # per-sample DPS normalization: eta / ||residual||
                res_norm = residual.detach().view(B, -1).norm(dim=1).clamp_min(1e-8)
                scale = (self.guided_eta / res_norm).view(B, 1, 1, 1)
                x = x.detach() + v_pred.detach() * dt - scale * grad

        elif self.model_type == 'diffusion':
            def model_fn(x, t):
                return self.flow_model(x, t.float().view(-1, 1))

            for i in list(range(self.diffusion.num_timesteps))[::-1]:
                t = torch.full((B,), i, device=self.device, dtype=torch.long)

                with torch.enable_grad():
                    x = x.detach().requires_grad_(True)
                    out = self.diffusion.p_sample(model_fn, x, t, model_kwargs={})
                    x_next = out["sample"]
                    x0_flat = out["pred_xstart"].view(B, -1)
                    y_hat = torch.bmm(x0_flat.unsqueeze(1), A_batch).squeeze(1)
                    residual = y_hat - gt_obs
                    g_loss = (residual ** 2).sum()
                    grad = torch.autograd.grad(g_loss, x)[0]

                res_norm = residual.detach().view(B, -1).norm(dim=1).clamp_min(1e-8)
                scale = (self.guided_eta / res_norm).view(B, 1, 1, 1)
                x = x_next.detach() - scale * grad
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        self.flow_model.train()
        return x.detach()

    @torch.no_grad()
    def _guided_sample_all(self):
        return torch.cat([self._guided_sampling_batch(self.gt_observations[s:s + self.guided_batch_size],
                                                      self.A[s:s + self.guided_batch_size])
                          for s in range(0, self.num_images, self.guided_batch_size)])

    def on_train_epoch_start(self):
        # The prior is frozen between refinements and guided sampling does not depend on the
        # latents, so one guided sample per image is drawn up front for the whole epoch.
        t0 = time.time()
        since_warmup = self.current_epoch - self._warmup_epochs
        every = self._flow_refine_every if self.use_persistent_latents else 1
        flow_loss = float('nan')
        if since_warmup >= 0 and since_warmup % every == 0:
            flow_loss = self._train_flow_model_on_latents()
        t1 = time.time()

        self._guided = None
        if not self.use_persistent_latents:
            with torch.no_grad():
                self.latent_images.copy_(self._guided_sample_all())
        elif self.flow_weight > 0 and self._flow_model_ready and since_warmup >= 0:
            self._guided = self._guided_sample_all()
        print(f"Epoch {self.current_epoch}: {self.model_type} loss={flow_loss:.4f} "
              f"refine={t1 - t0:.1f}s guided={time.time() - t1:.1f}s", flush=True)

    def training_step(self, batch, batch_idx):
        (indices,) = batch
        B = indices.shape[0]
        gt_obs = self.gt_observations[indices]
        A_batch = self.A[indices]
        latent = self.latent_images[indices]

        y_hat = torch.bmm(latent.view(B, 1, -1), A_batch).squeeze(1)
        render_loss = F.mse_loss(y_hat, gt_obs)
        self.log('train_render_loss', render_loss, on_step=False, on_epoch=True)
        if not self.use_persistent_latents:
            return None  # latents were replaced by the guided samples, nothing to optimize

        guidance_loss = torch.tensor(0.0, device=self.device)
        combined_loss = self.render_weight * render_loss
        if self._guided is not None:
            guidance_loss = F.l1_loss(self._guided[indices], latent)
            combined_loss = self.flow_weight * guidance_loss + combined_loss

        self.log('train_guidance_loss', guidance_loss, on_step=False, on_epoch=True)
        self.log('train_combined_loss', combined_loss, on_step=False, on_epoch=True)
        return combined_loss

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % self.log_every_n_epochs == 0:
            self._log_latent_samples()

    def on_train_end(self):
        # extra refinement of the prior on the final latents
        t0 = time.time()
        flow_loss = self._train_flow_model_on_latents(epochs=100)
        print(f"Final refinement: {self.model_type} loss={flow_loss:.4f} ({time.time() - t0:.0f}s)", flush=True)

    @torch.no_grad()
    def generate(self, n_samples):
        self.flow_model.eval()
        B = int(n_samples)

        if self.model_type == 'flow':
            z = torch.randn(B, self.channels, self.img_size, self.img_size, device=self.device)
            dt = 1.0 / self.fm_steps
            for s in range(self.fm_steps):
                t_val = s * dt
                t_c = torch.full((B, 1), t_val, device=self.device)
                x_pred = self.flow_model(z, t_c)
                denom = max(1.0 - t_val, self.flow_model.t_eps)
                v_pred = (x_pred - z) / denom
                z = z + v_pred * dt
        else:
            z = self.diffusion.p_sample_loop(
                self.flow_model,
                (B, self.channels, self.img_size, self.img_size),
                clip_denoised=True,
                model_kwargs={}
            )

        self.flow_model.train()
        return z

    def _psnr(self, pred, target, max_val=1.0):
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return torch.tensor(float('inf'))
        return 10 * torch.log10(max_val ** 2 / mse)

    def _ssim(self, pred, target, window_size=11):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        pad = window_size // 2
        mu_p = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
        mu_t = F.avg_pool2d(target, window_size, stride=1, padding=pad)
        mu_pp = mu_p * mu_p
        mu_tt = mu_t * mu_t
        mu_pt = mu_p * mu_t
        sigma_pp = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pp
        sigma_tt = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_tt
        sigma_pt = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pt
        ssim_map = ((2 * mu_pt + C1) * (2 * sigma_pt + C2)) / \
                   ((mu_pp + mu_tt + C1) * (sigma_pp + sigma_tt + C2))
        return ssim_map.mean()

    @torch.no_grad()
    def _log_latent_samples(self):
        latents_01 = self.latent_images.detach().mul(0.5).add(0.5).clamp(0, 1)
        gt_01 = self.gt_images.mul(0.5).add(0.5).clamp(0, 1)

        mse_per = ((latents_01 - gt_01) ** 2).flatten(1).mean(dim=1).clamp_min(1e-12)
        psnr_mean = (10.0 * torch.log10(1.0 / mse_per)).mean().item()
        ssim_mean = self._ssim(latents_01, gt_01).item()

        self.log('latent_psnr', psnr_mean)
        self.log('latent_ssim', ssim_mean)
        print(f"Epoch {self.current_epoch}: latents PSNR={psnr_mean:.2f} dB SSIM={ssim_mean:.4f}", flush=True)

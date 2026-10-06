import csv
import os
import time

import numpy as np
import torch

from data import forward_op, project_to_2d
from diem import posterior_ddpm_sample
from diffusion import VPSchedule, diffusion_loss
from flow import flow_loss, guided_flow_sample


def render_loss(hat_x, A, y, forward_kind="linear"):
    # mean_i || F(A_i hat_x_i) - y_i ||_1
    resid = forward_op(torch.einsum("nmd,nd->nm", A, hat_x), forward_kind) - y
    return resid.abs().sum(dim=-1).mean()


def gt_loss(hat_x, x_gt):
    return (hat_x - x_gt).pow(2).sum(dim=-1).sqrt().mean()


def sliced_wasserstein_2(X, Y, n_projections=128, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    dirs = torch.randn(n_projections, X.shape[1], generator=g)
    dirs = dirs / dirs.norm(dim=1, keepdim=True)
    dirs = dirs.to(X.device).to(X.dtype)
    px = (X @ dirs.T).sort(dim=0).values
    py = (Y @ dirs.T).sort(dim=0).values
    return float(((px - py) ** 2).mean().sqrt().item())


def run_warmup(A, y, steps, lr, init_scale, seed, forward_kind):
    # rendering-only optimization of the latents (no prior)
    n, _, D = A.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    hat_x = (init_scale * torch.randn(n, D, generator=g)).to(A.device).requires_grad_(True)
    opt = torch.optim.Adam([hat_x], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        render_loss(hat_x, A, y, forward_kind=forward_kind).backward()
        opt.step()
    return hat_x.detach()


def run_em(net, A, y, x_gt, P, cfg, out_dir, device):
    A, y, x_gt, P, net = A.to(device), y.to(device), x_gt.to(device), P.to(device), net.to(device)
    diem = cfg.method == "diem"
    schedule = VPSchedule(cfg.T, cfg.beta_min, cfg.beta_max).to(device) if diem else None

    print(f"[em] warmup: {cfg.S_warmup} rendering-only steps")
    hat_x = run_warmup(A, y, cfg.S_warmup, cfg.warmup_lr, cfg.warmup_init_scale, cfg.seed, cfg.forward_kind)
    n = hat_x.shape[0]

    # one optimizer across the bootstrap and all M-steps
    opt = torch.optim.Adam(net.parameters(), lr=cfg.M_step_lr)

    def train_prior(steps):
        for _ in range(steps):
            idx = torch.randint(0, n, (cfg.M_step_batch,), device=device)
            xb = hat_x[idx].detach()
            loss = diffusion_loss(net, xb, schedule) if diem else flow_loss(net, xb, t_eps=cfg.flow_t_eps)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.M_step_grad_clip)
            opt.step()
        return loss.item()

    print(f"[em] prior bootstrap: {cfg.prior_bootstrap_steps} steps on the warmup latents")
    train_prior(cfg.prior_bootstrap_steps)

    perm = torch.randperm(n)[:min(cfg.n_eval_samples, n)]
    gt_2d = project_to_2d(x_gt[perm], P).detach().cpu()

    history, scatters = [], {}
    for outer in range(1, cfg.N_outer + 1):
        t0 = time.time()
        prev_hat_x = hat_x.clone()

        # E-step: one guided / posterior sample per observation
        if diem:
            sampled = posterior_ddpm_sample(net, A, y, schedule, cfg.sigma, N=cfg.sampler_steps,
                                            forward_kind=cfg.forward_kind)
        else:
            sampled = guided_flow_sample(net, A, y, cfg.eta, cfg.sampler_steps, forward_kind=cfg.forward_kind)
        # persistent latents: EMA of the samples; alpha = 0 replaces them
        hat_x = sampled.detach() if cfg.alpha == 0 else cfg.alpha * hat_x + (1.0 - cfg.alpha) * sampled

        m_loss = train_prior(cfg.M_step_inner_steps)

        with torch.no_grad():
            r = render_loss(hat_x, A, y, forward_kind=cfg.forward_kind).item()
            g = gt_loss(hat_x, x_gt).item()
            drift = (hat_x - prev_hat_x).pow(2).sum(-1).sqrt().mean().item()
            samples_2d = project_to_2d(hat_x[perm], P).detach().cpu()
            swd = sliced_wasserstein_2(samples_2d, gt_2d)

        history.append((outer, r, g, swd, drift, m_loss))
        if outer % cfg.log_scatter_every == 0 or outer == cfg.N_outer:
            scatters[f"outer{outer:03d}"] = samples_2d.numpy()
        print(f"[em outer {outer}/{cfg.N_outer}] render={r:.4f} gt={g:.4f} swd={swd:.4f} "
              f"drift={drift:.4f} M={m_loss:.4f} ({time.time() - t0:.1f}s)", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "em_history.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["outer", "render_loss", "gt_loss", "swd_2d", "latent_drift", "m_loss"])
        w.writerows(history)
    np.savez(os.path.join(out_dir, "scatter.npz"), gt=gt_2d.numpy(), **scatters)
    return history

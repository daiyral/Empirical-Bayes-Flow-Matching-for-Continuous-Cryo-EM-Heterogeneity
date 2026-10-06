import torch


class VPSchedule:
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.02):
        self.T = T
        self.betas = torch.linspace(beta_min, beta_max, T)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.abar = self.abar.to(device)
        return self


def diffusion_loss(net, x1, schedule, eps_floor=1e-5):
    # SNR-weighted x-loss (VP-DDPM v-loss with an x-prediction net): ||x_hat - x_1||^2 / (1 - abar_t)
    B = x1.shape[0]
    device = x1.device
    schedule.to(device)
    T = schedule.T

    t_idx = torch.randint(0, T, (B,), device=device)
    abar_t = schedule.abar[t_idx]
    one_minus = (1.0 - abar_t).clamp(min=eps_floor)
    sqrt_abar = abar_t.sqrt()
    sqrt_one_minus = one_minus.sqrt()

    eps = torch.randn_like(x1)
    abar_b = sqrt_abar.view(B, *([1] * (x1.dim() - 1)))
    one_minus_sqrt_b = sqrt_one_minus.view(B, *([1] * (x1.dim() - 1)))
    xt = abar_b * x1 + one_minus_sqrt_b * eps

    t_norm = (t_idx.float() + 1.0) / T
    x_hat_1 = net(xt, t_norm)

    sq = (x_hat_1 - x1).pow(2).sum(dim=-1)
    weight = 1.0 / one_minus
    return (sq * weight).mean()

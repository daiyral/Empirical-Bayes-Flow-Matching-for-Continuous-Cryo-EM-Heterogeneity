import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128, max_period=1000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class XPredMLP(nn.Module):
    # net(x_t, t) -> x_hat_1; shared by the flow and diffusion priors
    def __init__(self, D, hidden=256, depth=5, t_embed_dim=128):
        super().__init__()
        self.D = D
        self.t_embed = SinusoidalTimeEmbedding(t_embed_dim)
        self.t_proj = nn.Linear(t_embed_dim, hidden)

        layers = [nn.Linear(D + hidden, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, D)

    def forward(self, x_t, t):
        h_t = self.t_proj(self.t_embed(t))
        h = torch.cat([x_t, h_t], dim=-1)
        return self.head(self.trunk(h))

import argparse
import time

import numpy as np
import torch

from config import FORWARDS, METHODS, M_VALUES, cell_config, run_dir, run_name
from data import make_A, make_observations, make_two_moons
from em import run_em
from model import XPredMLP


def train_cell(method, forward, M):
    cfg = cell_config(method, forward, M)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = run_dir(method, forward, M)
    print(f"[run] {run_name(method, forward, M)} steps={cfg.sampler_steps} eta={cfg.eta} "
          f"alpha={cfg.alpha} N_outer={cfg.N_outer} seed={cfg.seed} device={device}")

    P, x = make_two_moons(n=cfg.n, noise=cfg.noise, D=cfg.D, seed=cfg.seed)
    A = make_A(n=cfg.n, M=cfg.M, D=cfg.D, seed=cfg.seed)
    y = make_observations(x, A, sigma=cfg.sigma, seed=cfg.seed, forward_kind=cfg.forward_kind)
    net = XPredMLP(D=cfg.D, hidden=cfg.hidden, depth=cfg.depth, t_embed_dim=cfg.t_embed_dim)

    t0 = time.time()
    history = run_em(net=net, A=A, y=y, x_gt=x, P=P, cfg=cfg, out_dir=out, device=device)
    swd = [h[3] for h in history]
    print(f"[done] {run_name(method, forward, M)} final_swd={swd[-1]:.4f} min_swd={min(swd):.4f} "
          f"(outer {history[swd.index(min(swd))][0]}) wall={time.time() - t0:.1f}s -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default=None, choices=[m for m, _ in METHODS],
                   help="default: all three")
    p.add_argument("--forward", default=None, choices=FORWARDS, help="default: both")
    p.add_argument("--M", type=int, default=None, choices=M_VALUES, help="default: all six")
    args = p.parse_args()

    for method, _ in METHODS:
        for forward in FORWARDS:
            for M in M_VALUES:
                if ((args.method in (None, method)) and (args.forward in (None, forward))
                        and (args.M in (None, M))):
                    train_cell(method, forward, M)

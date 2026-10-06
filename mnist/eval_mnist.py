import argparse
import json
import os

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from cryoflow import paths
from model_mnist import ImageFlow
from utils import DATA_ROOT, metrics_path


def psnr_matrix(a, b, max_val=2.0, chunk_size=64):
    # a: (M, C, H, W), b: (N, C, H, W). Images in [-1, 1] -> max_val=2.0
    # returns (M, N)
    M, N = a.shape[0], b.shape[0]
    a_flat = a.reshape(M, -1)
    b_flat = b.reshape(N, -1)
    out = torch.empty(M, N, device=a.device)
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        diff = a_flat[s:e, None, :] - b_flat[None, :, :]
        mse = (diff ** 2).mean(dim=-1).clamp(min=1e-12)
        out[s:e] = 10 * torch.log10(max_val ** 2 / mse)
    return out


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim_matrix(a, b, max_val=2.0, window_size=11, sigma=1.5, chunk_size=16):
    # a: (M, C, H, W), b: (N, C, H, W). Images in [-1, 1] -> max_val=2.0
    # returns (M, N) with mean SSIM per pair
    M, C, H, W = a.shape
    N = b.shape[0]
    device = a.device
    dtype = a.dtype

    window = _gaussian_window(window_size, sigma, device, dtype)
    window = window.expand(C, 1, window_size, window_size).contiguous()
    pad = window_size // 2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    def stats(x):
        mu = F.conv2d(x, window, padding=pad, groups=C)
        mu_sq = mu * mu
        sigma_sq = F.conv2d(x * x, window, padding=pad, groups=C) - mu_sq
        return mu, mu_sq, sigma_sq

    mu_a, mu_a_sq, sigma_a_sq = stats(a)
    mu_b, mu_b_sq, sigma_b_sq = stats(b)

    out = torch.empty(M, N, device=device, dtype=dtype)
    for s in range(0, M, chunk_size):
        e = min(s + chunk_size, M)
        m = e - s
        ab = (a[s:e].unsqueeze(1) * b.unsqueeze(0)).reshape(m * N, C, H, W)
        mu_ab = F.conv2d(ab, window, padding=pad, groups=C).view(m, N, C, H, W)

        mu_a_c = mu_a[s:e].unsqueeze(1)
        mu_a_sq_c = mu_a_sq[s:e].unsqueeze(1)
        sigma_a_sq_c = sigma_a_sq[s:e].unsqueeze(1)
        mu_b_b = mu_b.unsqueeze(0)
        mu_b_sq_b = mu_b_sq.unsqueeze(0)
        sigma_b_sq_b = sigma_b_sq.unsqueeze(0)

        mu_a_mu_b = mu_a_c * mu_b_b
        sigma_ab = mu_ab - mu_a_mu_b

        num = (2 * mu_a_mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a_sq_c + mu_b_sq_b + C1) * (sigma_a_sq_c + sigma_b_sq_b + C2)
        out[s:e] = (num / den).mean(dim=(2, 3, 4))

    return out


# matrix: (GT, generated) similarities
def precision(matrix):  # fidelity: each generated -> nearest GT
    return float(matrix.max(dim=0).values.mean())


def recall(matrix):  # coverage: each GT -> nearest generated
    return float(matrix.max(dim=1).values.mean())


def emd(matrix):  # Hungarian assignment
    cost = -matrix.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(-cost[row_ind, col_ind].mean())


@torch.no_grad()
def generate_samples(model, num_samples, batch_size=1000):
    samples = []
    for s in range(0, num_samples, batch_size):
        b = min(batch_size, num_samples - s)
        samples.append(model.generate(n_samples=b))
    return torch.cat(samples, dim=0)


@torch.no_grad()
def evaluate(gt, gen):
    metrics = {}
    for name, matrix_fn in (("psnr", psnr_matrix), ("ssim", ssim_matrix)):
        matrix = matrix_fn(gt, gen)
        metrics[f"emd_{name}"] = emd(matrix)
        metrics[f"precision_{name}"] = precision(matrix)
        metrics[f"recall_{name}"] = recall(matrix)
    return metrics


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = args.checkpoint_path or paths.checkpoint(args.run_name)
    model = ImageFlow.load_from_checkpoint(ckpt, map_location=device, data_root=args.data_root)
    model = model.to(device).eval()

    if args.fm_steps > 0:
        model.fm_steps = args.fm_steps

    n_samples = args.num_samples or model.num_images
    print(f"Generating {n_samples} unguided samples...")
    gen = generate_samples(model, n_samples)
    metrics = evaluate(model.gt_images, gen)
    metrics.update(num_samples=n_samples, n_measurements=model.hparams.n_measurements,
                   model_type=model.model_type, PL=model.use_persistent_latents, checkpoint=ckpt)
    print(json.dumps(metrics, indent=2))

    os.makedirs(paths.results(args.run_name), exist_ok=True)
    with open(metrics_path(args.run_name), "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote", metrics_path(args.run_name))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', type=str, required=True)
    p.add_argument('--checkpoint_path', type=str, default=None,
                   help='default: runs/checkpoints/<run_name>/last.ckpt')
    p.add_argument('--data_root', type=str, default=DATA_ROOT)
    p.add_argument('--num_samples', type=int, default=0,
                   help='0 = use model.num_images (one sample per GT image)')
    p.add_argument('--fm_steps', type=int, default=0,
                   help='0 = use the value stored in the checkpoint')
    args = p.parse_args()
    main(args)

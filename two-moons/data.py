import numpy as np
import torch
from sklearn.datasets import make_moons


def make_two_moons(n=10_000, noise=0.05, D=128, seed=0):
    raw, _ = make_moons(n_samples=n, noise=noise, random_state=seed)
    raw = raw.astype(np.float64)
    raw = (raw - raw.mean(axis=0, keepdims=True)) / raw.std(axis=0, keepdims=True)

    # fixed column-orthonormal lift P (D x 2): QR of a Gaussian
    rng = np.random.default_rng(seed)
    P, _ = np.linalg.qr(rng.standard_normal(size=(D, 2)))
    x = raw @ P.T
    return torch.from_numpy(P.astype(np.float32)), torch.from_numpy(x.astype(np.float32))


def project_to_2d(x, P):
    return x @ P


def make_A(n, M, D, seed=0):
    # per-sample A_i (M x D), entries iid N(0, 1/D)
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n, M, D, generator=g) / (D ** 0.5)
    return A.to(torch.float32)


def forward_op(Ax, kind):
    if kind == "linear":
        return Ax
    if kind == "sin":
        return torch.sin(Ax)
    raise ValueError(f"unknown forward_kind: {kind!r}")


def make_observations(x, A, sigma, seed=0, forward_kind="linear"):
    Ax = torch.einsum("nmd,nd->nm", A, x)
    g = torch.Generator(device="cpu").manual_seed(seed)
    eps = torch.randn(Ax.shape, generator=g) * float(sigma)
    return (forward_op(Ax, forward_kind) + eps.to(Ax.device)).to(torch.float32)

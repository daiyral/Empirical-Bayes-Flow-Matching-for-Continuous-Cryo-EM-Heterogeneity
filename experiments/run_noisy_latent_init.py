import argparse
import json
import os

import MDAnalysis as mda
import numpy as np
import torch


def _rigid_basis(ref):
    n = ref.shape[0]
    c = ref - ref.mean(0)
    cols = []
    for k in range(3):
        t = torch.zeros(n, 3, device=ref.device)
        t[:, k] = 1.0
        cols.append(t.reshape(-1))
    for k in range(3):
        ax = torch.zeros(3, device=ref.device)
        ax[k] = 1.0
        cols.append(torch.cross(ax.expand(n, 3), c, dim=1).reshape(-1))
    q, _ = torch.linalg.qr(torch.stack(cols, 1))
    return q


def _shake(x, ref, pairs, iters):
    i, j = pairs[:, 0], pairs[:, 1]
    d0 = (ref[i] - ref[j]).norm(dim=-1).view(1, -1, 1)
    deg = torch.zeros(x.shape[1], device=x.device).index_add_(
        0, torch.cat([i, j]), torch.ones(2 * len(i), device=x.device)).clamp_min(1).view(1, -1, 1)
    for _ in range(iters):
        v = x[:, i] - x[:, j]
        d = v.norm(dim=-1, keepdim=True)
        corr = 0.5 * v * ((d0 - d) / d)
        upd = torch.zeros_like(x)
        upd.index_add_(1, i, corr)
        upd.index_add_(1, j, -corr)
        x = x + upd / deg
    return x


def make_noisy_init(ref, n_frames, amp, ell, seed, iters=200, chunk=2000, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    ref = ref.to(device)
    n = ref.shape[0]

    d = torch.cdist(ref, ref)
    k = torch.exp(-d ** 2 / (2 * ell ** 2)) + 1e-6 * torch.eye(n, device=device)
    ev, vec = torch.linalg.eigh(k.double())
    sqrt_k = (vec * ev.clamp_min(0).sqrt()).float()          # K^{1/2}, K = sqrt_k sqrt_k^T
    rigid = _rigid_basis(ref)
    idx = torch.arange(n, device=device)
    pairs = torch.cat([torch.stack([idx[:-1], idx[1:]], 1),
                       torch.stack([idx[:-2], idx[2:]], 1)])

    gen = torch.Generator(device=device).manual_seed(seed)
    out = []
    for start in range(0, n_frames, chunk):
        b = min(chunk, n_frames - start)
        w = torch.randn(b, n, 3, generator=gen, device=device)
        f = torch.einsum('ij,bjk->bik', sqrt_k, w).reshape(b, -1)
        f = (f - (f @ rigid) @ rigid.T).reshape(b, n, 3)
        rms = f.pow(2).sum(-1).mean(-1, keepdim=True).sqrt().unsqueeze(-1)
        out.append(_shake(ref + f * (amp / rms), ref, pairs, iters).cpu())
    return torch.cat(out)


def validate(coords, ref, universe, backbone, sample=200):
    def dev(x, off):
        a = (x[:, :-off] - x[:, off:]).norm(dim=-1)
        b = (ref[:-off] - ref[off:]).norm(dim=-1)
        return (a - b).abs()

    sub = coords[torch.randperm(len(coords))[:sample]].float()
    md = []
    for i, _ in enumerate(universe.trajectory):
        if i % 50 == 0:
            md.append(torch.tensor(backbone.positions.copy()))
    md = torch.stack(md)
    universe.trajectory[0]

    rep = {}
    for tag, x in (("noisy", sub), ("md", md)):
        rep[tag] = {f"d{o}_mean": float(dev(x, o).mean()) for o in (1, 2)}
        rep[tag].update({f"d{o}_max": float(dev(x, o).max()) for o in (1, 2)})
    rep["rmsd_to_frame0"] = float((sub - ref).pow(2).sum(-1).mean(-1).sqrt().mean())
    rep["md_rmsd_to_frame0"] = float((md - ref).pow(2).sum(-1).mean(-1).sqrt().mean())
    rep["pairwise_rmsd"] = float(_pairwise_rmsd(sub[:100]))
    return rep


def _pairwise_rmsd(x):
    from cryoflow import utils
    a = x.unsqueeze(1).expand(-1, len(x), -1, -1).reshape(-1, x.shape[1], 3)
    b = x.unsqueeze(0).expand(len(x), -1, -1, -1).reshape(-1, x.shape[1], 3)
    return torch.sqrt(((utils.rmsdalign(a, b) - a) ** 2).sum(-1).mean(-1)).mean()


def build(pdb_file, xtc_file, out_path, n_frames, amp, ell, seed, iters=200):
    u = mda.Universe(pdb_file, xtc_file)
    bb = u.select_atoms('backbone')
    u.trajectory[0]
    ref = torch.tensor(bb.positions.copy(), dtype=torch.float32)

    coords = make_noisy_init(ref, n_frames, amp, ell, seed, iters=iters)
    rep = validate(coords, ref, u, bb)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez_compressed(out_path, coords=coords.numpy().astype(np.float32))
    meta = dict(pdb_file=pdb_file, xtc_file=xtc_file, n_frames=n_frames, n_atoms=len(ref),
                amp=amp, ell=ell, seed=seed, shake_iters=iters, validation=rep)
    with open(os.path.splitext(out_path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)

    n_, m_ = rep["noisy"], rep["md"]
    print(f"{out_path}: {n_frames}x{len(ref)}x3  rmsd/frame0 {rep['rmsd_to_frame0']:.2f} A "
          f"(MD {rep['md_rmsd_to_frame0']:.2f}), pairwise rmsd {rep['pairwise_rmsd']:.2f} A")
    print(f"  i,i+1 dev mean/max {n_['d1_mean']:.4f}/{n_['d1_max']:.4f} "
          f"(MD {m_['d1_mean']:.4f}/{m_['d1_max']:.4f})")
    print(f"  i,i+2 dev mean/max {n_['d2_mean']:.4f}/{n_['d2_max']:.4f} "
          f"(MD {m_['d2_mean']:.4f}/{m_['d2_max']:.4f})")
    if n_["d1_max"] > m_["d1_max"] or n_["d2_max"] > m_["d2_max"]:
        raise SystemExit("noise distorts backbone geometry beyond the MD trajectory's own range")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_file", required=True)
    ap.add_argument("--xtc_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_frames", type=int, default=10001)
    ap.add_argument("--amp", type=float, default=1.0, help="RMSD in A of each latent from frame 0")
    ap.add_argument("--ell", type=float, default=12.0, help="RBF length scale in A")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shake_iters", type=int, default=200)
    a = ap.parse_args()
    build(a.pdb_file, a.xtc_file, a.out, a.n_frames, a.amp, a.ell, a.seed, a.shake_iters)

#!/usr/bin/env python3
# cryoDRGN side of the cryoDRGN-guided AlphaFold3 baseline. Needs an environment with
# cryoDRGN (https://github.com/ml-struct-bio/cryodrgn, `pip install cryodrgn`):
#
#   python tools/cryodrgn_maps.py prepare --sim_dir "$CRYO_DATASET_ROOT/1a62_A_sim_output" --out_dir cdrgn/1a62_A
#   cryodrgn train_vae cdrgn/1a62_A/particles.mrcs --poses cdrgn/1a62_A/pose.pkl --ctf cdrgn/1a62_A/ctf.pkl \
#       --zdim 8 -n 100 -b 16 --uninvert-data --no-window -o cdrgn/1a62_A/train
#   python tools/cryodrgn_maps.py decode --workdir cdrgn/1a62_A/train --epoch 100 --n 1250 --out_dir cdrgn/1a62_A/maps
#
# The maps then guide AlphaFold3 through tools/af3_cryodrgn_guided.py.
import argparse
import os
import pickle

import numpy as np


def prepare(args):
    # simulated dataset -> cryoDRGN particle stack, poses and CTF parameters
    from scipy.spatial.transform import Rotation
    from cryodrgn.mrcfile import write_mrc
    ds_dir, tem_dir = os.path.join(args.sim_dir, "dataset"), os.path.join(args.sim_dir, "tem_runs")
    names = sorted(f for f in os.listdir(ds_dir) if f.endswith(".npz"))
    My = np.diag([1.0, -1.0, 1.0])
    imgs, rots, ctfs = [], [], []
    for name in names:
        d = np.load(os.path.join(ds_dir, name))
        assert d["images"].shape[0] == 1, f"{name}: expected 1 image per frame"
        # TEM-simulator coordinate file: x, y, z (nm) and ZXZ Euler angles (deg)
        lines = [l.strip() for l in open(os.path.join(tem_dir, name[:-4] + "_coords.txt"))
                 if l.strip() and not l.startswith("#")]
        row = np.loadtxt(lines[1:]).reshape(-1)
        R = Rotation.from_euler("ZXZ", row[3:6], degrees=True).as_matrix()
        rots.append(My @ R.T @ My)
        # the simulator adds the particle's height to the defocus
        df = args.defocus + 10.0 * row[2]
        img = d["images"][0].astype(np.float32)
        ctfs.append([img.shape[-1], args.apix, df, df, 0.0, args.kv, args.cs, args.amp_contrast, 0.0])
        imgs.append(img)
    os.makedirs(args.out_dir, exist_ok=True)
    imgs = np.stack(imgs)
    write_mrc(os.path.join(args.out_dir, "particles.mrcs"), imgs, is_vol=False, Apix=args.apix)
    with open(os.path.join(args.out_dir, "pose.pkl"), "wb") as f:
        # particles were shifted onto the box centre at extraction, so translations are 0
        pickle.dump((np.stack(rots).astype(np.float32), np.zeros((len(rots), 2), dtype=np.float32)), f)
    with open(os.path.join(args.out_dir, "ctf.pkl"), "wb") as f:
        pickle.dump(np.array(ctfs, dtype=np.float32), f)
    print(f"wrote particles.mrcs {imgs.shape}, pose.pkl, ctf.pkl -> {args.out_dir}")


class VolumeDecoder:
    # batched equivalent of cryodrgn's FTPositionalDecoder.eval_volume
    def __init__(self, config_path, weights_path, device):
        import cryodrgn.config
        from cryodrgn.models import HetOnlyVAE
        cfg = cryodrgn.config.load(config_path)
        self.D = cfg["lattice_args"]["D"]
        self.zdim = cfg["model_args"]["zdim"]
        model, self.lattice = HetOnlyVAE.load(cfg, weights_path, device=device)
        self.model = model.decoder
        self.model.eval()
        self.norm = cfg["dataset_args"]["norm"]
        self.device = device

    def __call__(self, z):
        import torch
        from cryodrgn import fft
        with torch.no_grad():
            z = torch.as_tensor(np.asarray(z), dtype=torch.float32, device=self.device)
            B, D = z.shape[0], self.D
            coords = self.lattice.coords
            vol_f = torch.zeros((B, D, D, D), dtype=torch.float32, device=self.device)
            for i, dz in enumerate(np.linspace(-0.5, 0.5, D, endpoint=True, dtype=np.float32)):
                x = coords + torch.tensor([0.0, 0.0, dz], device=self.device)
                keep = x.pow(2).sum(dim=1) <= 0.25
                xk = x[keep]
                xb = torch.cat([xk[None].expand(B, -1, -1),
                                z[:, None, :].expand(B, xk.shape[0], self.zdim)], dim=-1)
                if dz == 0.0:
                    y = self.model.forward(xb)
                else:
                    out = self.model.decode(xb)
                    y = out[..., 0] - out[..., 1]
                sl = torch.zeros(B, D * D, device=self.device)
                sl[:, keep] = y
                vol_f[:, i] = sl.view(B, D, D)
            vol_f = vol_f * self.norm[1] + self.norm[0]
            return torch.stack([fft.ihtn_center(v[:-1, :-1, :-1]) for v in vol_f])


def decode(args):
    # one potential map per particle latent. Voxel (iz, iy, ix) sits at ((ix, iy, iz) - D/2) * apix
    # relative to the particle's bounding-box centre; the map is written with its origin at voxel 0.
    import torch
    from cryodrgn.mrcfile import write_mrc
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.workdir, f"z.{args.epoch}.pkl"), "rb") as f:
        z = pickle.load(f)
    ind = np.arange(len(z)) if args.n is None else np.linspace(0, len(z) - 1, args.n).astype(int)
    print(f"{len(z)} latents of dim {z.shape[1]}; decoding {len(ind)} maps -> {args.out_dir}")
    dec = VolumeDecoder(os.path.join(args.workdir, "config.yaml"),
                        os.path.join(args.workdir, f"weights.{args.epoch}.pkl"), device)
    os.makedirs(args.out_dir, exist_ok=True)
    for s in range(0, len(ind), args.batch):
        sel = ind[s:s + args.batch]
        vols = dec(z[sel])
        for k, i in enumerate(sel):
            write_mrc(os.path.join(args.out_dir, f"esp_{i:06d}.map"),
                      vols[k].cpu().numpy().astype(np.float32), Apix=args.apix, is_vol=True)
        print(f"  {min(s + args.batch, len(ind))}/{len(ind)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="simulated dataset -> particles.mrcs, pose.pkl, ctf.pkl")
    p.add_argument("--sim_dir", required=True, help="<protein>_sim_output from tools/simulate_cryo_em.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--apix", type=float, default=1.2)
    p.add_argument("--kv", type=float, default=300.0)
    p.add_argument("--cs", type=float, default=2.1)
    p.add_argument("--amp_contrast", type=float, default=0.06)
    p.add_argument("--defocus", type=float, default=5000.0, help="nominal defocus (A)")
    p.set_defaults(func=prepare)
    p = sub.add_parser("decode", help="trained cryoDRGN model -> one esp_*.map per particle")
    p.add_argument("--workdir", required=True, help="cryodrgn train_vae output directory")
    p.add_argument("--epoch", type=int, default=100)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n", type=int, default=None, help="particles to decode, evenly spaced (default: all)")
    p.add_argument("--apix", type=float, default=1.2)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(func=decode)
    args = ap.parse_args()
    args.func(args)

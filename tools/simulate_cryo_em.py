#!/usr/bin/env python3
import argparse
import os
import json
import shutil
import subprocess
import numpy as np
import MDAnalysis as mda
import mrcfile
import matplotlib.pyplot as plt
from pathlib import Path

from cryoflow import paths
from scipy.spatial.transform import Rotation
from scipy.ndimage import fourier_shift


# ── Parameters ──────────────────────────────────────────────────────────────
BOX_SIZE = 64
PIXEL_SIZE_NM = 0.12  # 1.2 Å
NUM_EXAMPLES = 11


# ── Euler angle conversion ──────────────────────────────────────────────────
_M_y = np.diag([1.0, -1.0, 1.0])

def zxz_to_zyz_degrees(phi_zxz, theta_zxz, psi_zxz):
    # ZXZ intrinsic (TEM-sim) → flip-Y + inverse → extrinsic zyz (cryoJAX)
    R = Rotation.from_euler('ZXZ', [phi_zxz, theta_zxz, psi_zxz], degrees=True).as_matrix()
    R_jax = _M_y @ R.T @ _M_y
    phi_zyz, theta_zyz, psi_zyz = Rotation.from_matrix(R_jax).as_euler('zyz', degrees=True)
    return phi_zyz, theta_zyz, psi_zyz


def extract_frames(pdb_file, xtc_file, frame_indices, frames_dir):
    print(f"Loading trajectory: {pdb_file} + {xtc_file}")
    u = mda.Universe(pdb_file, xtc_file)
    total = len(u.trajectory)
    print(f"  Total frames in trajectory: {total}")
    print(f"  Extracting {len(frame_indices)} frames ...")

    os.makedirs(frames_dir, exist_ok=True)
    atoms = u.select_atoms('backbone')

    frame_paths = []
    for i, idx in enumerate(frame_indices):
        u.trajectory[idx]
        out_path = os.path.join(frames_dir, f"frame_{i:05d}.pdb")
        atoms.write(out_path)
        frame_paths.append(out_path)

    print(f"  Saved {len(frame_paths)} PDB files to {frames_dir}/")
    return frame_paths


def generate_input(tem_input, frame_pdb, frame_idx, work_dir):
    with open(tem_input) as f:
        template = f.read()

    prefix = f"frame_{frame_idx:05d}"
    replacements = {
        "PDB_FILE_PLACEHOLDER": os.path.abspath(frame_pdb),
        "COORD_FILE_PLACEHOLDER": os.path.join(work_dir, f"{prefix}_coords.txt"),
        "MRC_FILE_PLACEHOLDER": os.path.join(work_dir, f"{prefix}.mrc"),
        "MAP_FILE_PLACEHOLDER": os.path.join(work_dir, f"{prefix}_map.mrc"),
        "LOG_FILE_PLACEHOLDER": os.path.join(work_dir, f"{prefix}.log"),
    }

    for key, val in replacements.items():
        template = template.replace(key, val)

    input_path = os.path.join(work_dir, f"{prefix}_input.txt")
    with open(input_path, "w") as f:
        f.write(template)

    return input_path, replacements["MRC_FILE_PLACEHOLDER"], replacements["COORD_FILE_PLACEHOLDER"]


def run_tem_simulator(tem_bin, input_file):
    result = subprocess.run(
        [tem_bin, input_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  WARNING: TEM-simulator failed for {input_file}")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    return True


def extract_particles(mrc_path, coord_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        data = mrc.data
        micrograph = data[0].astype(np.float32) if data.ndim == 3 else data.astype(np.float32)

    ny, nx = micrograph.shape
    half = BOX_SIZE // 2

    lines = [l.strip() for l in open(coord_path) if l.strip() and not l.startswith("#")]
    if len(lines) < 2:
        return [], np.empty((0, 3), dtype=np.float32)
    coords = np.loadtxt(lines[1:])
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)

    particles = []
    particle_angles = []
    for row in coords:
        x_nm, y_nm = row[0], row[1]

        # TEM-simulator coordinate file columns 3,4,5 are ZXZ Euler angles (degrees)
        phi_zxz = row[3] if row.shape[0] > 3 else 0.0
        theta_zxz = row[4] if row.shape[0] > 4 else 0.0
        psi_zxz = row[5] if row.shape[0] > 5 else 0.0

        # Continuous micrograph pixel coordinate of the particle center
        cx_f = x_nm / PIXEL_SIZE_NM + nx // 2
        cy_f = y_nm / PIXEL_SIZE_NM + ny // 2
        cx = int(round(cx_f))
        cy = int(round(cy_f))
        # Sub-pixel residue left over from integer rounding
        residue_x = cx_f - cx
        residue_y = cy_f - cy
        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half
        if 0 <= x0 and x1 <= nx and 0 <= y0 and y1 <= ny:
            patch = micrograph[y0:y1, x0:x1].astype(np.float32)
            # Fourier-shift by -residue so the particle lands exactly on the
            # patch's pixel-32 center, eliminating extraction rounding jitter.
            if residue_x != 0 or residue_y != 0:
                patch = np.fft.ifft2(
                    fourier_shift(np.fft.fft2(patch), shift=(-residue_y, -residue_x))
                ).real.astype(np.float32)
            # Zero-mean, unit-std normalization
            patch = patch - patch.mean()
            std = patch.std()
            if std > 0:
                patch = patch / std
            particles.append(patch)

            # Convert ZXZ -> ZYZ for cryoJAX
            phi_zyz, theta_zyz, psi_zyz = zxz_to_zyz_degrees(phi_zxz, theta_zxz, psi_zxz)
            particle_angles.append([phi_zyz, theta_zyz, psi_zyz])

    angles = np.array(particle_angles, dtype=np.float32) if particle_angles else np.empty((0, 3), dtype=np.float32)
    return particles, angles


def get_backbone_coords(frame_pdb):
    u = mda.Universe(frame_pdb)
    backbone = u.select_atoms('backbone')
    return backbone.positions.astype(np.float32)


def plot_particles(particles, output_path):
    n = min(NUM_EXAMPLES, len(particles))
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 2.5 * rows))
    for i, ax in enumerate(np.array(axes).flat):
        if i < n:
            ax.imshow(particles[i], cmap="gray", origin="lower")
            ax.set_title(f"Particle {i+1}", fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Picked Particles ({len(particles)} total, {n} shown)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cryo-EM simulation pipeline from MD trajectory")
    parser.add_argument('--pdb_file', type=str, default=paths.md_files("1a62_A")[0], help='Path to PDB file')
    parser.add_argument('--xtc_file', type=str, default=paths.md_files("1a62_A")[1], help='Path to XTC trajectory file')
    parser.add_argument("--tem_bin", type=str, default=paths.TEM_SIMULATOR, help="TEM-simulator executable (default: on PATH, or $TEM_SIMULATOR)")
    parser.add_argument("--tem_input", type=str, required=True, help="TEM-simulator input file; the per-frame paths are written as PDB_FILE_PLACEHOLDER, COORD_FILE_PLACEHOLDER, MRC_FILE_PLACEHOLDER")
    parser.add_argument("--num_frames", type=int, default=None, help="Number of evenly spaced frames")
    parser.add_argument("--frame_step", type=int, default=None, help="Use every Nth frame")
    parser.add_argument("--output_dir", type=str, default=os.path.join(paths.CRYO_DATASET, "1a62_A_sim_output"), help="Output directory")
    args = parser.parse_args()
    if shutil.which(args.tem_bin) is None:
        raise SystemExit(f"TEM-simulator not found: {args.tem_bin} (see README, Getting the data)")
    with open(args.tem_input) as f:
        tem_input = f.read()
    missing = [k for k in ("PDB_FILE_PLACEHOLDER", "COORD_FILE_PLACEHOLDER", "MRC_FILE_PLACEHOLDER") if k not in tem_input]
    if missing:
        raise SystemExit(f"{args.tem_input} lacks {missing} (see README, Getting the data)")

    # Determine frame indices
    u_tmp = mda.Universe(args.pdb_file, args.xtc_file)
    total_traj = len(u_tmp.trajectory)
    if args.frame_step is not None:
        traj_indices = list(range(0, total_traj, args.frame_step))
    elif args.num_frames is not None:
        traj_indices = np.linspace(0, total_traj - 1, args.num_frames, dtype=int).tolist()
    else:
        traj_indices = list(range(total_traj))
    print(f"Using {len(traj_indices)} frames from trajectory of {total_traj}")

    frames_dir = os.path.join(args.output_dir, "frames")
    work_dir = os.path.join(args.output_dir, "tem_runs")
    dataset_dir = os.path.join(args.output_dir, "dataset")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # ── Step 1: Extract PDB frames ──
    frame_paths = extract_frames(args.pdb_file, args.xtc_file, traj_indices, frames_dir)

    # ── Step 2 & 3: Generate inputs and run TEM-simulator ──
    MAX_RETRIES = 3
    all_particles = []
    frame_files = []
    n_atoms = None
    for i, frame_pdb in enumerate(frame_paths):
        print(f"\r  Simulating frame {i+1}/{len(frame_paths)}", end="", flush=True)

        input_file, mrc_path, coord_path = generate_input(args.tem_input, frame_pdb, i, work_dir)

        particles = []
        angles = np.empty((0, 3), dtype=np.float32)
        for attempt in range(MAX_RETRIES):
            # Clean stale outputs before retry
            for f in [mrc_path, coord_path]:
                if os.path.exists(f):
                    os.remove(f)

            if not run_tem_simulator(args.tem_bin, input_file):
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for frame {i}")
                continue

            if not os.path.exists(mrc_path) or not os.path.exists(coord_path):
                print(f"  Retry {attempt+1}/{MAX_RETRIES} for frame {i} (missing output)")
                continue

            particles, angles = extract_particles(mrc_path, coord_path)
            if len(particles) > 0:
                break
            print(f"  Retry {attempt+1}/{MAX_RETRIES} for frame {i} (0 particles)")

        if len(particles) == 0:
            print(f"\n  WARNING: frame {i} failed after {MAX_RETRIES} retries, skipping")
            continue

        all_particles.extend(particles)

        # Save per-frame structured data
        # NOTE: angles are now in ZYZ convention (degrees), ready for cryoJAX
        backbone_coords = get_backbone_coords(frame_pdb)
        if n_atoms is None:
            n_atoms = backbone_coords.shape[0]

        fname = f"frame_{i:05d}.npz"
        np.savez(
            os.path.join(dataset_dir, fname),
            images=np.array(particles, dtype=np.float32) if particles else np.empty((0, BOX_SIZE, BOX_SIZE), dtype=np.float32),
            angles=angles,  # ZYZ degrees, ready for cryoJAX EulerAnglePose
            coords_3d=backbone_coords,
            trajectory_frame_idx=np.array(traj_indices[i]),
        )
        frame_files.append(fname)

        # Clean up large MRC and map files to save disk space
        for f in [mrc_path, mrc_path.replace(".mrc", "_map.mrc")]:
            if os.path.exists(f):
                os.remove(f)

    print(f"\n\nTotal particles extracted: {len(all_particles)}")

    # ── Save dataset metadata ──
    metadata = {
        "n_frames": len(frame_files),
        "n_atoms": int(n_atoms) if n_atoms is not None else 0,
        "image_size": [BOX_SIZE, BOX_SIZE],
        "pixel_size_nm": PIXEL_SIZE_NM,
        "angle_convention": "ZYZ_intrinsic_degrees",
        "angle_note": "Converted from TEM-simulator ZXZ to cryoJAX ZYZ. "
                      "Use directly as (phi_angle, theta_angle, psi_angle) in EulerAnglePose.",
        "pdb_file": args.pdb_file,
        "xtc_file": args.xtc_file,
        "trajectory_frame_indices": traj_indices,
        "frame_files": frame_files,
    }
    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved dataset metadata to {dataset_dir}/metadata.json")

    if len(all_particles) == 0:
        print("ERROR: No particles extracted.")
        return

    # ── Step 5: Shuffle and save ──
    rng = np.random.default_rng(42)
    rng.shuffle(all_particles)

    stack = np.array(all_particles, dtype=np.float32)
    out_npy = os.path.join(args.output_dir, "particles.npy")
    np.save(out_npy, stack)
    print(f"Saved {stack.shape[0]} particles ({stack.shape}) to {out_npy}")

    # ── Step 6: Plot examples ──
    out_img = os.path.join(args.output_dir, "picked_particles.png")
    plot_particles(all_particles, out_img)


if __name__ == "__main__":
    main()
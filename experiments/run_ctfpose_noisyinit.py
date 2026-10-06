import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from config import (CKPT_ROOT, COMMON, EPOCHS, EVAL_NUM_SAMPLES, LOG_DIR,
                    OMP_THREADS_PER_JOB, PROTEINS, eval_cmd, train_cmd)
from cryoflow import paths
# One source of truth for the noisy-init npz naming and construction.
from run_noisy_init import ensure_init, init_path

# Metadata perturbation, identical to run_ctf_pose_errors.py.
ERROR_FRAC = 1.0
CTF_DEFOCUS_ERROR = 0.05  # max relative defocus miss -> ~U(2.5%, 5%) of 5000 A
POSE_ERROR_DEG = 5.0      # max pose tilt -> ~U(2.5, 5) deg
ERROR_SEED = 0

TAG = "ctfpose_small_noisyinit"
MAX_TRAIN_ATTEMPTS = 20


def build_jobs(amp, ell, seed, smoke=False):
    jobs = []
    for protein, pcfg in PROTEINS.items():
        cfg = dict(COMMON)
        cfg.update(pcfg)

        run_name = f"{protein}_{TAG}"
        epochs = EPOCHS
        eval_num_samples = EVAL_NUM_SAMPLES
        eval_N = cfg["eval_N"]
        if smoke:
            run_name = f"smoke_{run_name}"
            epochs = 3
            cfg["warmup_epochs"] = 1
            cfg["flow_warmup_epochs"] = 1
            cfg["log_every_n_epochs"] = 1
            eval_num_samples = 200
            eval_N = 5

        train_args = train_cmd(cfg, run_name, epochs, first_frame_initial=False, extra=[
            # metadata perturbation, applied to every particle
            "--pose_error_frac", str(ERROR_FRAC),
            "--ctf_error_frac", str(ERROR_FRAC),
            "--pose_error_deg", str(POSE_ERROR_DEG),
            "--ctf_defocus_error", str(CTF_DEFOCUS_ERROR),
            "--error_seed", str(ERROR_SEED),
            # frame-0 + per-latent noise; coords are already in the trajectory frame
            "--latent_init_npz", init_path(protein, amp, ell, seed),
            "--latent_init_recenter", "none",
        ])

        eval_args = eval_cmd(cfg, run_name, eval_num_samples, eval_N)

        jobs.append((run_name, train_args, eval_args))
    return jobs


def ckpt_path(run_name):
    return os.path.join(CKPT_ROOT, run_name, "last.ckpt")


def run_job(index, run_name, train_args, eval_args, gpus, wandb_mode):
    gpu_id = gpus[index % len(gpus)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = OMP_THREADS_PER_JOB
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode

    log_path = os.path.join(LOG_DIR, f"{run_name}.log")
    print(f"=== Launching {run_name} on GPU {gpu_id} (log: {log_path}) ===", flush=True)
    ckpt = ckpt_path(run_name)

    with open(log_path, "a") as log_file:
        # A crashed train stage restarts from last.ckpt (written every epoch); train.py
        # picks the wandb run id back up from the checkpoint dir, so the run continues
        # rather than forking. Attempts that leave the checkpoint untouched are real
        # errors, not crashes, so they stop the retry loop instead of spinning.
        for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
            args = list(train_args)
            if attempt > 1:
                args += ["--resume_from_checkpoint", ckpt]
            before = os.path.getmtime(ckpt) if os.path.exists(ckpt) else -1
            log_file.write(f"\n===== train (attempt {attempt}): {' '.join(args)} =====\n")
            log_file.flush()
            rc = subprocess.run(args, env=env, stdout=log_file,
                                stderr=subprocess.STDOUT).returncode
            print(f"=== {run_name} train finished (exit {rc}, attempt {attempt}) ===", flush=True)
            if rc == 0:
                break
            after = os.path.getmtime(ckpt) if os.path.exists(ckpt) else -1
            if after <= before:
                print(f"=== {run_name} made no progress; not resuming ===", flush=True)
                return run_name, "train", rc, log_path
            print(f"=== {run_name} resuming from {ckpt} ===", flush=True)
        else:
            return run_name, "train", rc, log_path

        log_file.write(f"\n===== eval: {' '.join(eval_args)} =====\n")
        log_file.flush()
        rc = subprocess.run(eval_args, env=env, stdout=log_file,
                            stderr=subprocess.STDOUT).returncode
        print(f"=== {run_name} eval finished (exit {rc}) ===", flush=True)
        if rc != 0:
            return run_name, "eval", rc, log_path
    return run_name, "done", 0, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--amp", type=float, default=1.0,
                        help="RMSD in A between each noisy latent and frame 0")
    parser.add_argument("--ell", type=float, default=12.0,
                        help="RBF length scale in A; larger = smoother, gentler on bonds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shake_iters", type=int, default=200)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids, one job each (default: 0..n_gpus-1)")
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs / 200 eval samples, offline wandb, smoke_ run names")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated run-name substrings to keep (e.g. 1a62_A)")
    parser.add_argument("--wandb_mode", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the commands and the GPU assignment, launch nothing")
    cli = parser.parse_args()

    jobs = build_jobs(cli.amp, cli.ell, cli.seed, smoke=cli.smoke)
    if cli.only:
        keys = [k.strip() for k in cli.only.split(",") if k.strip()]
        jobs = [j for j in jobs if any(k in j[0] for k in keys)]

    wanted = [p for p in PROTEINS if any(p in j[0] for j in jobs)]
    if not cli.dry_run:
        # Reuses the npz run_noisy_init.py already wrote, so both experiments start from
        # the identical perturbation and the only difference is the metadata error.
        for p in wanted:
            ensure_init(p, PROTEINS[p], cli.amp, cli.ell, cli.seed, cli.shake_iters)

    wandb_mode = cli.wandb_mode or ("offline" if cli.smoke else None)

    if cli.gpus:
        gpus = [int(g) for g in cli.gpus.split(",") if g.strip() != ""]
    else:
        gpus = list(range(max(1, torch.cuda.device_count())))
    if len(gpus) < len(jobs):
        print(f"Warning: {len(jobs)} jobs but {len(gpus)} GPUs given; "
              f"some jobs will share a GPU (assigned round-robin).")

    print(f"Running {len(jobs)} jobs: {[j[0] for j in jobs]}")
    if cli.dry_run:
        for i, (run_name, train_args, eval_args) in enumerate(jobs):
            print(f"\n--- {run_name}  GPU {gpus[i % len(gpus)]}")
            print("train:", " ".join(train_args))
            print("eval: ", " ".join(eval_args))
        sys.exit(0)

    os.makedirs(LOG_DIR, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(run_job, i, *job, gpus, wandb_mode)
                   for i, job in enumerate(jobs)]
        results = [f.result() for f in futures]

    failures = [r for r in results if r[2] != 0]
    if failures:
        print(f"\n{len(failures)}/{len(jobs)} jobs failed:")
        for run_name, stage, rc, log_path in failures:
            print(f"  {run_name}: {stage} exit {rc}, see {log_path}")
        sys.exit(1)
    print(f"\nAll {len(jobs)} jobs completed successfully.")

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from cryoflow import paths

from config import (CKPT_ROOT, COMMON, EPOCHS, EVAL_NUM_SAMPLES, LOG_DIR,
                    OMP_THREADS_PER_JOB, PROTEINS, eval_cmd, train_cmd)

# Applied to every particle (see module docstring).
ERROR_FRAC = 1.0
CTF_DEFOCUS_ERROR = 0.05  # max relative defocus miss -> ~U(2.5%, 5%) of 5000 A
POSE_ERROR_DEG = 5.0      # max pose tilt -> ~U(2.5, 5) deg
ERROR_SEED = 0
TAG = "small"

# A crashed stage is retried; train resumes from last.ckpt (wandb run id is
# picked up from <ckpt_dir>/wandb_run_id.txt, so the run continues, not restarts).
MAX_ATTEMPTS = 4


def build_jobs(smoke=False):
    jobs = []
    for protein, pcfg in PROTEINS.items():
        cfg = dict(COMMON)
        cfg.update(pcfg)

        run_name = f"{protein}_ctfpose_{TAG}"
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

        train_args = train_cmd(cfg, run_name, epochs, extra=[
            # metadata perturbation, applied to every particle
            "--pose_error_frac", str(ERROR_FRAC),
            "--ctf_error_frac", str(ERROR_FRAC),
            "--pose_error_deg", str(POSE_ERROR_DEG),
            "--ctf_defocus_error", str(CTF_DEFOCUS_ERROR),
            "--error_seed", str(ERROR_SEED),
        ])

        eval_args = eval_cmd(cfg, run_name, eval_num_samples, eval_N)

        jobs.append((run_name, train_args, eval_args))
    return jobs


def run_job(index, run_name, train_args, eval_args, n_gpus, wandb_mode):
    gpu_id = index % n_gpus if n_gpus > 0 else 0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = OMP_THREADS_PER_JOB
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode

    log_path = os.path.join(LOG_DIR, f"{run_name}.log")
    ckpt = os.path.join(CKPT_ROOT, run_name, "last.ckpt")
    print(f"=== Launching {run_name} on GPU {gpu_id} (log: {log_path}) ===", flush=True)

    # Append: a resumed job must not clobber the log of the attempt that died.
    with open(log_path, "a") as log_file:
        for stage, args in (("train", train_args), ("eval", eval_args)):
            for attempt in range(MAX_ATTEMPTS):
                if attempt:  # crashed -> pick up where it left off
                    if stage == "train":
                        if not os.path.exists(ckpt):
                            print(f"=== {run_name} train crashed, no {ckpt} to resume from ===",
                                  flush=True)
                            return run_name, stage, rc, log_path
                        args = args + ["--resume_from_checkpoint", ckpt]
                    print(f"=== {run_name} {stage} retry {attempt}/{MAX_ATTEMPTS - 1} ===",
                          flush=True)
                log_file.write(f"\n===== {stage} (attempt {attempt + 1}): {' '.join(args)} =====\n")
                log_file.flush()
                rc = subprocess.run(args, env=env, stdout=log_file,
                                    stderr=subprocess.STDOUT).returncode
                print(f"=== {run_name} {stage} finished (exit {rc}) ===", flush=True)
                if rc == 0:
                    break
            if rc != 0:
                return run_name, stage, rc, log_path
    return run_name, "done", 0, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs / 200 eval samples, offline wandb, smoke_ run names")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated run-name substrings to keep (e.g. 1a62_A)")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        help="Value for WANDB_MODE (smoke defaults to offline)")
    cli = parser.parse_args()

    jobs = build_jobs(smoke=cli.smoke)
    if cli.only:
        keys = [k.strip() for k in cli.only.split(",") if k.strip()]
        jobs = [j for j in jobs if any(k in j[0] for k in keys)]

    wandb_mode = cli.wandb_mode or ("offline" if cli.smoke else None)

    n_gpus = torch.cuda.device_count()
    if n_gpus < len(jobs):
        print(f"Warning: {len(jobs)} jobs but only {n_gpus} GPUs visible; "
              f"some jobs will share a GPU (assigned round-robin).")

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Running {len(jobs)} jobs: {[j[0] for j in jobs]}")

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(run_job, i, *job, n_gpus, wandb_mode)
                   for i, job in enumerate(jobs)]
        results = [f.result() for f in futures]

    failures = [r for r in results if r[2] != 0]
    if failures:
        print(f"\n{len(failures)}/{len(jobs)} jobs failed:")
        for run_name, stage, rc, log_path in failures:
            print(f"  {run_name}: {stage} exit {rc}, see {log_path}")
        sys.exit(1)
    print(f"\nAll {len(jobs)} jobs completed successfully.")

    if cli.smoke or cli.only:
        print("Skipping report table (partial run).")
    else:
        print("=== Building reports/table_ablations.{tex,md,png} ===", flush=True)
        table = os.path.join(paths.ROOT, "vis", "tables", "make_ablation_table.py")
        rc = subprocess.run([sys.executable, table]).returncode
        if rc != 0:
            print(f"Table generation failed (exit {rc})")
            sys.exit(1)

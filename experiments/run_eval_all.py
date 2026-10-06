import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from config import (COMMON, EVAL_NUM_SAMPLES, LOG_DIR, OMP_THREADS_PER_JOB,
                    PROTEINS, eval_cmd)
from cryoflow import paths

MAX_ATTEMPTS = 2


def build_jobs(suffix, num_samples, N, smoke=False):
    jobs = []
    for protein, pcfg in PROTEINS.items():
        cfg = dict(COMMON)
        cfg.update(pcfg)

        run_name = f"{protein}{suffix}"
        n_samples = num_samples if num_samples is not None else EVAL_NUM_SAMPLES
        eval_N = N if N is not None else cfg["eval_N"]
        if smoke:
            run_name = f"smoke_{run_name}"
            n_samples, eval_N = 200, 5

        jobs.append((run_name, eval_cmd(cfg, run_name, n_samples, eval_N)))
    return jobs


def free_gpus(n_needed):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"], text=True)
        used = [(int(l.split(",")[1]), int(l.split(",")[0]))
                for l in out.strip().splitlines()]
        used.sort()
        return [idx for _, idx in used][:n_needed]
    except Exception:
        return list(range(torch.cuda.device_count()))[:n_needed]


def run_job(index, run_name, args, gpu_ids, wandb_mode):
    gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else 0
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = OMP_THREADS_PER_JOB
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode

    log_path = os.path.join(LOG_DIR, f"{run_name}_eval.log")
    print(f"=== Launching {run_name} on GPU {gpu_id} (log: {log_path}) ===", flush=True)

    with open(log_path, "w") as log_file:
        for attempt in range(MAX_ATTEMPTS):
            if attempt:
                print(f"=== {run_name} eval retry {attempt}/{MAX_ATTEMPTS - 1} ===", flush=True)
            log_file.write(f"\n===== eval (attempt {attempt + 1}): {' '.join(args)} =====\n")
            log_file.flush()
            rc = subprocess.run(args, env=env, stdout=log_file,
                                stderr=subprocess.STDOUT).returncode
            print(f"=== {run_name} eval finished (exit {rc}) ===", flush=True)
            if rc == 0:
                return run_name, 0, log_path
    return run_name, rc, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="",
                        help="Appended to the protein name to form the run name "
                             "(e.g. _ctfpose_small). Default: the protein name itself.")
    parser.add_argument("--only", default=None,
                        help="Comma-separated run-name substrings to keep (e.g. 1a62_A)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help=f"Samples to draw (default {EVAL_NUM_SAMPLES})")
    parser.add_argument("--N", type=int, default=None,
                        help="Sampling steps (default: per-protein eval_N from config.py)")
    parser.add_argument("--smoke", action="store_true",
                        help="200 samples / N=5, offline wandb, smoke_ run names")
    parser.add_argument("--wandb_mode", default=None, help="Value for WANDB_MODE")
    cli = parser.parse_args()

    jobs = build_jobs(cli.suffix, cli.num_samples, cli.N, smoke=cli.smoke)
    if cli.only:
        keys = [k.strip() for k in cli.only.split(",") if k.strip()]
        jobs = [j for j in jobs if any(k in j[0] for k in keys)]

    missing = [rn for rn, _ in jobs if not os.path.exists(paths.checkpoint(rn))]
    if missing:
        print("No checkpoint for: " + ", ".join(f"{m} ({paths.checkpoint(m)})" for m in missing))
        sys.exit(1)

    wandb_mode = cli.wandb_mode or ("offline" if cli.smoke else None)

    os.makedirs(LOG_DIR, exist_ok=True)
    gpu_ids = free_gpus(len(jobs))
    print(f"Running {len(jobs)} evals on GPUs {gpu_ids}: {[j[0] for j in jobs]}")

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(run_job, i, *job, gpu_ids, wandb_mode)
                   for i, job in enumerate(jobs)]
        results = [f.result() for f in futures]

    failures = [r for r in results if r[1] != 0]
    if failures:
        print(f"\n{len(failures)}/{len(jobs)} evals failed:")
        for run_name, rc, log_path in failures:
            print(f"  {run_name}: exit {rc}, see {log_path}")
        sys.exit(1)
    print(f"\nAll {len(jobs)} evals completed successfully.")

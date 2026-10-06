import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch

from config import (CKPT_ROOT, COMMON, EPOCHS, EVAL_NUM_SAMPLES, LOG_DIR,
                    OMP_THREADS_PER_JOB, PROTEINS, eval_cmd, train_cmd)

# The ablation: no pairwise-distance regularizer.
PAIRWISE_DISTANCE_WEIGHT = 0.0


def build_jobs(smoke=False):
    jobs = []
    for protein, pcfg in PROTEINS.items():
        cfg = dict(COMMON)
        cfg.update(pcfg)
        cfg["pairwise_distance_weight"] = PAIRWISE_DISTANCE_WEIGHT

        run_name = f"{protein}_nopairwise"
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

        train_args = train_cmd(cfg, run_name, epochs)

        eval_args = eval_cmd(cfg, run_name, eval_num_samples, eval_N)

        jobs.append((run_name, train_args, eval_args))
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


def run_job(index, run_name, train_args, eval_args, gpu_ids, wandb_mode):
    gpu_id = gpu_ids[index % len(gpu_ids)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["OMP_NUM_THREADS"] = OMP_THREADS_PER_JOB
    if wandb_mode:
        env["WANDB_MODE"] = wandb_mode

    log_path = os.path.join(LOG_DIR, f"{run_name}.log")
    print(f"=== Launching {run_name} on GPU {gpu_id} (log: {log_path}) ===", flush=True)

    with open(log_path, "w") as log_file:
        for stage, args in (("train", train_args), ("eval", eval_args)):
            log_file.write(f"\n===== {stage}: {' '.join(args)} =====\n")
            log_file.flush()
            rc = subprocess.run(args, env=env, stdout=log_file,
                                stderr=subprocess.STDOUT).returncode
            print(f"=== {run_name} {stage} finished (exit {rc}) ===", flush=True)
            if rc != 0:
                return run_name, stage, rc, log_path
    return run_name, "done", 0, log_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="3 epochs / 200 eval samples, offline wandb, smoke_ run names")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated run-name substrings to keep (e.g. 1a62_A)")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids (default: least-loaded ones)")
    parser.add_argument("--wandb_mode", type=str, default=None,
                        help="Value for WANDB_MODE (smoke defaults to offline)")
    cli = parser.parse_args()

    jobs = build_jobs(smoke=cli.smoke)
    if cli.only:
        keys = [k.strip() for k in cli.only.split(",") if k.strip()]
        jobs = [j for j in jobs if any(k in j[0] for k in keys)]

    wandb_mode = cli.wandb_mode or ("offline" if cli.smoke else None)

    if cli.gpus:
        gpu_ids = [int(g) for g in cli.gpus.split(",") if g.strip()]
    else:
        gpu_ids = free_gpus(len(jobs))
    if not gpu_ids:
        gpu_ids = [0]
    if len(gpu_ids) < len(jobs):
        print(f"Warning: {len(jobs)} jobs but only {len(gpu_ids)} GPUs; "
              f"some will share (round-robin).")

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Running {len(jobs)} jobs on GPUs {gpu_ids}: {[j[0] for j in jobs]}")

    with ThreadPoolExecutor(max_workers=max(1, len(jobs))) as pool:
        futures = [pool.submit(run_job, i, *job, gpu_ids, wandb_mode)
                   for i, job in enumerate(jobs)]
        results = [f.result() for f in futures]

    failures = [r for r in results if r[2] != 0]
    if failures:
        print(f"\n{len(failures)}/{len(jobs)} jobs failed:")
        for run_name, stage, rc, log_path in failures:
            print(f"  {run_name}: {stage} exit {rc}, see {log_path}")
        sys.exit(1)
    print(f"\nAll {len(jobs)} jobs completed successfully.")

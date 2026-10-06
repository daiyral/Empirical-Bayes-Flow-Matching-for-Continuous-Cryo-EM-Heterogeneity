import os
import sys

from cryoflow import paths

# Entry points the runners shell out to.
TRAIN_PY = os.path.join(paths.ROOT, "train", "train.py")
EVAL_PY = os.path.join(paths.ROOT, "eval", "eval.py")

LOG_DIR = paths.LOGS
CKPT_ROOT = paths.CHECKPOINTS

# Per-target inputs and the two hyperparameters that are tuned per protein.
PROTEINS = {
    "1a62_A": dict(
        pdb_file=paths.md_files("1a62_A")[0],
        xtc_file=paths.md_files("1a62_A")[1],
        data_dir=paths.sim_dataset("1a62_A"),
        pairwise_distance_weight=0.8,
        warmup_epochs=50,
        eval_N=20,
    ),
    "1egw_C": dict(
        pdb_file=paths.md_files("1egw_C")[0],
        xtc_file=paths.md_files("1egw_C")[1],
        data_dir=paths.sim_dataset("1egw_C"),
        pairwise_distance_weight=0.3,
        warmup_epochs=200,
        eval_N=6,
    ),
    "3rjs_A": dict(
        pdb_file=paths.md_files("3rjs_A")[0],
        xtc_file=paths.md_files("3rjs_A")[1],
        data_dir=paths.sim_dataset("3rjs_A"),
        pairwise_distance_weight=0.3,
        warmup_epochs=50,
        eval_N=20,
    ),
    "7r79_C": dict(
        pdb_file=paths.md_files("7r79_C")[0],
        xtc_file=paths.md_files("7r79_C")[1],
        data_dir=paths.sim_dataset("7r79_C"),
        pairwise_distance_weight=0.3,
        warmup_epochs=200,
        eval_N=7,
    ),
}

COMMON = dict(
    consecutive_distance_weight=1,
    flow_lr=2e-4,
    flow_model_path=None,
    flow_retrain_every=1,
    flow_train_batch_size=256,
    flow_train_epochs=1,
    flow_warmup_epochs=50,
    flow_weight=0.8,
    guided_N=10,
    guided_n_candidates=5,
    guided_N_min=3,
    guided_N_rampup_retrains=5,
    guided_strength=0.5,
    guided_warmup_steps=3,
    images_per_frame=1,
    log_every_n_epochs=50,
    num_frames=10000,
    render_loss_type="mae",
    render_weight=0.8,
    sigma=1,
    structure_lr=0.1,
    use_first_frame_initial=True,   # ignored when --latent_init_npz is given
)

EPOCHS = 3000
EVAL_NUM_SAMPLES = 10000

# CPU threads per job, so N concurrent jobs don't each grab all cores.
OMP_THREADS_PER_JOB = "16"


def train_cmd(cfg, run_name, epochs, extra=(), first_frame_initial=True):
    return [
        sys.executable, TRAIN_PY,
        "--pdb_file", cfg["pdb_file"],
        "--xtc_file", cfg["xtc_file"],
        "--data_dir", cfg["data_dir"],
        "--epochs", str(epochs),
        "--flow_lr", str(cfg["flow_lr"]),
        "--flow_retrain_every", str(cfg["flow_retrain_every"]),
        "--flow_train_batch_size", str(cfg["flow_train_batch_size"]),
        "--flow_train_epochs", str(cfg["flow_train_epochs"]),
        "--flow_warmup_epochs", str(cfg["flow_warmup_epochs"]),
        "--flow_weight", str(cfg["flow_weight"]),
        "--guided_N", str(cfg["guided_N"]),
        "--guided_n_candidates", str(cfg["guided_n_candidates"]),
        "--guided_N_min", str(cfg["guided_N_min"]),
        "--guided_N_rampup_retrains", str(cfg["guided_N_rampup_retrains"]),
        "--guided_strength", str(cfg["guided_strength"]),
        "--guided_warmup_steps", str(cfg["guided_warmup_steps"]),
        "--images_per_frame", str(cfg["images_per_frame"]),
        "--log_every_n_epochs", str(cfg["log_every_n_epochs"]),
        "--num_frames", str(cfg["num_frames"]),
        "--pairwise_distance_weight", str(cfg["pairwise_distance_weight"]),
        "--render_loss_type", cfg["render_loss_type"],
        "--render_weight", str(cfg["render_weight"]),
        "--sigma", str(cfg["sigma"]),
        "--structure_lr", str(cfg["structure_lr"]),
        *(("--use_first_frame_initial", str(cfg["use_first_frame_initial"]))
          if first_frame_initial else ()),
        "--warmup_epochs", str(cfg["warmup_epochs"]),
        *extra,
        # eval.py runs the metrics separately
        "--post_train_analysis", "False",
        "--run_name", run_name,
    ]


def eval_cmd(cfg, run_name, num_samples, N):
    return [
        sys.executable, EVAL_PY,
        "--checkpoint_path", paths.checkpoint(run_name),
        "--pdb_file", cfg["pdb_file"],
        "--xtc_file", cfg["xtc_file"],
        "--data_dir", cfg["data_dir"],
        "--num_samples", str(num_samples),
        "--N", str(N),
        "--run_name", run_name,
    ]

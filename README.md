# Empirical Bayes Flow Matching for Continuous Cryo-EM Heterogeneity

## Installation

Requires Python 3.12.

```bash
python -m venv cryoflowvenv && source cryoflowvenv/bin/activate
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -r requirements.txt     # includes JAX (CUDA) and cryoJAX for the renderer
pip install -e . --no-deps          # makes `import cryoflow` work everywhere
wandb login                         # training and eval log to wandb (or WANDB_MODE=offline)
```

**PyMOL** (optional) renders the ground-truth vs. generated trajectory GIFs during eval; the
metrics don't need it. It runs as an external `pymol` process, so install it system-wide:

```bash
sudo apt install pymol              # or: conda install -c conda-forge pymol-open-source
```

PyMOL opens a window, so on a headless machine run eval under `xvfb-run -a`
(`sudo apt install xvfb`).

---

## Hardware

The protein experiments ran on NVIDIA H100 80GB GPUs, one protein per GPU: ~13 h per protein
(~3 days on an RTX 4090).

---

## Dataset

**1. MD trajectories from [ATLAS](https://www.dsimb.inserm.fr/ATLAS)** (CC-BY-NC).
Examples below use four ATLAS proteins: **1a62_A**, **7r79_C**, **1egw_C**, **3rjs_A** — all
with replica R1, 10,000 frames, 1 particle image per frame. Download the `_protein` archive for
each target and unpack it to `data/md/<protein>/`:

```bash
for P in 1a62_A 7r79_C 1egw_C 3rjs_A; do
  mkdir -p "data/md/$P"
  curl -o "$P.zip" "https://www.dsimb.inserm.fr/ATLAS/database/ATLAS/$P/${P}_protein.zip"
  unzip -o "$P.zip" -d "data/md/$P" && rm "$P.zip"
done
```

~150 MB per target. Each gives `<protein>.pdb` plus three 100 ns replicas
(`_prod_R{1,2,3}_fit.xtc`, 10,000 frames at 10 ps) and their `.tpr` files. Everything here
uses replica **R1**. Expected layout:

```
data/md/1a62_A/1a62_A.pdb
data/md/1a62_A/1a62_A_prod_R1_fit.xtc
```

**2. Simulated particle stacks.** One micrograph per conformation, rendered with
[TEM-simulator](https://tem-simulator.sourceforge.net/). That site has the download, the user
manual and example inputs. Install it following the instructions there, then put the
`TEM-simulator` executable on your `PATH` (or set `TEM_SIMULATOR=/path/to/TEM-simulator`) so
`tools/simulate_cryo_em.py` finds it.

A TEM-simulator input file must be given with `--tem_input`; start from one of the examples
on the site. The simulation parameters we used are listed in the paper. The script fills in
the per-frame paths, so write `pdb_file_in`, `coord_file_out` and `image_file_out` as
`PDB_FILE_PLACEHOLDER`, `COORD_FILE_PLACEHOLDER` and `MRC_FILE_PLACEHOLDER`.

```bash
python tools/simulate_cryo_em.py \
  --pdb_file data/md/1a62_A/1a62_A.pdb \
  --xtc_file data/md/1a62_A/1a62_A_prod_R1_fit.xtc \
  --tem_input my_tem_input.txt \
  --output_dir "$CRYO_DATASET_ROOT/1a62_A_sim_output"
```

Training reads `<CRYO_DATASET_ROOT>/<protein>_sim_output/dataset`. Point
`CRYO_DATASET_ROOT` at wherever you have room; it defaults to `data/cryo-dataset` inside
the repo.

---

## Training

```bash
python train/train.py \
  --pdb_file  data/md/1a62_A/1a62_A.pdb \
  --xtc_file  data/md/1a62_A/1a62_A_prod_R1_fit.xtc \
  --data_dir  "$CRYO_DATASET_ROOT/1a62_A_sim_output/dataset" \
  --run_name  1a62_A_baseline \
  --epochs 3000 --num_frames 10000 \
  --pairwise_distance_weight 0.8 --warmup_epochs 50 \
  --post_train_analysis False
```

Writes `runs/checkpoints/1a62_A_baseline/last.ckpt` and
`runs/latents/1a62_A_baseline_parametric_structures.pt`, and logs to wandb
(`WANDB_MODE=offline` to disable). Re-running with `--resume_from_checkpoint` picks the
same wandb run back up. `python train/train.py --help` lists all hyperparameters.

The configuration for the four proteins used in the paper is in
[experiments/config.py](experiments/config.py): the shared hyperparameters plus the
per-protein `--pairwise_distance_weight`, `--warmup_epochs` and evaluation `--N`. To train and
then evaluate all four with it, one per GPU:

```bash
python experiments/run_eb_fm.py --suffix _baseline
```

Training is not seeded, so expect the metrics to differ by up to a few tenths of an Å between
runs of the same configuration.

---

## Evaluation

```bash
python eval/eval.py \
  --checkpoint_path runs/checkpoints/1a62_A_baseline/last.ckpt \
  --pdb_file  data/md/1a62_A/1a62_A.pdb \
  --xtc_file  data/md/1a62_A/1a62_A_prod_R1_fit.xtc \
  --data_dir  "$CRYO_DATASET_ROOT/1a62_A_sim_output/dataset" \
  --run_name  1a62_A_baseline --num_samples 10000 --N 20
```

Samples the trained flow prior, matches the generated ensemble against the ground-truth
trajectory and logs all metrics to wandb project `cryo-em-flow-eval`. Structures,
trajectories and plots land in `runs/results/1a62_A_baseline/`. `--N` is the number of
sampling steps.

To evaluate all four targets at once — one per GPU — use the run-name suffix the training
launcher used:

```bash
python experiments/run_eval_all.py --suffix _ctfpose_small
python experiments/run_eval_all.py --suffix _nopairwise --only 1a62_A,3rjs_A
```

---

## Ablation

Each launcher runs all four proteins, one per GPU, train then eval, and retries a crashed
stage from `last.ckpt`.

| Command | What it tests | Run names |
|---|---|---|
| `python experiments/run_ctf_pose_errors.py` | small pose + defocus error on **every** particle | `<P>_ctfpose_small` |
| `python experiments/run_no_pairwise_ablation.py` | backbone (pairwise-distance) loss removed | `<P>_nopairwise` |
| `python experiments/run_noisy_init.py` | latents = frame 0 + smooth per-latent noise | `<P>_noisyinit` |
| `python experiments/run_ctfpose_noisyinit.py` | small pose/defocus error + noisy init at once — does symmetry breaking still help when the forward model is also wrong? | `<P>_ctfpose_small_noisyinit` |

`experiments/run_eval_all.py` re-runs just the eval half of any of these.

The ablation table also uses `<P>_ctfpose_err10` and `<P>_ctfpose_err30`, where 10% / 30% of
the particles get a random orientation and a defocus error of up to 40%. To produce them, set
`ERROR_FRAC = 0.1` (or `0.3`), `POSE_ERROR_DEG = 0.0`, `CTF_DEFOCUS_ERROR = 0.4` and
`TAG = "err10"` (or `"err30"`) at the top of `experiments/run_ctf_pose_errors.py`.

The AlphaFold comparisons need [cryodrgn](https://github.com/ml-struct-bio/cryodrgn) and
[guided_alphafold](https://github.com/sai-advaith/guided_alphafold): download and set up each
on its own, following its instructions, then use the scripts in the two AlphaFold sections
below.

---

## Tables

Each script writes LaTeX + PNG (+ Markdown, metrics JSON) into `reports/`. Every row, including
the plain EB-FM baseline, is pulled from your own finished wandb eval runs — none of the numbers
are hardcoded. The baseline is looked up as `<protein>_baseline`, matching the `--run_name` used
in Training/Evaluation above; run `eval/eval.py` with that name first if a table script
complains it's missing.

| Script | Source | Output in `reports/` |
|---|---|---|
| `make_eb_fm_table.py` | wandb `<P>_baseline` (or `--suffix <P><suffix>`) | `table_eb_fm.*` — main EB-FM results |
| `make_compare_models_table.py` | wandb `<P>_baseline`, `<P>_vae` (from `vae/eval_vae.py`), `runs/alphafold/*/metrics.json` | `table_compare_models.*` — EB-FM vs VAE vs AF3 MSA sub vs cryoDRGN-guided AF3 |
| `make_ablation_table.py` | wandb `<P>_baseline` + the ablation run suffixes | `table_ablations.*` — EB-FM vs small-error-all, p=0.1, p=0.3 |

```bash
python vis/tables/make_ablation_table.py     # after the ctf/pose error runs finish
```

---

## MNIST and two moons

Each lives in its own folder; outputs go to `runs/` and `reports/` like everything else.

### MNIST

Diffusion, EB-FM w/o PL and EB-FM, each with M = 50 and 100 observations. MNIST is
downloaded to `data/mnist/` on first use.

```bash
python mnist/train_mnist.py --run_name mnist_diffusion_m50 --model_type diffusion --n_measurements 50 --seed 0
python mnist/train_mnist.py --run_name mnist_ebfm_nopl_m50 --model_type flow --n_measurements 50 --seed 1
python mnist/train_mnist.py --run_name mnist_ebfm_m50 --model_type flow --PL True --n_measurements 50 --seed 0
python mnist/eval_mnist.py --run_name mnist_ebfm_m50    # once per run
python mnist/make_mnist_table.py                        # -> reports/table_mnist.*
```

Same for M = 100 (`--n_measurements 100`, `_m100` names, `--seed 0`). Keep the
`mnist_<method>_m<M>` run names, the table script looks them up. Each run takes 5-11 h on
one GPU.

### Two moons

```bash
python two-moons/train_two_moons.py                                     # all 36 cells, ~2.5 min each
python two-moons/train_two_moons.py --method ebfm --forward sin --M 8   # a single cell
python two-moons/make_two_moons_figure.py                               # -> reports/figure_two_moons.{png,pdf}
```

---

## VAE comparison

Implementation of ["Inferring a Continuous Distribution of Atom Coordinates from Cryo-EM
Images using VAEs"](https://arxiv.org/abs/2106.14108). It trains a variational autoencoder
that maps each particle image to a latent conformation and decodes it back to atomic
coordinates, giving the VAE column of the model-comparison table on the same particle stacks
used for EB-FM.

```bash
python vae/train_vae.py --protein 1a62_A          # ~35 min on one GPU
python vae/eval_vae.py --protein 1a62_A           # logs 1a62_A_vae to wandb
python vis/tables/make_compare_models_table.py    # once all four proteins are evaluated
```

---

## cryoDRGN-guided AlphaFold3

cryoDRGN reconstructs per-particle latents/volumes (its own heterogeneity model) from the same
particle stacks; `guided_alphafold` then conditions AlphaFold3 on each latent to predict a
matching conformation. Results come from two external repos — download and set up each one on
its own, following its instructions:

1. [cryodrgn](https://github.com/ml-struct-bio/cryodrgn)
2. [guided_alphafold](https://github.com/sai-advaith/guided_alphafold)

Both are used unmodified, driven by scripts in `tools/`:

```bash
# in the cryoDRGN environment: convert the dataset, train, decode one map per particle
python tools/cryodrgn_maps.py prepare --sim_dir "$CRYO_DATASET_ROOT/1a62_A_sim_output" --out_dir cdrgn/1a62_A
cryodrgn train_vae cdrgn/1a62_A/particles.mrcs --poses cdrgn/1a62_A/pose.pkl --ctf cdrgn/1a62_A/ctf.pkl \
    --zdim 8 -n 100 -b 16 --uninvert-data --no-window -o cdrgn/1a62_A/train
python tools/cryodrgn_maps.py decode --workdir cdrgn/1a62_A/train --epoch 100 --n 1250 --out_dir cdrgn/1a62_A/maps

# in the guided_alphafold environment: 8 structures per map
python tools/af3_cryodrgn_guided.py --guided_alphafold /path/to/guided_alphafold \
    --maps_dir cdrgn/1a62_A/maps --sim_dir "$CRYO_DATASET_ROOT/1a62_A_sim_output" \
    --md_pdb data/md/1a62_A/1a62_A.pdb --run_name 1a62_A --out_dir af3_guided/1a62_A

# back in this repo's environment: score the ensemble
python eval/eval_guided_alphafold.py --ensemble_dir af3_guided/1a62_A --protein 1a62_A
```

A map takes about 100 s on one H100 (about 35 GPU-hours per protein); `--shard` / `--nshards`
split the maps over GPUs.

---

## AF3 MSA subsampling baseline

An unguided AlphaFold3 ensemble: each run gets a random subsample of the full MSA instead of
the whole thing, which leaves the model less certain of a single fold, so its predictions
scatter across conformations instead of collapsing to one. It runs inside an unmodified
[guided_alphafold](https://github.com/sai-advaith/guided_alphafold) checkout (built on
[Protenix](https://github.com/bytedance/Protenix), a PyTorch AlphaFold3 reproduction), keeping
64 MSA sequences per subsample and drawing 8 structures from each:

```bash
# in the guided_alphafold environment
python tools/af3_msa_subsampling.py --guided_alphafold /path/to/guided_alphafold \
    --md_pdb data/md/1a62_A/1a62_A.pdb --run_name 1a62_A --out_dir af3_msa/1a62_A

# back in this repo's environment: score the ensemble
python eval/eval_unguided_alphafold.py --ensemble_dir af3_msa/1a62_A --protein 1a62_A
```

`--shard` / `--nshards` split the work over GPUs. The scoring scripts accept any folder of
`.pdb` or `.cif` files.

---

## Citation

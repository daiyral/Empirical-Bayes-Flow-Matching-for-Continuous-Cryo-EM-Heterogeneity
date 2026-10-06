import os

ROOT = os.environ.get(
    "CRYOFLOW_ROOT",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

# ---- inputs -------------------------------------------------------------------
DATA = os.path.join(ROOT, "data")
MD = os.path.join(DATA, "md")                 # ATLAS pdb/xtc, one dir per target
# TEM-simulator particle stacks; large, usually outside the repo
CRYO_DATASET = os.environ.get("CRYO_DATASET_ROOT", os.path.join(DATA, "cryo-dataset"))
# TEM-simulator executable (https://tem-simulator.sourceforge.net/), looked up on PATH by default
TEM_SIMULATOR = os.environ.get("TEM_SIMULATOR", "TEM-simulator")

# ---- outputs ------------------------------------------------------------------
RUNS = os.environ.get("CRYOFLOW_RUNS", os.path.join(ROOT, "runs"))
CHECKPOINTS = os.path.join(RUNS, "checkpoints")        # <run_name>/last.ckpt
RESULTS = os.path.join(RUNS, "results")                # <run_name>/ eval artifacts
LATENTS = os.path.join(RUNS, "latents")                # <run_name>_parametric_structures.pt
LATENT_INITS = os.path.join(RUNS, "latent_inits")      # noisy / cryoDRGN latent inits
LOGS = os.path.join(RUNS, "logs")
ANALYSIS = os.path.join(RUNS, "analysis")              # gifs, optimization trajectories
FLOW_CHECKPOINTS = os.path.join(RUNS, "flow_checkpoints")
GUIDANCE = os.path.join(RUNS, "guidance")
ALPHAFOLD = os.path.join(RUNS, "alphafold")            # guided/ and unguided/
REPORTS = os.path.join(ROOT, "reports")                # paper tables (.tex/.png/.json)

# ---- the four ATLAS targets used in the paper ---------------------------------
PROTEINS = ["1a62_A", "7r79_C", "1egw_C", "3rjs_A"]


def md_files(protein):
    d = os.path.join(MD, protein)
    return (os.path.join(d, f"{protein}.pdb"),
            os.path.join(d, f"{protein}_prod_R1_fit.xtc"))


def sim_dataset(protein):
    return os.path.join(CRYO_DATASET, f"{protein}_sim_output", "dataset")


def checkpoint(run_name):
    return os.path.join(CHECKPOINTS, run_name, "last.ckpt")


def results(run_name):
    return os.path.join(RESULTS, run_name)


def latents(run_name):
    return os.path.join(LATENTS, f"{run_name}_parametric_structures.pt")

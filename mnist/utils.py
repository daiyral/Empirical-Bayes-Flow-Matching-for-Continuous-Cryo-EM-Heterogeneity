import gzip
import hashlib
import os
import urllib.request

import numpy as np
import torch

from cryoflow import paths

DATA_ROOT = os.path.join(paths.DATA, "mnist")
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"
MNIST_MD5 = "f68b3c2dcbeaaa9fbdd348bbdeb94873"

# (run tag, table label) x observation counts of the paper table
METHODS = [("diffusion", "Diffusion"), ("ebfm_nopl", "EB-FM w/o PL"), ("ebfm", "EB-FM")]
OBS = [50, 100]


def run_name(tag, m):
    return f"mnist_{tag}_m{m}"


def metrics_path(run):
    return os.path.join(paths.results(run), "metrics.json")


def str2bool(x):
    return str(x).lower() == "true"


def load_mnist(root=DATA_ROOT, num_images=None):
    path = os.path.join(root, os.path.basename(MNIST_URL))
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        urllib.request.urlretrieve(MNIST_URL, tmp)
        os.replace(tmp, path)
    with open(path, "rb") as f:
        raw = f.read()
    if hashlib.md5(raw).hexdigest() != MNIST_MD5:
        raise RuntimeError(f"corrupt MNIST file: {path}")
    images = np.frombuffer(gzip.decompress(raw), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    # first num_images of the train split, normalized to [-1, 1]
    return torch.from_numpy(images[:num_images].copy()).float().div(255).sub(0.5).div(0.5)

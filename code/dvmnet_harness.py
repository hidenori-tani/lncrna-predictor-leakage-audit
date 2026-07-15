"""
Shared harness for running the DVMnet deep model itself under alternative evaluation protocols.

Upstream repo (external/DVMnet, github.com/liuliwei1980/DVMnet) is NOT modified. Two shims make
its CUDA-only code run on CPU without touching any model math:
  1. torch.Tensor.cuda(...) -> identity (the model calls .cuda(0) on already-CPU tensors)
  2. torch.cuda.set_device(...) -> no-op
Everything else (architecture, hyperparameters, feature construction) is imported verbatim.

Node layout in dataset/index_value.csv: indices 0..283 = lncRNA, 284..803 = miRNA.
"""
import os
import sys
import pickle
import random
import string

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DVMNET_DIR = os.environ.get(
    "DVMNET_DIR", os.path.join(HERE, "..", "external", "DVMnet")
)
DVMNET_DIR = os.path.abspath(DVMNET_DIR)

N_LNC = 284
N_MI = 520
N_NODES = N_LNC + N_MI

# ---------------------------------------------------------------- device shim
_TENSOR_CUDA_ORIG = torch.Tensor.cuda


def _cuda_identity(self, *args, **kwargs):
    return self


def install_cpu_shim():
    """Make upstream's .cuda(0) calls no-ops so the model runs unchanged on CPU."""
    torch.Tensor.cuda = _cuda_identity
    torch.cuda.set_device = lambda *a, **k: None


install_cpu_shim()

if DVMNET_DIR not in sys.path:
    sys.path.insert(0, DVMNET_DIR)


def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ------------------------------------------------------- restricted unpickling
# vectors_{lnc,mi}.pkl ship with the third-party DVMnet repo, so they are untrusted
# input. Static pickletools inspection shows only numpy reconstruction opcodes, and
# this loader enforces that: anything other than a numpy ndarray/dtype raises rather
# than executing. Both files load cleanly under it (dicts of float64 arrays).
_ALLOWED_PICKLE_GLOBALS = {
    ("numpy", "dtype"),
    ("numpy", "ndarray"),
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked pickle global: {module}.{name}")


def load_array_pickle(path):
    with open(path, "rb") as f:
        return _RestrictedUnpickler(f).load()


# ---------------------------------------------------------------- data loading
def _read_sequences(match_table, unique_lnc, unique_mi):
    """Verbatim re-implementation of main.py's sequence extraction."""
    lnc_seq, mi_seq = [], []
    for i in unique_lnc:
        seq = list(match_table[match_table["lncrna"] == i]["lncrna_seq"])[0]
        seq = seq.translate(str.maketrans("", "", string.punctuation))
        lnc_seq.append(seq)
    for i in unique_mi:
        seq = list(match_table[match_table["mirna"] == i]["mirna_seq"])[0]
        seq = seq.replace(".", "")
        if "," in seq:
            seq = seq.split(",")[0]
        mi_seq.append(seq)
    return lnc_seq, mi_seq


def build_features(embed_cache=None, doc2vec_seed=None):
    """Build x_embedding / x_lnc / x_mi exactly as main.py does.

    The doc2vec step is stochastic and is the slowest part, so the resulting
    node embedding is cached: every protocol must see the SAME embedding, or
    protocol effects would be confounded with doc2vec noise.
    """
    from utils import k_mers, train_doc2vec_model, get_vector_embeddings

    cwd = os.getcwd()
    os.chdir(DVMNET_DIR)
    try:
        match_table = pd.read_csv("dataset/mirna_lncrna_interaction.csv")
        unique_lnc = list(set(match_table["lncrna"]))
        unique_mi = list(set(match_table["mirna"]))
        graph_table = pd.read_csv("dataset/index_value.csv")
        graph_label = list(graph_table["rna"])

        if embed_cache and os.path.exists(embed_cache):
            graph_embedding = np.load(embed_cache)["graph_embedding"]
        else:
            lnc_seq, mi_seq = _read_sequences(match_table, unique_lnc, unique_mi)
            all_mers = [k_mers(3, s) for s in lnc_seq] + [k_mers(3, s) for s in mi_seq]
            all_name = unique_lnc + unique_mi
            if doc2vec_seed is not None:
                np.random.seed(doc2vec_seed)
                random.seed(doc2vec_seed)
            pretrain_model = train_doc2vec_model(all_mers, all_name)
            vectors = get_vector_embeddings(all_mers, all_name, pretrain_model)
            graph_embedding = np.zeros((len(graph_label), 100))
            for node, vec in vectors.items():
                graph_embedding[graph_label.index(node)] = vec
            if embed_cache:
                np.savez_compressed(embed_cache, graph_embedding=graph_embedding)

        vectors_lnc = load_array_pickle("vectors_lnc.pkl")
        vectors_mi = load_array_pickle("vectors_mi.pkl")

        graph_lnc = np.zeros((N_LNC, 64, 64))
        for node, vec in vectors_lnc.items():
            graph_lnc[graph_label.index(node)] = vec
        graph_mi_full = np.zeros((len(graph_label), 16, 16))
        for node, vec in vectors_mi.items():
            graph_mi_full[graph_label.index(node)] = vec
        graph_mi = graph_mi_full[N_LNC:]

        ser_di = pd.read_csv("dataset/di.csv")
        mi_sub = pd.read_csv("dataset/sub.csv")
        node_table = pd.read_csv("dataset/node_link.csv")
    finally:
        os.chdir(cwd)

    return dict(
        x_embedding=torch.tensor(graph_embedding).float(),
        x_lnc=torch.tensor(graph_lnc).float(),
        x_mi=torch.tensor(graph_mi).float(),
        graph_label=graph_label,
        ser_di=ser_di,
        mi_sub=mi_sub,
        node_table=node_table,
    )


def edges_as_arrays(node_table):
    u = np.array([int(x) for x in node_table["node1"]])
    v = np.array([int(x) for x in node_table["node2"]])
    return u, v


def normalize_edge_orientation(u, v):
    """Return (lnc_idx, mi_idx) regardless of which column holds which."""
    lnc = np.where(u < N_LNC, u, v)
    mi = np.where(u < N_LNC, v, u)
    return lnc, mi

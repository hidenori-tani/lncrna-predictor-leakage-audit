#!/usr/bin/env python3
"""Read the trained LncPTPred model that the authors released, without executing it.

Reviewer 2 (R1 round), point 3: "Ideally, the authors should reproduce the model using the full
dataset and the original training settings, where available."

The released repository (github.com/zglabDIB/lncptpred @ 621391287402) ships `models/models.zip`,
which holds two pickles: `StandardScaler_lncRNA_Protein.sav` and `LightGBM_lncRNA_Protein.sav`
(an `lightgbm.sklearn.LGBMClassifier`, lightgbm 2.3.1 per `requirement.txt`). The R1 analysis
never opened them: the local clone held `dataset/` only.

A pickle is a program, so neither file is loaded with `pickle.load`. `find_class` below never
returns a real object: every global the stream names becomes an inert placeholder class that only
records its arguments and state. Nothing from the stream can run. The numpy arrays inside the
scaler are decoded from their raw bytes by hand.

What is extracted:
  * every constructor parameter of the LGBMClassifier (the sklearn-level training settings);
  * the LightGBM model string (the Booster's pickled state), written to a text file so that it can
    be loaded with `lightgbm.Booster(model_file=...)`, which parses text and executes nothing;
  * the header of that model string, its `parameters:` block, the number of trees, and the root
    `internal_count` of the first tree (= number of training rows, when bagging is off);
  * the scaler's `n_samples_seen_` (= number of rows it was fitted on) and its 720 means/scales.

Usage:  python3 -I inspect_released_lncptpred_model.py <dir with the two .sav files> <out dir>

The out dir holds a text copy of a third-party model and must stay outside git (the upstream
repository carries no licence): use external/LncPTPred_released_models/derived/, which is ignored.
Only released_model_report.json is copied into revision_r2/provenance/.
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np


class Inert:
    """Placeholder for any global named in the pickle stream. Records, never acts."""
    qual = "?"

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.state = None

    def __setstate__(self, state):
        self.state = state

    # SETITEMS / APPENDS land here when the placeholder stands in for a container type
    # (e.g. collections.OrderedDict); the items are recorded, nothing else happens.
    def __setitem__(self, key, value):
        self.__dict__.setdefault("items", {})[key] = value

    def append(self, value):
        self.__dict__.setdefault("seq", []).append(value)

    def extend(self, values):
        self.__dict__.setdefault("seq", []).extend(values)


class InertUnpickler(pickle.Unpickler):
    # Safe by construction: every GLOBAL / STACK_GLOBAL / INST / OBJ / EXT* opcode resolves through
    # find_class, which here returns a fresh Inert subclass, so REDUCE / NEWOBJ / BUILD can only ever
    # call Inert.__init__ / Inert.__setstate__. persistent_load is not defined (raises by default).
    seen_globals = set()

    def find_class(self, module, name):
        qual = f"{module}.{name}"
        InertUnpickler.seen_globals.add(qual)
        return type(qual, (Inert,), {"qual": qual})


def inert_load(path):
    with open(path, "rb") as fh:
        return InertUnpickler(fh).load()


def decode_ndarray(obj):
    """numpy.core.multiarray._reconstruct(...) + BUILD(state) -> np.ndarray, from raw bytes."""
    if not isinstance(obj, Inert) or obj.state is None:
        return obj
    _version, shape, dtype_obj, is_fortran, raw = obj.state
    dtype_str = dtype_obj.args[0]
    endian = dtype_obj.state[1] if dtype_obj.state else "<"
    if endian in ("|", "="):
        endian = "<"
    dt = np.dtype(dtype_str).newbyteorder(endian)
    arr = np.frombuffer(raw, dtype=dt)
    return arr.reshape(shape, order="F" if is_fortran else "C")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def plain(v):
    """JSON-safe view of a decoded value."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, Inert):
        return f"<{v.qual}>"
    if isinstance(v, np.ndarray):
        return f"<ndarray {v.dtype} {v.shape}>"
    return repr(v)[:80]


def main(src, out):
    os.makedirs(out, exist_ok=True)
    report = {"sources": {}}
    for name in ("LightGBM_lncRNA_Protein.sav", "StandardScaler_lncRNA_Protein.sav"):
        p = os.path.join(src, name)
        report["sources"][name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}

    # ---- the scaler
    sc = inert_load(os.path.join(src, "StandardScaler_lncRNA_Protein.sav"))
    st = sc.state if isinstance(sc.state, dict) else {}
    scaler = {"class": sc.qual}
    for k, v in st.items():
        dv = decode_ndarray(v)
        if isinstance(dv, np.ndarray):
            scaler[k] = {"dtype": str(dv.dtype), "shape": list(dv.shape)}
            if k in ("n_samples_seen_",) or dv.size == 1:
                scaler[k]["value"] = dv.ravel().tolist()
            np.save(os.path.join(out, f"scaler_{k}.npy"), np.asarray(dv))
        else:
            scaler[k] = plain(dv)
    report["scaler"] = scaler

    # ---- the classifier
    clf = inert_load(os.path.join(src, "LightGBM_lncRNA_Protein.sav"))
    cst = clf.state if isinstance(clf.state, dict) else {}
    booster = cst.get("_Booster")
    report["classifier"] = {"class": clf.qual,
                            "params": {k: plain(v) for k, v in cst.items() if k != "_Booster"}}
    bst = booster.state if isinstance(booster, Inert) and isinstance(booster.state, dict) else {}
    report["booster_state_keys"] = sorted(bst.keys())
    report["booster_params"] = plain(bst.get("params"))
    model_str = bst.get("handle")
    if not isinstance(model_str, str):
        raise SystemExit("model string not found in the Booster state")
    model_path = os.path.join(out, "LightGBM_lncRNA_Protein.model.txt")
    with open(model_path, "w") as fh:
        fh.write(model_str)
    report["model_txt"] = {"path": model_path, "bytes": len(model_str), "sha256": sha256(model_path)}

    head, _, _ = model_str.partition("\nTree=0\n")
    report["model_header"] = {ln.split("=", 1)[0]: ln.split("=", 1)[1][:200]
                              for ln in head.splitlines() if "=" in ln and not ln.startswith("feature_names")}
    fn_line = next((ln for ln in head.splitlines() if ln.startswith("feature_names=")), "")
    report["model_header"]["n_feature_names"] = len(fn_line.split("=", 1)[1].split()) if fn_line else 0
    report["n_trees"] = model_str.count("\nTree=")
    # root internal_count of Tree=0
    t0 = model_str.split("\nTree=0\n", 1)[1].split("\n\n", 1)[0]
    for ln in t0.splitlines():
        if ln.startswith("internal_count="):
            report["tree0_root_internal_count"] = int(ln.split("=", 1)[1].split()[0])
        if ln.startswith("num_leaves="):
            report["tree0_num_leaves"] = int(ln.split("=", 1)[1])
    leaves = [int(ln.split("=", 1)[1]) for ln in model_str.splitlines() if ln.startswith("num_leaves=")]
    if leaves:
        report["num_leaves_over_trees"] = {"min": min(leaves), "max": max(leaves),
                                           "mean": round(sum(leaves) / len(leaves), 2)}
    pblock = model_str.split("\nparameters:\n", 1)
    if len(pblock) == 2:
        report["parameters_block"] = pblock[1].split("\nend of parameters", 1)[0].splitlines()
    # ---- the authors' own validation record (evals_result_ / best_score_), and the scaler's row count
    def items_of(o):
        return o.__dict__.get("items", {}) if isinstance(o, Inert) else (o if isinstance(o, dict) else {})

    def decode_scalar(o):
        # numpy.core.multiarray.scalar(dtype, raw_bytes) -> python number, decoded by hand
        if isinstance(o, Inert) and len(o.args) == 2 and isinstance(o.args[1], bytes):
            d = o.args[0]
            return np.frombuffer(o.args[1], dtype=np.dtype(d.args[0]).newbyteorder(
                (d.state[1] if d.state and d.state[1] not in ("|", "=") else "<")))[0].item()
        if isinstance(o, (int, float)):
            return o
        return plain(o)

    curves = {}
    for split_name, per_metric in items_of(cst.get("_evals_result")).items():
        for metric, values in items_of(per_metric).items():
            seq = values.__dict__.get("seq", []) if isinstance(values, Inert) else values
            vals = [float(decode_scalar(x)) for x in seq]
            bi = cst.get("_best_iteration")
            curves[f"{split_name}/{metric}"] = {
                "n_iterations_recorded": len(vals),
                "at_best_iteration": vals[bi - 1] if isinstance(bi, int) and 0 < bi <= len(vals) else None,
                "first": vals[0], "last": vals[-1], "max": max(vals), "min": min(vals),
                "argmax_1based": int(np.argmax(vals)) + 1, "argmin_1based": int(np.argmin(vals)) + 1}
            np.save(os.path.join(out, f"curve_{split_name}_{metric}.npy"), np.asarray(vals))
    report["evals_result_summary"] = curves
    report["best_score"] = {f"{s}/{m}": decode_scalar(v) for s, d in items_of(cst.get("_best_score")).items()
                            for m, v in items_of(d).items()}
    report["scaler"]["n_samples_seen_value"] = decode_scalar(st.get("n_samples_seen_"))
    report["globals_named_in_streams"] = sorted(InertUnpickler.seen_globals)

    with open(os.path.join(out, "released_model_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    return report


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    r = main(sys.argv[1], sys.argv[2])
    print(json.dumps({k: r[k] for k in ("classifier", "n_trees", "num_leaves_over_trees",
                                         "tree0_root_internal_count") if k in r}, indent=1)[:4000])
    print("scaler n_samples_seen_:", r["scaler"].get("n_samples_seen_value"))
    print("evals_result:", json.dumps(r.get("evals_result_summary"), indent=1))
    print("best_score:", r.get("best_score"))
    print("n_features_in_:", r["scaler"].get("n_features_in_"))
    print("parameters block lines:", len(r.get("parameters_block", [])))

#!/usr/bin/env python
"""Path A (decisive): score the shipped trained RNAlight LightGBM model on the
shipped test set with our featurization; expect the exact reported 0.7833.

SECURITY: unpickles a model from the trusted, published github.com/YangLab/RNAlight
repo (already cloned/inspected); risk accepted for reproduction only.
"""
import sys, types, itertools, numpy as np, pandas as pd, joblib
from sklearn import metrics

# The shipped .pkl was dumped with the old sklearn.externals.joblib (removed in
# sklearn >=0.23); shim it to modern joblib so the on-disk format still loads.
sys.modules.setdefault("sklearn.externals", types.ModuleType("sklearn.externals"))
sys.modules["sklearn.externals.joblib"] = joblib

# --- path config: edit these or set env vars. Defaults assume repo layout code/ + data/ ---
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)
DATA = _os.environ.get('AUDIT_DATA', _os.path.join(_ROOT, 'data'))
FIG  = _os.environ.get('AUDIT_FIGS', _os.path.join(_ROOT, 'figures'))
RNALIGHT_DIR = _os.environ.get('RNALIGHT_DIR', _os.path.join(_ROOT, 'external', 'RNAlight'))
DVMNET_DIR   = _os.environ.get('DVMNET_DIR',   _os.path.join(_ROOT, 'external', 'DVMnet'))
# --- end path config ---
REPO = RNALIGHT_DIR
TEST = f"{REPO}/lncRNA/03_Model_Construction/lncRNA_sublocation_TestSet.tsv"
MODEL = f"{REPO}/lncRNA/03_Model_Construction/01_Machine_Learning_Model/01_ML_Model_Output/LightGBM/best_LightGBM_model.pkl"

NUC = ["A","C","G","T"]
KMERS = ["".join(t) for k in (3,4,5) for t in itertools.product(NUC, repeat=k)]

def featurize(df):
    raw = np.zeros((len(df), len(KMERS)))
    cdna = df["cdna"].tolist()
    for j, mer in enumerate(KMERS):
        raw[:, j] = [s.count(mer) for s in cdna]
    return pd.DataFrame(raw / raw.sum(axis=1, keepdims=True), columns=KMERS, index=df.index)

test = pd.read_csv(TEST, sep="\t")
yte = test["tag"].values
Xte = featurize(test)

model = joblib.load(MODEL)
print("[A] model:", type(model).__name__)
# expected feature order from the trained booster
try:
    exp = list(model.booster_.feature_name())
    print(f"[A] booster feature_name n={len(exp)} first5={exp[:5]}  our first5={KMERS[:5]}  same_set={set(exp)==set(KMERS)}  same_order={exp==KMERS}")
    Xin = Xte[exp]  # align to booster's exact column order
except Exception as e:
    print(f"[A] feature-name align skipped ({type(e).__name__}: {e}); using our order")
    Xin = Xte

yprob = model.predict_proba(Xin)[:, 1]
auroc = metrics.roc_auc_score(yte, yprob)
auprc = metrics.average_precision_score(yte, yprob)
acc = metrics.accuracy_score(yte, (yprob>=0.5).astype(int))
mcc = metrics.matthews_corrcoef(yte, (yprob>=0.5).astype(int))
print(f"[A] AUROC={auroc:.10f}  AUPRC={auprc:.4f}  ACC={acc:.4f}  MCC={mcc:.4f}")
print(f"[A] reported AUROC=0.7833019077196096 | delta={abs(auroc-0.7833019077196096):.2e}")
print("[GATE]", "PASS (exact)" if abs(auroc-0.7833019077196096)<1e-6 else ("PASS (~)" if abs(auroc-0.7833019077196096)<0.01 else "CHECK"))

#!/usr/bin/env python
"""RNAlight reproduction GATE.

Goal: reproduce the reported lncRNA test-set AUROC = 0.7833 (RNAlight, Brief
Bioinform 2022) using the shipped Train/Test TSVs + shipped trained LightGBM
model, with our own re-implementation of the k-mer 3/4/5 featurization.

Two independent checks:
  (A) load the shipped best_LightGBM_model.pkl, re-score the shipped test set.
  (B) retrain a fresh LGBMClassifier (params read from the shipped model, else
      the notebook search space's typical best) on the shipped train set,
      eval on the shipped test set.

Passing the gate = (A) reproduces ~0.7833 (validates featurization + model),
and (B) lands in the same neighbourhood (validates we can retrain for the
later homology-aware re-split).
"""
# SECURITY NOTE: Path (A) below unpickles best_LightGBM_model.pkl. Unpickling
# executes arbitrary code and is only safe for TRUSTED sources. This file comes
# from the official, published academic repo github.com/YangLab/RNAlight
# (Yuan et al., Brief Bioinform 2022) which we cloned and inspected; risk accepted
# for reproduction. Path (B) retrains from scratch and needs NO pickle, so the
# gate does not depend solely on trusting the .pkl.
import sys, itertools, json
import numpy as np
import pandas as pd
from sklearn import metrics

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
MLDIR = f"{REPO}/lncRNA/03_Model_Construction/01_Machine_Learning_Model/01_ML_Model_Output"
TRAIN = f"{REPO}/lncRNA/03_Model_Construction/lncRNA_sublocation_TrainingSet.tsv"
TEST  = f"{REPO}/lncRNA/03_Model_Construction/lncRNA_sublocation_TestSet.tsv"
MODEL = f"{MLDIR}/LightGBM/best_LightGBM_model.pkl"
REPORTED_AUROC = 0.7833019077196096

# ---- k-mer 3/4/5 frequency featurization (matches RNAlight _count_kmer, k=345)
NUC = ["A", "C", "G", "T"]
def kmer_list(k):
    return ["".join(t) for t in itertools.product(NUC, repeat=k)]
KMERS = kmer_list(3) + kmer_list(4) + kmer_list(5)   # 64+256+1024 = 1344, dict-insertion order

def featurize(df):
    # raw non-overlapping counts via str.count (exactly as RNAlight: x.count(mer))
    raw = np.zeros((len(df), len(KMERS)), dtype=np.float64)
    cdna = df["cdna"].tolist()
    for j, mer in enumerate(KMERS):
        raw[:, j] = [s.count(mer) for s in cdna]
    # normalize each row by the total count across ALL 1344 k-mers (matches .apply(x/x.sum(),axis=1))
    rowsum = raw.sum(axis=1, keepdims=True)
    freq = raw / rowsum
    return pd.DataFrame(freq, columns=KMERS, index=df.index)

def main():
    train = pd.read_csv(TRAIN, sep="\t")
    test  = pd.read_csv(TEST,  sep="\t")
    print(f"[data] train={len(train)} test={len(test)} | train tag={dict(train.tag.value_counts())} test tag={dict(test.tag.value_counts())}")

    print("[featurize] computing k-mer 3/4/5 frequencies (this takes a few minutes)...")
    Xtr = featurize(train); ytr = train["tag"].values
    Xte = featurize(test);  yte = test["tag"].values
    print(f"[featurize] Xtrain={Xtr.shape} Xtest={Xte.shape} (expect 1344 features)")

    def report(tag, yprob):
        auroc = metrics.roc_auc_score(yte, yprob)
        auprc = metrics.average_precision_score(yte, yprob)
        acc   = metrics.accuracy_score(yte, (yprob >= 0.5).astype(int))
        mcc   = metrics.matthews_corrcoef(yte, (yprob >= 0.5).astype(int))
        print(f"[{tag}] AUROC={auroc:.6f}  AUPRC={auprc:.4f}  ACC={acc:.4f}  MCC={mcc:.4f}  (reported AUROC={REPORTED_AUROC:.4f})")
        return auroc

    # ---- (A) shipped trained model, re-scored
    aurocA = None
    try:
        import joblib
        sys.modules.setdefault("sklearn.externals.joblib", joblib)  # shim for old pickles
        model = joblib.load(MODEL)
        print(f"[A] loaded shipped model: {type(model).__name__}")
        # align columns to the model's expected feature names if present
        fn = getattr(model, "feature_name_", None) or getattr(getattr(model, "booster_", None), "feature_name", lambda: None)()
        Xte_A = Xte[fn] if (fn is not None and list(fn) != list(Xte.columns) and set(fn) == set(Xte.columns)) else Xte
        yprobA = model.predict_proba(Xte_A.values)[:, 1]
        aurocA = report("A shipped-model", yprobA)
        try:
            print("[A] model params:", {k: model.get_params()[k] for k in ("learning_rate","num_leaves","max_depth","n_estimators","reg_alpha","reg_lambda") if k in model.get_params()})
        except Exception:
            pass
    except Exception as e:
        print(f"[A] could not load/score shipped model: {type(e).__name__}: {e}")

    # ---- (B) retrain fresh LGBMClassifier
    try:
        import lightgbm as lgb
        params = {}
        if aurocA is not None:
            try: params = model.get_params()
            except Exception: params = {}
        # keep only meaningful hyperparams; fall back to notebook-typical values
        keep = {k: params[k] for k in ("learning_rate","num_leaves","max_depth","n_estimators","reg_alpha","reg_lambda","min_child_samples","subsample","colsample_bytree") if k in params and params[k] is not None}
        clf = lgb.LGBMClassifier(objective="binary", random_state=0, n_jobs=-1, verbose=-1, **keep)
        clf.fit(Xtr.values, ytr)
        yprobB = clf.predict_proba(Xte.values)[:, 1]
        print(f"[B] retrained with params: {keep if keep else '(defaults)'}")
        report("B retrained", yprobB)
    except Exception as e:
        print(f"[B] retrain failed: {type(e).__name__}: {e}")

    # ---- verdict
    if aurocA is not None:
        delta = abs(aurocA - REPORTED_AUROC)
        verdict = "PASS" if delta < 0.01 else ("CLOSE" if delta < 0.03 else "MISMATCH")
        print(f"\n[GATE] shipped-model AUROC={aurocA:.6f} vs reported {REPORTED_AUROC:.6f} | |delta|={delta:.6f} -> {verdict}")
    else:
        print("\n[GATE] shipped-model path unavailable; rely on retrain (B) result above.")

if __name__ == "__main__":
    main()

#!/bin/bash
# R2 computations, strictly one at a time, lowest priority (CLAUDE.md section 10).
# Each step refuses to start while the 1-minute load average exceeds 8 (the logical core count).
# Usage: ./run_queue_r2.sh [step ...]   (no argument = every step, in the order below)
cd "$(dirname "$0")" || exit 1
PY=./venv/bin/python
OUT=../revision_r2/results
mkdir -p "$OUT"

wait_for_load() {
  while :; do
    l=$(sysctl -n vm.loadavg | awk '{print $2}')
    awk -v l="$l" 'BEGIN{exit !(l < 8)}' && return
    echo "  load $l >= 8: waiting" ; sleep 60
  done
}

step() {  # step <name> <command...>
  local name="$1"; shift
  wait_for_load
  echo "== $(date '+%H:%M:%S') START $name"
  nice -n 19 "$@" > "lncptpred_r2_${name}.log" 2>&1
  local rc=$?
  echo "== $(date '+%H:%M:%S') END   $name rc=$rc"
  return $rc
}

run() {
  case "$1" in
    # DVMnet re-run with training negatives drawn from the training graph only (R2), and the
    # R2 'swap' negatives (the positives' own endpoints re-paired); evaluation sets of the R1
    # conditions are regenerated identically, so the R1 and R2 runs differ only in training
    dvmnet_r2_main)  step "$1" $PY -u exp_dvmnet_revision.py --train-scope train --threads 4 --repeats 5 \
                       --conditions random-swap,cold-swap,random-degmat,cold-degmat,cold-uniform \
                       --out "$OUT/dvmnet_r2.json" ;;
    dvmnet_r2_sens)  step "$1" $PY -u exp_dvmnet_revision.py --train-scope train --threads 4 --repeats 3 \
                       --conditions cold-alpha0.5,cold-joint,cold-nohub,random-joint,random-nohub \
                       --out "$OUT/dvmnet_r2.json" ;;
    dvmnet_baselines)   step "$1" $PY -u exp_dvmnet_baselines.py --r2 "$OUT/dvmnet_r2.json" \
                          --out "$OUT/dvmnet_baselines.json" ;;
    rnalight_baselines) step "$1" $PY -u exp_rnalight_baselines.py --out "$OUT/rnalight_baselines.json" ;;
    score_released)     step "$1" $PY -u exp_lncptpred_faithful.py score-released --out "$OUT/lncptpred_score_released.json" ;;
    retrain_original)   step "$1" $PY -u exp_lncptpred_faithful.py retrain --out "$OUT/lncptpred_retrain_original.json" ;;
    r1_settings)        step "$1" $PY -u exp_lncptpred_faithful.py r1-settings --out "$OUT/lncptpred_r1_settings_full.json" ;;
    sub400k)            step "$1" $PY -u exp_lncptpred_faithful.py sub400k --out "$OUT/lncptpred_original_n400k.json" ;;
    *) echo "unknown step $1"; return 2 ;;
  esac
}

STEPS=("$@")
[ ${#STEPS[@]} -eq 0 ] && STEPS=(dvmnet_r2_main score_released retrain_original dvmnet_r2_sens
                                 dvmnet_baselines rnalight_baselines r1_settings sub400k)
for s in "${STEPS[@]}"; do
  run "$s" || exit 1
done
echo "== $(date '+%H:%M:%S') QUEUE DONE"

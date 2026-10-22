#!/bin/bash
# Run the remaining revision experiments strictly one at a time (CLAUDE.md section 10).
cd "$(dirname "$0")"
PY=./venv/bin/python

# 1) wait for Stage A (already running) to exit
while pgrep -qf "exp_dvmnet_revision.py --conditions random-verbatim"; do sleep 60; done
echo "[queue] Stage A おわり $(date +%H:%M:%S)"

# 2) Stage B -- negative-sampling sensitivity (Reviewer 2 point 1)
nice -n 19 $PY exp_dvmnet_revision.py \
  --conditions cold-alpha0.5,cold-joint,cold-nohub,random-joint,random-nohub \
  --repeats 3 --out dvmnet_sensitivity_results.json > dvmnet_sensitivity_run.log 2>&1
echo "[queue] Stage B おわり $(date +%H:%M:%S)"

# 3) LncPTPred rebuilt from the authors' released feature code (Reviewer 2 point 4)
nice -n 19 $PY exp_lncptpred_exact.py > lncptpred_exact_run.log 2>&1
echo "[queue] LncPTPred おわり $(date +%H:%M:%S)"

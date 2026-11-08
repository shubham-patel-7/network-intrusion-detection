#!/usr/bin/env bash
# Full pipeline: tuning -> final fits and cost measurements -> ablations -> report.
# Every stage saves after each fit, so re-running this script resumes where it stopped.
#
# Tuning and the ablations do not report timings, so they may use more CPU threads (TUNE_THREADS).
# The final stage measures cost, so it uses IDS_THREADS=2 like the report (2 CPU cores) and runs alone.
#
# PARALLEL_TUNE=1 tunes in two streams at once (DT, RF, SVM and MLP in one; LSTM in the other), each with
# TUNE_THREADS threads. It is faster on a machine with spare cores and about 1.5 GB of RAM per deep-learning process.
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
TUNE_THREADS="${TUNE_THREADS:-8}"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
# settings of the reported run (override on the command line): 30,000 flows per attack type, 8 random configurations, 3 seeds
export IDS_CICIOT_CAP="${IDS_CICIOT_CAP:-30000}" IDS_N="${IDS_N:-8}" IDS_SEEDS="${IDS_SEEDS:-0,1,2}"

echo "== start $(date)"
# build the cached data once (for CICIoT2023 this is the slow sampling pass) so parallel streams do not repeat it
IDS_THREADS="$TUNE_THREADS" $PY -c "from idsbench import common as c; c.prepare()"
echo "== data ready $(date)"
if [ "${PARALLEL_TUNE:-0}" = "1" ]; then
  (IDS_THREADS="$TUNE_THREADS" $PY -m idsbench.tune DT RF SVM MLP) &
  (IDS_THREADS="$TUNE_THREADS" $PY -m idsbench.tune LSTM) &
  wait
else
  IDS_THREADS="$TUNE_THREADS" $PY -m idsbench.tune DT RF SVM MLP LSTM
fi
echo "== tuning done $(date)"
IDS_THREADS=2 $PY -m idsbench.final
echo "== final fits done $(date)"
IDS_THREADS="$TUNE_THREADS" $PY -m idsbench.ablate task imbalance features size
echo "== ablations done $(date)"
$PY -m idsbench.report
echo "== all done $(date)"

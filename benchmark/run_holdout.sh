#!/usr/bin/env bash
# Held-out attack-type experiment (CICIoT2023): the same five models, with 7 attack types removed from training so
# that they appear only in the test set. Run it AFTER run_all.sh has finished tuning: it reuses the configurations chosen
# there (results/best_*.json), so no retuning is needed, then refits every model (final.py) and reports.
#
# Held-out types (one or two per category, each with at least ~3,000 flows in the 30,000-per-class sample, and every
# category keeps at least one type in training; Brute Force has a single type and cannot be held out):
#   DDoS DDOS-PSHACK_FLOOD | DoS DOS-SYN_FLOOD | Mirai MIRAI-GREIP_FLOOD | Recon RECON-HOSTDISCOVERY
#   Spoofing DNS_SPOOFING | Web-based XSS and SQLINJECTION
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export IDS_CICIOT_CAP="${IDS_CICIOT_CAP:-30000}"
export IDS_SEEDS="${IDS_SEEDS:-0}"
export IDS_CICIOT_HOLDOUT="DDOS-PSHACK_FLOOD,DOS-SYN_FLOOD,MIRAI-GREIP_FLOOD,RECON-HOSTDISCOVERY,DNS_SPOOFING,XSS,SQLINJECTION"
export IDS_RES=results_holdout IDS_FIGS=figures_holdout
export IDS_THREADS=2   # cost is measured here too, so keep the thread count of the main final stage
mkdir -p "$IDS_RES"
cp results/best_*.json "$IDS_RES"/
echo "== holdout start $(date)"
$PY -m idsbench.final
echo "== holdout final fits done $(date)"
$PY -m idsbench.report
echo "== holdout all done $(date)"

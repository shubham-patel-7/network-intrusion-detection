# ML vs DL intrusion-detection benchmark (CICIoT2023)

A controlled comparison of three classical models (Decision Tree, Random Forest, RBF SVM) and two deep models (MLP, LSTM) on the
CICIoT2023 IoT intrusion dataset. Every model gets the same data, split, preprocessing, tuning budget and metric code; the test split
is used once, after all choices are frozen. The results and a summary of the findings are in the [repository README](../README.md).

## Protocol

1. **Sample.** The 63 `MERGED_CSV` files (45,019,234 flows, 39 numeric features) are streamed twice: first to count the labels, then to
   draw at most `IDS_CICIOT_CAP` (default 30,000) flows per attack type. The 34 labels map to 8 classes (Benign, DDoS, DoS, Mirai,
   Recon, Spoofing, Web-based, Brute Force; see `src/idsbench/ciciot.py`). Exact duplicates are dropped.
2. **Split.** Stratified 80/20 into training and test; the training part is split 80/20 again into training and validation. Min-max
   scaling is fitted on the training part only.
3. **Tuning.** Random search with the same budget for every model (`IDS_N`, default 8 configurations), scored by validation macro-F1.
   One configuration per model is frozen.
4. **Final fits.** Each frozen model is refitted on the whole training split for `IDS_SEEDS` (default 0,1,2) and tested once. Training
   time, batch time, per-record latency and model size are measured on 2 CPU threads.
5. **Statistics.** 1,000-resample bootstrap intervals for macro-F1, McNemar tests with Holm correction, and a practical-tie margin of
   0.01 macro-F1 that is fixed in advance.
6. **Ablations** (one seed, frozen settings): binary task, imbalance handling (none / class weights / SMOTE), feature groups and
   top-k mutual information, and training-set size.
7. **Held-out attack types** (`run_holdout.sh`): seven attack types are removed from training and appear only in the test split.

The class weights, SVM subsample (at most 30,000 records, all rare-class records kept) and the search spaces are in `src/idsbench/common.py`.

## Run it

```bash
pip install -r requirements.txt && pip install -e .
# download CICIoT2023 (MERGED_CSV) from https://www.unb.ca/cic/datasets/iotdataset-2023.html (registration form)
# and unzip it under data/ciciot2023/  (or set IDS_CICIOT_DIR)
bash run_all.sh          # tuning -> final fits and costs -> ablations -> tables and figures (hours on a laptop)
bash run_holdout.sh      # held-out attack types, after run_all.sh has finished tuning
pytest -q                # unit tests; the tests marked `data` need the CSV files
python run_model.py      # demo: train a Decision Tree, test it, classify 10 flows
```

`run_all.sh` saves after every fit, so running it again resumes where it stopped. Environment variables: `IDS_CICIOT_CAP`,
`IDS_N`, `IDS_SEEDS`, `IDS_THREADS`, `IDS_MODELS` (restrict the ablations), `IDS_RES` / `IDS_FIGS` (output folders),
`KERAS_BACKEND` (`tensorflow` or `torch`; torch is used when TensorFlow is not installed).

## Layout

| Path | Content |
|---|---|
| `src/idsbench/ciciot.py` | CICIoT2023 loader: streamed sampling, label map, split, scaling, held-out types |
| `src/idsbench/common.py` | Models, metrics, class weights, random-search spaces, cost timing |
| `src/idsbench/tune.py`, `final.py`, `ablate.py`, `report.py` | The pipeline stages |
| `src/idsbench/runner.py`, `run_model.py` | Train, save and use a single model |
| `scripts/export_for_app.py` | Export the five models for the Streamlit app |
| `results/` | `summary.json` and `results_tables.md` (all numbers), tuning and per-seed results, ablations |
| `results_holdout/` | The same for the held-out attack-type run |
| `figures/`, `figures_holdout/` | The figures made by `report.py` |
| `tests/` | Unit tests |

Per-seed prediction files (`preds_*.npz`) are not committed; `final.py` recreates them.

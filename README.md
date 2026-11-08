# Machine Learning vs Deep Learning for IoT Intrusion Detection (CICIoT2023)

> **Can deep learning beat classical machine learning at spotting IoT attacks?** A controlled comparison of five classifiers
> (Decision Tree, Random Forest, SVM, MLP, LSTM) on the CICIoT2023 dataset: same data, same split, same tuning budget, results
> reported per class, on attack types the models never saw, and with the computational cost next to the accuracy.

**[Live demo](https://nsl-kdd-ids-demo.streamlit.app/)** (free hosting: if the app has been idle it takes about 30 seconds to wake up)

Project of the Information Security Lab at MIT Manipal (Manipal Academy of Higher Education), by Shubham Patel, Vinamra Lilariaa,
Tejas Baid and Anushka Chitranshi, under the guidance of Sudhina Kumar GK.

## What is in this repository

| Folder | What it is |
|---|---|
| `app/` | The Streamlit app: study results (Overview, Models, Results, Ablations, About) and a **Detector** that classifies a CSV of flows with any of the five models |
| `models/study/` | The five trained models as plain scikit-learn / Keras files, with their preprocessing |
| `sample_data/` | 496 flows from the held-out test split, to try the Detector |
| `benchmark/` | The benchmark code, results and figures (tuning, final fits, cost timing, ablations, held-out attack types) |
| `tests/` | Tests of the app's data and detector code |

## Data

[CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) (Neto et al., *Sensors* 2023, Canadian Institute for
Cybersecurity): IoT traffic from a testbed of 105 devices, with 33 attack types in 7 categories plus benign traffic. The `MERGED_CSV`
download (63 files, 45,019,234 flows, 39 numeric features) was used. The files need a short registration form and are **not
included here**.

The 34 labels are grouped into 8 classes: Benign, DDoS, DoS, Mirai, Recon, Spoofing, Web-based and Brute Force. To make the study
affordable on a laptop, at most 30,000 flows of each attack type were sampled (216,866 flows, 212,196 after removing exact duplicates),
split 80/20 (stratified) into 169,756 training and 42,440 test flows, with 20% of the training part used for validation.

## Results

Held-out test split of 42,440 flows, mean ± standard deviation over 3 seeds (8 random-search configurations per model).

| Model | Accuracy | Macro-F1 | Attack detection | False alarms on benign | Training time | Latency / flow | Model size |
|---|---|---|---|---|---|---|---|
| Decision Tree | 0.742 | 0.723 ± 0.000 | 94.8% | 32.9% | 2.5 s | 0.12 ms | 0.3 MB |
| **Random Forest** | **0.775** | **0.759 ± 0.000** | 95.0% | **24.4%** | 27.4 s | 12.39 ms | 545 MB |
| SVM (RBF) | 0.655 | 0.637 ± 0.004 | 90.8% | 33.2% | 113.4 s | 2.71 ms | 10.7 MB |
| MLP | 0.739 | 0.722 ± 0.005 | 94.3% | 32.0% | 263.7 s | 1.57 ms | 0.4 MB |
| LSTM | 0.739 | 0.722 ± 0.003 | 93.8% | 30.0% | 476.8 s | 1.97 ms | 0.5 MB |

1. **The Random Forest wins, deep learning does not.** It is the most accurate (macro-F1 0.759, 95% interval 0.754 to 0.762) and
   better than every other model by more than the practical margin of 0.01. The Decision Tree, MLP and LSTM are tied at about 0.72 and
   the SVM is last (0.637).
2. **Deep learning costs the most and gains nothing here.** The MLP and LSTM need 264 to 477 s of CPU training against 2.5 to 113 s
   for the classical models. The forest trains in 27 s but is a 545 MB model with the slowest prediction per flow; the Decision Tree is
   the cheapest model by a wide margin.
3. **All models raise many false alarms.** Between 24% and 33% of benign flows are flagged as attacks, and benign recall is only
   0.67 to 0.76, so none of the five is ready to run unattended.
4. **Unseen attack types are noticed but not named.** In a second run with seven attack types removed from training, 93% to 97% of
   the unseen flows were flagged as attacks, but only 48% to 53% received the right category. One held-out DDoS type was labelled
   correctly by the MLP (98%) and almost never by the Random Forest (0%); the held-out DoS type was missed by every model.
5. **Ablations** (single seed, Decision Tree / Random Forest / MLP): removing the packet-size statistics hurts most (Random Forest
   0.758 to 0.681, MLP 0.725 to 0.660); SMOTE oversampling does not help; more training data helps the MLP most (0.668 with 10% of the
   sample, 0.725 with all of it).

### Limitations

- The split is random, so flows from the same capture can sit on both sides; scores would probably be lower on a different testbed.
- The classes are balanced by capping each attack type, so the class mix differs from real traffic.
- Reduced budgets because of the data size: 8 random configurations per model and 3 seeds; the ablations use one seed and cover
  the Decision Tree, Random Forest and MLP (the SVM only for the binary task); the held-out run uses one seed per model.
- Timings come from one CPU-only machine (2 threads) and the neural latency includes framework overhead; no GPU was measured.
- The Detector ships the first 40 of the Random Forest's 146 trees (full forest: 545 MB in memory); on the test split they score
  macro-F1 0.756 against 0.758 for the full forest. The LSTM sees one time step per flow, so it is a gated feed-forward layer.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
pytest -q
```

The benchmark has its own instructions in [`benchmark/README.md`](benchmark/README.md). Python 3.10 or newer; Keras 3 runs the MLP
and LSTM on the PyTorch backend (`KERAS_BACKEND=torch`, set by the app).

## Licence

Code: see [LICENSE](LICENSE). The CICIoT2023 dataset belongs to its authors and is not redistributed here; the 496-flow sample in
`sample_data/` is a small excerpt for demonstration.

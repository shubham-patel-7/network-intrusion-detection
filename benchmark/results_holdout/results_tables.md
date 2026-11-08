### Detection quality on the held-out test split (mean ± standard deviation over seeds)

| Model | Accuracy | Macro-F1 | Attack detection rate | False-positive rate (Normal) | ROC-AUC (macro) | ROC-AUC (Normal vs Attack) |
|---|---|---|---|---|---|---|
| Decision Tree | 0.621 ± 0.000 | 0.586 ± 0.000 | 95.8% ± 0.0 | 32.9% ± 0.0 | 0.881 ± 0.000 | 0.943 ± 0.000 |
| Random Forest | 0.641 ± 0.000 | 0.608 ± 0.000 | 95.3% ± 0.0 | 24.6% ± 0.0 | 0.940 ± 0.000 | 0.961 ± 0.000 |
| SVM (RBF) | 0.578 ± 0.000 | 0.548 ± 0.000 | 92.5% ± 0.0 | 36.7% ± 0.0 | 0.875 ± 0.000 | 0.908 ± 0.000 |
| MLP | 0.619 ± 0.000 | 0.586 ± 0.000 | 94.2% ± 0.0 | 30.9% ± 0.0 | 0.891 ± 0.000 | 0.938 ± 0.000 |
| LSTM | 0.608 ± 0.000 | 0.575 ± 0.000 | 94.0% ± 0.0 | 32.6% ± 0.0 | 0.858 ± 0.000 | 0.935 ± 0.000 |

### Recall / F1 per class (mean over seeds)

| Model | Benign | DDoS | DoS | Mirai | Recon | Spoofing | Web-based | Brute Force |
|---|---|---|---|---|---|---|---|---|
| Decision Tree | 0.67 / 0.62 | 0.46 / 0.40 | 0.37 / 0.42 | 0.99 / 1.00 | 0.56 / 0.55 | 0.71 / 0.81 | 0.49 / 0.52 | 0.63 / 0.38 |
| Random Forest | 0.75 / 0.65 | 0.44 / 0.39 | 0.36 / 0.40 | 0.99 / 1.00 | 0.63 / 0.58 | 0.76 / 0.85 | 0.48 / 0.54 | 0.62 / 0.47 |
| SVM (RBF) | 0.63 / 0.50 | 0.72 / 0.57 | 0.39 / 0.48 | 0.99 / 1.00 | 0.57 / 0.52 | 0.44 / 0.54 | 0.38 / 0.39 | 0.47 / 0.38 |
| MLP | 0.69 / 0.58 | 0.85 / 0.63 | 0.36 / 0.48 | 1.00 / 1.00 | 0.59 / 0.55 | 0.48 / 0.60 | 0.47 / 0.49 | 0.59 / 0.36 |
| LSTM | 0.67 / 0.56 | 0.82 / 0.62 | 0.37 / 0.49 | 1.00 / 0.99 | 0.59 / 0.55 | 0.45 / 0.58 | 0.43 / 0.47 | 0.63 / 0.34 |

### Held-out (unseen) attack types

| Model | Detection rate | Correct category | Recall per category (DDoS / DoS / Mirai / Recon / Spoofing / Web-based / Brute Force) |
|---|---|---|---|
| Decision Tree | 96.7% ± 0.0 | 51.7% ± 0.0 | 0.03 / 0.04 / 0.99 / 0.55 / 0.67 / 0.43 / - |
| Random Forest | 95.9% ± 0.0 | 52.6% ± 0.0 | 0.00 / 0.01 / 0.99 / 0.58 / 0.74 / 0.40 / - |
| SVM (RBF) | 92.6% ± 0.0 | 48.5% ± 0.0 | 0.79 / 0.03 / 0.99 / 0.56 / 0.37 / 0.32 / - |
| MLP | 94.5% ± 0.0 | 51.7% ± 0.0 | 0.98 / 0.00 / 0.99 / 0.56 / 0.40 / 0.40 / - |
| LSTM | 94.2% ± 0.0 | 50.3% ± 0.0 | 0.95 / 0.01 / 1.00 / 0.56 / 0.37 / 0.37 / - |

### Computational cost (CPU, mean ± standard deviation)

| Model | Training time (s) | Epochs | Batch inference, whole test set (s) | Latency per record, median (p95) ms | Model size (MB) | Size counts |
|---|---|---|---|---|---|---|
| Decision Tree | 8.6 ± 0.0 | - | 0.044 ± 0.000 | 0.57 (1.53) | 0.33 | 2,533 nodes |
| Random Forest | 72.6 ± 0.0 | - | 4.534 ± 0.000 | 49.68 (71.01) | 433.90 | 146 trees, 3,380,934 nodes |
| SVM (RBF) | 252.7 ± 0.0 | - | 679.613 ± 0.000 | 4.91 (5.49) | 9.22 | 21,541 support vectors |
| MLP | 334.6 ± 0.0 | 59 | 0.225 ± 0.000 | 1.88 (2.65) | 0.40 | 30,896 parameters |
| LSTM | 627.0 ± 0.0 | 57 | 0.558 ± 0.000 | 3.96 (5.75) | 0.53 | 41,044 parameters |

### 95% bootstrap intervals for macro-F1 (seed 0)

| Model | Interval |
|---|---|
| Decision Tree | 0.583 to 0.590 |
| Random Forest | 0.604 to 0.612 |
| SVM (RBF) | 0.545 to 0.552 |
| MLP | 0.582 to 0.589 |
| LSTM | 0.572 to 0.579 |

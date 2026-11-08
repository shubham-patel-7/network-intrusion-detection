### Detection quality on the held-out test split (mean ± standard deviation over seeds)

| Model | Accuracy | Macro-F1 | Attack detection rate | False-positive rate (Normal) | ROC-AUC (macro) | ROC-AUC (Normal vs Attack) |
|---|---|---|---|---|---|---|
| Decision Tree | 0.742 ± 0.000 | 0.723 ± 0.000 | 94.8% ± 0.0 | 32.9% ± 0.0 | 0.956 ± 0.000 | 0.943 ± 0.000 |
| Random Forest | 0.775 ± 0.000 | 0.759 ± 0.000 | 95.0% ± 0.0 | 24.4% ± 0.2 | 0.968 ± 0.000 | 0.960 ± 0.000 |
| SVM (RBF) | 0.655 ± 0.002 | 0.637 ± 0.004 | 90.8% ± 0.6 | 33.2% ± 3.6 | 0.908 ± 0.002 | 0.900 ± 0.006 |
| MLP | 0.739 ± 0.005 | 0.722 ± 0.005 | 94.3% ± 0.5 | 32.0% ± 1.0 | 0.959 ± 0.001 | 0.943 ± 0.001 |
| LSTM | 0.739 ± 0.003 | 0.722 ± 0.003 | 93.8% ± 0.4 | 30.0% ± 2.1 | 0.960 ± 0.001 | 0.943 ± 0.001 |

### Recall / F1 per class (mean over seeds)

| Model | Benign | DDoS | DoS | Mirai | Recon | Spoofing | Web-based | Brute Force |
|---|---|---|---|---|---|---|---|---|
| Decision Tree | 0.67 / 0.68 | 0.67 / 0.76 | 0.91 / 0.81 | 1.00 / 1.00 | 0.57 / 0.62 | 0.84 / 0.87 | 0.57 / 0.56 | 0.63 / 0.50 |
| Random Forest | 0.76 / 0.73 | 0.68 / 0.77 | 0.91 / 0.81 | 1.00 / 1.00 | 0.65 / 0.67 | 0.85 / 0.89 | 0.64 / 0.62 | 0.60 / 0.59 |
| SVM (RBF) | 0.67 / 0.60 | 0.62 / 0.70 | 0.86 / 0.76 | 0.99 / 0.99 | 0.49 / 0.54 | 0.52 / 0.57 | 0.49 / 0.44 | 0.45 / 0.50 |
| MLP | 0.68 / 0.67 | 0.64 / 0.75 | 0.92 / 0.81 | 1.00 / 1.00 | 0.60 / 0.64 | 0.78 / 0.80 | 0.58 / 0.55 | 0.59 / 0.56 |
| LSTM | 0.70 / 0.68 | 0.62 / 0.74 | 0.95 / 0.81 | 1.00 / 1.00 | 0.58 / 0.64 | 0.77 / 0.81 | 0.61 / 0.55 | 0.58 / 0.55 |

### Computational cost (CPU, mean ± standard deviation)

| Model | Training time (s) | Epochs | Batch inference, whole test set (s) | Latency per record, median (p95) ms | Model size (MB) | Size counts |
|---|---|---|---|---|---|---|
| Decision Tree | 2.5 ± 0.1 | - | 0.007 ± 0.001 | 0.12 (0.16) | 0.33 | 2,575 nodes |
| Random Forest | 27.4 ± 0.4 | - | 1.031 ± 0.285 | 12.39 (15.03) | 545.14 | 146 trees, 4,252,992 nodes |
| SVM (RBF) | 113.4 ± 3.8 | - | 106.450 ± 1.008 | 2.71 (2.84) | 10.72 | 25,068 support vectors |
| MLP | 263.7 ± 6.7 | 59 | 0.085 ± 0.002 | 1.57 (2.49) | 0.40 | 30,896 parameters |
| LSTM | 476.8 ± 2.9 | 57 | 0.157 ± 0.005 | 1.97 (3.11) | 0.53 | 41,044 parameters |

### 95% bootstrap intervals for macro-F1 (seed 0)

| Model | Interval |
|---|---|
| Decision Tree | 0.719 to 0.727 |
| Random Forest | 0.754 to 0.762 |
| SVM (RBF) | 0.629 to 0.638 |
| MLP | 0.721 to 0.729 |
| LSTM | 0.718 to 0.727 |

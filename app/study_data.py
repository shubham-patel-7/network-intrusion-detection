"""Results of the lab report "Deep Learning vs. Machine Learning for Network Intrusion Detection: A Comparative
Study on NSL-KDD" (Information Security Lab, MIT Manipal), transcribed from its tables.

All numbers come from the report's own run (2 CPU cores, scikit-learn 1.9.1, TensorFlow 2.21, five seeds for the
main results, one seed for the ablations). They are not recomputed by this app.
"""

MODELS = ["Decision Tree", "Random Forest", "SVM (RBF)", "MLP", "LSTM"]
FAMILY = {"Decision Tree": "Classical", "Random Forest": "Classical", "SVM (RBF)": "Classical",
          "MLP": "Deep learning", "LSTM": "Deep learning"}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
# Colours that read on a dark background; blue/indigo = classical, amber/orange = deep learning.
COLORS = {"Decision Tree": "#93c5fd", "Random Forest": "#3b82f6", "SVM (RBF)": "#818cf8",
          "MLP": "#fbbf24", "LSTM": "#f97316"}

# Table II: five-class distribution (records).
DISTRIBUTION = {"Normal": (67343, 9711), "DoS": (45927, 7460), "Probe": (11656, 2421),
                "R2L": (995, 2885), "U2R": (52, 67)}

# Table IX: frozen configuration and validation macro-F1.
CONFIG = {
    "Decision Tree": ("depth 15, min leaf 5, entropy", 0.908),
    "Random Forest": ("287 trees, unlimited depth, sqrt features, min leaf 1", 0.953),
    "SVM (RBF)": ("C = 17.3, gamma = 0.857, 29,999-record subsample", 0.858),
    "MLP": ("165 and 82 units, dropout 0.06, learning rate 0.0012, batch 256, 44 epochs", 0.890),
    "LSTM": ("83 units then 41, dropout 0.01, learning rate 0.0064, batch 512, 39 epochs", 0.891),
}

# Table XI: KDDTest+ (22,544 records), mean and standard deviation over five seeds. Rates are in percent.
MAIN = {
    "Decision Tree": dict(acc=(0.751, 0.003), mf1=(0.530, 0.002), det=(69.6, 0.4), fpr=(3.4, 0.1), auc=(0.736, 0.001), aucb=(0.836, 0.002)),
    "Random Forest": dict(acc=(0.743, 0.001), mf1=(0.510, 0.004), det=(58.9, 0.2), fpr=(2.8, 0.0), auc=(0.930, 0.004), aucb=(0.962, 0.001)),
    "SVM (RBF)": dict(acc=(0.761, 0.006), mf1=(0.612, 0.003), det=(65.0, 1.1), fpr=(3.4, 0.4), auc=(0.900, 0.005), aucb=(0.948, 0.003)),
    "MLP": dict(acc=(0.801, 0.010), mf1=(0.631, 0.014), det=(73.6, 2.6), fpr=(3.4, 0.1), auc=(0.916, 0.012), aucb=(0.925, 0.003)),
    "LSTM": dict(acc=(0.801, 0.011), mf1=(0.641, 0.027), det=(71.7, 2.7), fpr=(3.7, 0.2), auc=(0.916, 0.014), aucb=(0.929, 0.018)),
}

# Table XII: (recall, F1) per class, mean over five seeds.
PER_CLASS = {
    "Decision Tree": {"Normal": (0.97, 0.82), "DoS": (0.74, 0.84), "Probe": (0.73, 0.64), "R2L": (0.08, 0.15), "U2R": (0.15, 0.21)},
    "Random Forest": {"Normal": (0.97, 0.77), "DoS": (0.76, 0.85), "Probe": (0.63, 0.72), "R2L": (0.02, 0.05), "U2R": (0.09, 0.16)},
    "SVM (RBF)": {"Normal": (0.97, 0.80), "DoS": (0.78, 0.86), "Probe": (0.62, 0.70), "R2L": (0.16, 0.26), "U2R": (0.39, 0.45)},
    "MLP": {"Normal": (0.97, 0.83), "DoS": (0.83, 0.89), "Probe": (0.71, 0.76), "R2L": (0.24, 0.35), "U2R": (0.48, 0.31)},
    "LSTM": {"Normal": (0.96, 0.82), "DoS": (0.83, 0.89), "Probe": (0.75, 0.78), "R2L": (0.23, 0.37), "U2R": (0.48, 0.34)},
}

# 95% bootstrap interval for macro-F1 (1,000 resamples of the test records, seed 0).
CI = {"Decision Tree": (0.506, 0.553), "Random Forest": (0.493, 0.539), "SVM (RBF)": (0.589, 0.637),
      "MLP": (0.630, 0.668), "LSTM": (0.647, 0.691)}

# Macro-F1 of the seed-0 run, the run the intervals above and the ablations refer to (Table XVII, class weights).
SEED0_MF1 = {"Decision Tree": 0.531, "Random Forest": 0.516, "SVM (RBF)": 0.613, "MLP": 0.649, "LSTM": 0.670}

# Table XIII: harder KDDTest-21 (11,850 records) and the 3,750 records of the 17 attack types absent from training.
HARD = {
    "Decision Tree": dict(t21_acc=(0.544, 0.006), t21_mf1=(0.442, 0.002), t21_det=(60.1, 0.5), u_det=(52.7, 0.4), u_acc=(25.4, 1.5), u_rec=dict(DoS=0.11, Probe=0.57, R2L=0.01, U2R=0.17)),
    "Random Forest": dict(t21_acc=(0.513, 0.002), t21_mf1=(0.412, 0.005), t21_det=(45.6, 0.3), u_det=(17.2, 0.6), u_acc=(12.1, 0.6), u_rec=dict(DoS=0.02, Probe=0.32, R2L=0.00, U2R=0.11)),
    "SVM (RBF)": dict(t21_acc=(0.552, 0.011), t21_mf1=(0.520, 0.004), t21_det=(53.7, 1.5), u_det=(30.6, 3.6), u_acc=(18.3, 3.4), u_rec=dict(DoS=0.10, Probe=0.38, R2L=0.00, U2R=0.36)),
    "MLP": dict(t21_acc=(0.623, 0.019), t21_mf1=(0.547, 0.015), t21_det=(65.1, 3.5), u_det=(48.5, 5.0), u_acc=(31.2, 5.8), u_rec=dict(DoS=0.30, Probe=0.49, R2L=0.01, U2R=0.45)),
    "LSTM": dict(t21_acc=(0.623, 0.020), t21_mf1=(0.555, 0.029), t21_det=(62.5, 3.5), u_det=(43.5, 7.5), u_acc=(33.4, 6.2), u_rec=dict(DoS=0.29, Probe=0.56, R2L=0.01, U2R=0.46)),
}
UNSEEN_COUNTS = {"DoS": 1719, "Probe": 1315, "R2L": 686, "U2R": 30}

# Table XIV: cost on 2 CPU cores, mean and standard deviation over five runs.
COST = {
    "Decision Tree": dict(train=(0.9, 0.0), epochs=None, batch=(0.0036, 0.0002), lat=0.10, lat95=0.15, size=0.06, counts="573 nodes"),
    "Random Forest": dict(train=(11.3, 0.2), epochs=None, batch=(0.28, 0.02), lat=17.94, lat95=25.41, size=38.03, counts="287 trees, 353,881 nodes"),
    "SVM (RBF)": dict(train=(6.8, 0.3), epochs=None, batch=(3.97, 0.11), lat=0.34, lat95=0.46, size=1.79, counts="1,726 support vectors"),
    "MLP": dict(train=(70.2, 2.2), epochs=44, batch=(0.13, 0.01), lat=2.85, lat95=4.06, size=0.44, counts="34,157 parameters"),
    "LSTM": dict(train=(78.7, 1.6), epochs=39, batch=(0.31, 0.07), lat=8.73, lat95=12.18, size=0.90, counts="71,714 parameters"),
}

# Table XVI: binary task (Normal vs Attack), one seed, threshold 0.5.
BINARY = {
    "Decision Tree": dict(acc=0.793, f1=0.784, det=65.9, fpr=3.0, auc=0.835, unseen=36.4),
    "Random Forest": dict(acc=0.777, f1=0.762, det=62.8, fpr=2.7, auc=0.961, unseen=28.3),
    "SVM (RBF)": dict(acc=0.827, f1=0.825, det=71.8, fpr=3.0, auc=0.926, unseen=58.8),
    "MLP": dict(acc=0.815, f1=0.818, det=73.0, fpr=7.3, auc=0.886, unseen=47.4),
    "LSTM": dict(acc=0.827, f1=0.828, det=73.3, fpr=4.9, auc=0.921, unseen=46.3),
}

# Table XVII: macro-F1 under three imbalance strategies, with R2L and U2R F1: (macro-F1, R2L F1, U2R F1).
IMBALANCE = {
    "Decision Tree": {"No correction": (0.545, 0.16, 0.21), "Class weights": (0.531, 0.14, 0.22), "SMOTE": (0.554, 0.04, 0.41)},
    "Random Forest": {"No correction": (0.498, 0.08, 0.08), "Class weights": (0.516, 0.05, 0.18), "SMOTE": (0.538, 0.11, 0.24)},
    "SVM (RBF)": {"No correction": (0.605, 0.18, 0.48), "Class weights": (0.613, 0.27, 0.44), "SMOTE": (0.621, 0.26, 0.48)},
    "MLP": {"No correction": (0.674, 0.36, 0.55), "Class weights": (0.649, 0.37, 0.34), "SMOTE": (0.666, 0.33, 0.55)},
    "LSTM": {"No correction": (0.662, 0.36, 0.48), "Class weights": (0.670, 0.39, 0.46), "SMOTE": (0.655, 0.36, 0.51)},
}

# Table XVIII: feature ablation: (inputs, macro-F1, accuracy, unseen-type detection %). Random Forest and LSTM only.
FEATURES = {
    "Random Forest": {
        "All features": (121, 0.516, 0.743, 18.1), "Without basic connection": (31, 0.481, 0.717, 14.3),
        "Without content": (109, 0.562, 0.760, 22.3), "Without time-based traffic": (112, 0.495, 0.732, 17.4),
        "Without host-based traffic": (111, 0.613, 0.770, 45.9), "Top 20 by mutual information": (20, 0.482, 0.753, 25.1),
        "Top 40 by mutual information": (40, 0.485, 0.740, 16.5)},
    "LSTM": {
        "All features": (121, 0.670, 0.803, 44.9), "Without basic connection": (31, 0.526, 0.738, 32.7),
        "Without content": (109, 0.644, 0.789, 31.8), "Without time-based traffic": (112, 0.591, 0.743, 33.8),
        "Without host-based traffic": (111, 0.612, 0.818, 72.3), "Top 20 by mutual information": (20, 0.559, 0.785, 46.3),
        "Top 40 by mutual information": (40, 0.561, 0.759, 46.7)},
}

# Table XIX: macro-F1 by training-set size (share of KDDTrain+: 12,597 / 31,493 / 62,986 / 125,973 records).
SIZE = {
    "Decision Tree": [0.517, 0.565, 0.578, 0.531], "Random Forest": [0.515, 0.507, 0.529, 0.516],
    "SVM (RBF)": [0.535, 0.586, 0.617, 0.613], "MLP": [0.620, 0.624, 0.618, 0.649], "LSTM": [0.618, 0.644, 0.602, 0.670],
}
SIZE_PERCENT = [10, 25, 50, 100]

RESEARCH_QUESTIONS = [
    ("RQ1. Do the deep learning models detect attacks better than the classical models?",
     "Yes on the five-class task with default decision rules, with limits. The MLP and LSTM reach macro-F1 0.631 and "
     "0.641 and accuracy 0.801, against 0.510 to 0.612 and 0.743 to 0.761 for the classical models. The gain comes from "
     "DoS, Probe, R2L and U2R, not from Normal traffic. It did not hold on the binary task, where the SVM ties the LSTM, "
     "and not on ROC-AUC, where the Random Forest is highest."),
    ("RQ2. What training and inference cost does any accuracy gain carry?",
     "Mostly training time. On 2 CPU cores the neural networks need 70 to 79 s against 0.9 to 11.3 s for the classical "
     "models, about 10 times the SVM and 80 times the Decision Tree. At inference they are still fast in absolute terms "
     "(2.85 ms per record for the MLP, 8.73 ms for the LSTM) and small. The Random Forest is both the slowest per record "
     "(17.9 ms) and the largest model (38 MB). GPU training time was not measured."),
    ("RQ3. How well are attack types absent from training detected?",
     "Poorly, and no family is reliable. The models flag 17% (Random Forest) to 53% (Decision Tree) of the 3,750 unseen "
     "records as attacks. The neural networks assign the correct family most often (31% to 33%). Unseen R2L records "
     "were essentially never recognised."),
    ("RQ4. Which approach suits a resource-constrained deployment, and which suits an accuracy-first one?",
     "By the rule fixed before any result was seen, the MLP is preferable for resource-constrained use (within 0.01 of "
     "the best macro-F1 and faster per record than the LSTM), and the MLP and LSTM tie for accuracy-first use. The "
     "MLP-versus-LSTM margin (0.0099) is close to the 0.01 threshold, so this is sensitive. A stricter budget could "
     "reasonably accept the SVM, which gives up 0.028 of macro-F1 for 0.34 ms per record and 6.8 s of training."),
]

FINDINGS = [
    ("Neural networks lead, at a price",
     "MLP and LSTM reach macro-F1 0.631 and 0.641 against 0.510 to 0.612, but need 70 to 79 s of CPU training against "
     "0.9 to 11.3 s."),
    ("Rare attacks decide the ranking",
     "All five models do well on Normal, DoS and Probe. R2L recall runs from 0.02 to 0.24, and no model detects R2L well."),
    ("Validation scores mislead",
     "The Random Forest won validation (0.953 macro-F1) and ranked last on the test file (0.510), because the test "
     "file holds 17 attack types never seen in training."),
    ("Unseen attacks are mostly missed",
     "Only 17% to 53% of the 3,750 unseen-type records were even flagged as attacks, whichever model family was used."),
    ("Accuracy alone hides the problem",
     "Normal and DoS are 76% of the test file, so accuracy of 0.74 to 0.80 still leaves 26% to 41% of attacks undetected."),
]

LIMITATIONS = [
    "NSL-KDD is derived from traffic recorded in 1998; results describe offline classification of pre-extracted "
    "connection features, not modern networks, raw packets or encrypted traffic.",
    "Validation did not predict test performance (Random Forest: best validation score, lowest test score). A validation "
    "split that holds out attack types would give a more honest selection signal.",
    "Ablations use one seed and frozen hyperparameters; differences below the seed-to-seed spread are not interpreted.",
    "U2R has only 67 test records, so its scores move by 1.5 percentage points of recall per record.",
    "The LSTM sees each record as a sequence of one time step, so it is a gated feed-forward layer and not a true "
    "sequence model.",
    "Costs were measured on one 2-core CPU session without a GPU; absolute times will differ on other hardware.",
    "No decision thresholds were tuned, and the optional CICIDS2017 replication was not run.",
]

REFERENCES = [
    "M. Tavallaee, E. Bagheri, W. Lu and A. A. Ghorbani, A detailed analysis of the KDD CUP 99 data set, IEEE CISDA, 2009.",
    "N. V. Chawla, K. W. Bowyer, L. O. Hall and W. P. Kegelmeyer, SMOTE: Synthetic minority over-sampling technique, JAIR 16, 2002.",
    "L. Breiman, Random forests, Machine Learning 45(1), 2001.",
    "S. Hochreiter and J. Schmidhuber, Long short-term memory, Neural Computation 9(8), 1997.",
    "D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, ICLR, 2015.",
    "T. G. Dietterich, Approximate statistical tests for comparing supervised classification learning algorithms, Neural Computation 10(7), 1998.",
    "F. Pedregosa et al., Scikit-learn: Machine learning in Python, JMLR 12, 2011.",
]

AUTHORS = ["Shubham Patel", "Vinamra Lilariaa", "Tejas Baid", "Anushka Chitranshi"]

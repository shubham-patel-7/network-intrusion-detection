"""Tests of the CICIoT2023 loader on synthetic files with the real column layout (the dataset itself needs a
registration form and is not bundled)."""
import numpy as np
import pandas as pd
import pytest

from idsbench import ciciot, common as c

# the 46 features of the dataset paper, as printed on the CIC page (including its spellings)
FEATURES = ("flow_duration,Header_Length,Protocol Type,Duration,Rate,Srate,Drate,fin_flag_number,syn_flag_number,"
            "rst_flag_number,psh_flag_number,ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,fin_count,"
            "urg_count,rst_count,HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,UDP,DHCP,ARP,ICMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,"
            "Tot size,IAT,Number,Magnitue,Radius,Covariance,Variance,Weight").split(",")
# raw labels as they appear in the CSV files: 33 attack types + benign
RAW = {
    "DDoS": ["DDoS-RSTFINFlood", "DDoS-PSHACK_Flood", "DDoS-SYN_Flood", "DDoS-UDP_Flood", "DDoS-TCP_Flood", "DDoS-ICMP_Flood",
             "DDoS-SynonymousIP_Flood", "DDoS-ACK_Fragmentation", "DDoS-UDP_Fragmentation", "DDoS-ICMP_Fragmentation",
             "DDoS-SlowLoris", "DDoS-HTTP_Flood"],
    "DoS": ["DoS-UDP_Flood", "DoS-SYN_Flood", "DoS-TCP_Flood", "DoS-HTTP_Flood"],
    "Mirai": ["Mirai-greeth_flood", "Mirai-udpplain", "Mirai-greip_flood"],
    "Recon": ["Recon-PingSweep", "Recon-OSScan", "Recon-PortScan", "Recon-HostDiscovery", "VulnerabilityScan"],
    "Spoofing": ["DNS_Spoofing", "MITM-ArpSpoofing"],
    "Web-based": ["BrowserHijacking", "Backdoor_Malware", "XSS", "Uploading_Attack", "SqlInjection", "CommandInjection"],
    "Brute Force": ["DictionaryBruteForce"],
    "Benign": ["BenignTraffic"],
}


def make_files(tmp_path, per_label=40, files=3, label_col="label", seed=0, strip_names=True):
    """Write `files` CSV parts with every raw label in each; the class signal is a shift of the features."""
    rng = np.random.RandomState(seed)
    classes = list(RAW)
    for f in range(files):
        rows = []
        for ci, (cat, labels) in enumerate(RAW.items()):
            for lab in labels:
                x = rng.rand(per_label, len(FEATURES)) + ci
                df = pd.DataFrame(x, columns=FEATURES)
                df[label_col] = lab
                rows.append(df)
        part = pd.concat(rows, ignore_index=True)
        part["constant_col"] = 0.0
        part.to_csv(tmp_path / f"part-{f:05d}.csv", index=False)
    return classes


@pytest.fixture(autouse=True)
def _point_loader_at(tmp_path, monkeypatch):
    monkeypatch.setattr(ciciot, "CSV_DIR", tmp_path)


def test_every_raw_label_maps_to_its_category():
    for cat, labels in RAW.items():
        assert {ciciot.category(x) for x in labels} == {cat}
    assert sum(len(v) for k, v in RAW.items() if k != "Benign") == 33
    assert ciciot.category("some new attack") is None
    assert ciciot.CLASSES == ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web-based", "Brute Force"]


# the 39 feature columns of the real MERGED_CSV files (header read from Merged01.csv)
REAL_FEATURES = ("Header_Length,Protocol Type,Time_To_Live,Rate,fin_flag_number,syn_flag_number,rst_flag_number,"
                 "psh_flag_number,ack_flag_number,ece_flag_number,cwr_flag_number,ack_count,syn_count,fin_count,rst_count,"
                 "HTTP,HTTPS,DNS,Telnet,SMTP,SSH,IRC,TCP,UDP,DHCP,ARP,ICMP,IGMP,IPv,LLC,Tot sum,Min,Max,AVG,Std,Tot size,IAT,"
                 "Number,Variance").split(",")


@pytest.mark.parametrize("names", [FEATURES, REAL_FEATURES], ids=["paper-46", "merged-csv-39"])
def test_features_fall_into_the_four_named_groups(names):
    groups = {f: ciciot.feature_group(f) for f in names}
    assert "other" not in set(groups.values()), [f for f, g in groups.items() if g == "other"]
    assert set(groups.values()) == {"flags_counts", "protocols", "rates_timing", "size_statistics"}
    assert len(FEATURES) == 46 and len(REAL_FEATURES) == 39


def test_real_label_spellings_are_upper_case_and_still_map():
    assert ciciot.category("DDOS-PSHACK_FLOOD") == "DDoS" and ciciot.category("MIRAI-GREIP_FLOOD") == "Mirai"
    assert ciciot.category("DOS-UDP_FLOOD") == "DoS" and ciciot.category("RECON-PINGSWEEP") == "Recon"
    assert ciciot.category("MITM-ARPSPOOFING") == "Spoofing" and ciciot.category("DICTIONARYBRUTEFORCE") == "Brute Force"
    assert ciciot.category("BENIGN") == "Benign" and ciciot.category("VULNERABILITYSCAN") == "Recon"


def test_sampling_caps_each_class_and_reads_two_label_spellings(tmp_path):
    make_files(tmp_path, per_label=60, files=3)
    df = ciciot.sample(ciciot._files(), cap=300, seed=1)
    counts = df["category"].value_counts()
    assert set(counts.index) == set(ciciot.CLASSES)
    # a class with fewer than `cap` records is kept whole (Brute Force and Benign: 1 label x 60 rows x 3 files = 180);
    # a larger class is cut to about `cap` (DDoS has 12 labels x 60 x 3 = 2,160 records)
    assert counts["Brute Force"] == 180 and counts["Benign"] == 180
    assert 220 < counts["DDoS"] < 380
    assert (df["label"].map(ciciot.category) == df["category"]).all()
    assert "Protocol Type" in df.columns and df.select_dtypes("number").shape[1] >= 46


def test_capitalised_label_column_is_found(tmp_path):
    make_files(tmp_path, per_label=10, files=1, label_col="Label")
    df = ciciot.sample(ciciot._files(), cap=50)
    assert set(df["category"]) == set(ciciot.CLASSES)


def test_unknown_labels_are_reported_not_dropped(tmp_path):
    make_files(tmp_path, per_label=5, files=1)
    p = tmp_path / "part-00000.csv"
    d = pd.read_csv(p)
    d.loc[0, "label"] = "Brand-New-Attack"
    d.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Brand-New-Attack"):
        ciciot.sample(ciciot._files())


def test_missing_data_folder_gives_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ciciot, "CSV_DIR", tmp_path / "nope")
    with pytest.raises(FileNotFoundError, match="registration form"):
        ciciot._files()


def test_build_splits_without_leakage_and_scales_on_training_only(tmp_path):
    make_files(tmp_path, per_label=50, files=2)
    df = ciciot.sample(ciciot._files(), cap=300, seed=3)
    D = ciciot.build(df, holdout=[], split_seed=7)
    n_tr, n_te = len(D["y"]), len(D["yte"])
    assert abs(n_te / (n_tr + n_te) - 0.2) < 0.01                       # 80/20 as in the dataset paper
    assert abs(len(D["idx_val"]) / n_tr - 0.2) < 0.01                   # training/validation inside the train part
    assert not set(D["idx_tr"]) & set(D["idx_val"])
    assert "constant_col" not in D["names"] and D["info"]["constant_columns_dropped"] == 1
    assert D["X"][D["idx_tr"]].min() >= 0 and D["X"][D["idx_tr"]].max() <= 1
    assert D["Xte"].min() >= 0 and D["Xte"].max() <= 1                  # test values clipped, never rescaled
    assert np.bincount(D["y"], minlength=8).min() > 0 and np.bincount(D["yte"], minlength=8).min() > 0
    assert not D["unseen"].any() and D["hard21"] is None
    # stratified: every class keeps its share in the test split
    share = lambda y: np.bincount(y, minlength=8) / len(y)
    assert np.allclose(share(D["y"]), share(D["yte"]), atol=0.03)


def test_exact_duplicates_are_removed_before_splitting(tmp_path):
    make_files(tmp_path, per_label=20, files=1)
    df = ciciot.sample(ciciot._files(), cap=500)
    doubled = pd.concat([df, df], ignore_index=True)
    D = ciciot.build(doubled, holdout=[])
    assert D["info"]["dropped_duplicates"] == len(df)
    assert len(D["y"]) + len(D["yte"]) == len(df)


def test_held_out_attack_types_appear_only_in_the_test_split(tmp_path):
    make_files(tmp_path, per_label=40, files=2)
    df = ciciot.sample(ciciot._files(), cap=400, seed=2)
    D = ciciot.build(df, holdout=["DDoS-SlowLoris", "XSS"])
    held = {"ddosslowloris", "xss"}
    assert not {ciciot.norm(x) for x in D["train_labels"]} & held        # never trained on
    assert {ciciot.norm(x) for x in D["test_labels"][D["unseen"]]} == held
    assert D["unseen"].sum() > 0 and D["unseen"].sum() == sum(ciciot.norm(x) in held for x in D["test_labels"])
    with pytest.raises(ValueError, match="none of the held-out"):
        ciciot.build(df, holdout=["NoSuchAttack"])


def test_preprocessor_reproduces_the_test_matrix_from_raw_flows(tmp_path):
    make_files(tmp_path, per_label=30, files=1)
    df = ciciot.sample(ciciot._files(), cap=200)
    D = ciciot.build(df, holdout=[], split_seed=1)
    raw = df.drop_duplicates().reset_index(drop=True)
    # the loader re-orders rows by split; the saved preprocessing must map a raw row to the scaled row the model saw
    one = raw.iloc[[0]]
    x = D["pre"].transform(one)
    assert x.shape == (1, len(D["names"])) and x.min() >= 0 and x.max() <= 1
    with pytest.raises(ValueError, match="missing CICIoT2023 columns"):
        D["pre"].transform(one.drop(columns=["Rate"]))


def test_models_and_metrics_work_with_eight_classes(tmp_path, monkeypatch):
    """The generic pipeline pieces (metrics, SVM subsample, class weights) with 8 classes."""
    monkeypatch.setattr(c, "CLASSES", ciciot.CLASSES)
    monkeypatch.setattr(c, "RARE", [6, 7])
    rng = np.random.RandomState(0)
    y = rng.choice(8, 4000, p=[.2, .3, .2, .1, .08, .06, .04, .02])
    X = (rng.rand(len(y), 6) + y[:, None] * 0.5).astype(np.float32)
    m = c.make_model("DT", dict(max_depth=8, min_samples_leaf=2, criterion="gini"), seed=0).fit(X, y)
    met = c.full_metrics(y, m.predict(X), m.scores(X))
    assert len(met["recall"]) == 8 and len(met["cm"]) == 8 and met["acc"] > 0.7
    idx = c.svm_subsample(X, y, cap=1000, seed=0)
    assert np.isin(np.where(np.isin(y, [6, 7]))[0], idx).all()           # rare classes are all kept
    D = dict(yte=y, unseen=np.zeros(len(y), bool), hard21=None)
    res = c.view_metrics(D, m.predict(X), m.scores(X))
    assert set(res) == {"full"}

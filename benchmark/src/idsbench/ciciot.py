"""CICIoT2023 loader: the same benchmark protocol on a modern, much larger IoT intrusion dataset.

CICIoT2023 (Neto et al., Sensors 2023): 33 attack types in seven categories plus benign traffic. The paper reports
46.7 million flows; the MERGED_CSV download (63 files, 8.7 GB) that this loader was checked against holds 45,019,234
labelled flows with 39 numeric features and a `Label` column in upper case (e.g. DDOS-PSHACK_FLOOD); every file
contains all 34 labels. The files are NOT bundled with this repository and the download needs a short registration
form at https://www.unb.ca/cic/datasets/iotdataset-2023.html . Put the CSV files in data/ciciot2023/ (subfolders are
searched) or point IDS_CICIOT_DIR at them. Download only one of the CSV and MERGED_CSV folders, not both: they hold the
same flows and would be counted twice.

Protocol:
  * the 33 attack labels are mapped to 7 categories, so there are 8 classes (Benign + 7), see common.CLASSES;
  * the CSV files are streamed twice (count the classes, then draw a sample of at most IDS_CICIOT_CAP records per
    class; default 30,000), so the 47 million rows never sit in memory; the sample is cached;
  * exact duplicate rows are dropped before splitting, so a record cannot appear in both train and test;
  * a stratified 80/20 split gives train and test (the split reported in the dataset paper); the train part is
    split 80/20 again into training and validation as the main run does;
  * min-max scaling is fitted on the training partition only (values outside it are clipped);
  * optionally IDS_CICIOT_HOLDOUT="DDoS-SlowLoris,XSS" removes those attack types from training: they appear only in
    the test set and are scored as the "unseen attack types" view.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from . import common as c

CLASSES = c._DATASETS["ciciot2023"]["classes"]  # Benign + the seven categories, whichever dataset is active
CSV_DIR = Path(os.environ.get("IDS_CICIOT_DIR", c.ROOT / "data" / "ciciot2023"))
CAP = int(os.environ.get("IDS_CICIOT_CAP", "30000"))
SAMPLE_SEED = int(os.environ.get("IDS_CICIOT_SEED", "0"))
HOLDOUT = [s.strip() for s in os.environ.get("IDS_CICIOT_HOLDOUT", "").split(",") if s.strip()]
LABEL_COLUMNS = ("label", "class", "attack", "attack_type")
TIME_COLUMNS = ("ts", "timestamp")


def norm(s):
    """Lower-case letters and digits only, so 'DDoS-RSTFINFlood' and 'ddos_rstfin_flood' compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# raw attack label -> category. Matching is on normalised names; the lists follow the dataset paper (Table 2).
_WEB = ["browserhijacking", "backdoormalware", "xss", "uploadingattack", "sqlinjection", "commandinjection"]


def category(raw):
    """Category (one of common.CLASSES) of a raw CICIoT2023 label, or None if the label is not recognised."""
    n = norm(raw)
    if n in ("benigntraffic", "benign"):
        return "Benign"
    if n.startswith("ddos"):
        return "DDoS"
    if n.startswith("dos"):
        return "DoS"
    if n.startswith("mirai"):
        return "Mirai"
    if n.startswith("recon") or n == "vulnerabilityscan":
        return "Recon"
    if n in ("dnsspoofing", "arpspoofing") or n.startswith("mitm"):
        return "Spoofing"
    if n in _WEB:
        return "Web-based"
    if n == "dictionarybruteforce":
        return "Brute Force"
    return None


# feature groups for the feature ablation (the 46 features of the dataset paper, by name)
_GROUPS = {
    "flags_counts": ["finflagnumber", "synflagnumber", "rstflagnumber", "pshflagnumber", "ackflagnumber", "eceflagnumber",
                     "cwrflagnumber", "ackcount", "syncount", "fincount", "urgcount", "rstcount"],
    "protocols": ["http", "https", "dns", "telnet", "smtp", "ssh", "irc", "tcp", "udp", "dhcp", "arp", "icmp", "igmp", "ipv",
                  "llc", "protocoltype"],
    "rates_timing": ["flowduration", "duration", "rate", "srate", "drate", "iat", "headerlength", "timetolive"],
    "size_statistics": ["totsum", "min", "max", "avg", "std", "totsize", "number", "magnitue", "magnitude", "radius",
                        "covariance", "variance", "weight"],
}
_GROUP_OF = {n: g for g, names in _GROUPS.items() for n in names}


def feature_group(name):
    return _GROUP_OF.get(norm(name), "other")


class Preprocessor:
    """Fitted scaler plus the kept feature columns. Picklable, so a trained model can be reused on new flows."""

    def __init__(self, cols, scaler):
        self.cols, self.scaler = list(cols), scaler

    def raw(self, d):
        x = d[self.cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, d):
        missing = [k for k in self.cols if k not in d.columns]
        if missing:
            raise ValueError(f"missing CICIoT2023 columns: {missing[:6]}")
        return self.scaler.transform(self.raw(d)).astype(np.float32)


def _files():
    files = sorted(CSV_DIR.rglob("*.csv")) if CSV_DIR.exists() else []
    if not files:
        raise FileNotFoundError(
            f"no CICIoT2023 CSV files under {CSV_DIR}. The dataset needs a registration form at "
            "https://www.unb.ca/cic/datasets/iotdataset-2023.html ; put the CSV files there or set IDS_CICIOT_DIR.")
    return files


def _label_column(file):
    header = pd.read_csv(file, nrows=0).columns.tolist()
    for want in LABEL_COLUMNS:
        for col in header:
            if col.strip().lower() == want:
                return col
    raise ValueError(f"no label column ({'/'.join(LABEL_COLUMNS)}) in {file.name}: {header[:8]} ...")


def sample(files, cap=None, seed=None):
    """Two streaming passes over the CSV files: count the classes, then keep ~`cap` random records per class.
    Returns a DataFrame with the numeric features, a `label` column (raw label) and a `category` column."""
    cap = CAP if cap is None else cap
    seed = SAMPLE_SEED if seed is None else seed
    label_col = _label_column(files[0])
    totals, unknown = {}, set()
    for f in files:  # pass 1: class counts, reading the label column only
        lab = pd.read_csv(f, usecols=[label_col])[label_col].astype(str).str.strip()
        for raw, n in lab.value_counts().items():
            cat = category(raw)
            if cat is None:
                unknown.add(raw)
            else:
                totals[cat] = totals.get(cat, 0) + int(n)
    if unknown:
        raise ValueError(f"unrecognised CICIoT2023 labels: {sorted(unknown)}")
    keep_p = {cat: min(1.0, cap / n) for cat, n in totals.items()}
    rng = np.random.RandomState(seed)
    parts = []
    for f in files:  # pass 2: sample each class with its own probability
        df = pd.read_csv(f)
        df.columns = [k.strip() for k in df.columns]
        raw = df[label_col.strip()].astype(str).str.strip()
        cat = raw.map(category)
        keep = rng.random_sample(len(df)) < cat.map(keep_p).to_numpy(float)
        df = df.loc[keep].copy()
        df["label"], df["category"] = raw[keep].values, cat[keep].values
        if label_col.strip() != "label":
            df = df.drop(columns=[label_col.strip()])
        num = [k for k in df.columns if k not in ("label", "category") and k.lower() not in TIME_COLUMNS]
        df[num] = df[num].apply(pd.to_numeric, errors="coerce").astype(np.float32)
        parts.append(df.drop(columns=[k for k in df.columns if k.lower() in TIME_COLUMNS]))
    out = pd.concat(parts, ignore_index=True)
    out.attrs["class_totals"] = totals
    return out


def _cached_sample():
    files = _files()
    cache = CSV_DIR / f"_sample_cap{CAP}_seed{SAMPLE_SEED}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    df = sample(files)
    df.to_pickle(cache)
    return df


def build(df, holdout=None, split_seed=None):
    """Split, scale and package a sampled DataFrame into the dict every pipeline stage expects."""
    holdout = HOLDOUT if holdout is None else holdout
    split_seed = c.SPLIT_SEED if split_seed is None else split_seed
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)  # identical flows must not land in both train and test
    feat = [k for k in df.columns if k not in ("label", "category")]
    y_all = df["category"].map({k: i for i, k in enumerate(CLASSES)}).to_numpy(int)
    held = df["label"].map(norm).isin({norm(h) for h in holdout}).to_numpy() if holdout else np.zeros(len(df), bool)
    if holdout and not held.any():
        raise ValueError(f"none of the held-out labels {holdout} occur in the sample")
    rest = np.where(~held)[0]
    tr_i, te_i = train_test_split(rest, test_size=0.2, stratify=y_all[rest], random_state=split_seed)
    te_i = np.concatenate([te_i, np.where(held)[0]])  # held-out attack types are never trained on
    unseen = np.concatenate([np.zeros(len(te_i) - held.sum(), bool), np.ones(int(held.sum()), bool)])

    Xtr_raw = df.loc[tr_i, feat].to_numpy(np.float64)
    Xte_raw = df.loc[te_i, feat].to_numpy(np.float64)
    Xtr_raw, Xte_raw = (np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0) for a in (Xtr_raw, Xte_raw))
    y, yte = y_all[tr_i], y_all[te_i]
    idx_tr, idx_val = train_test_split(np.arange(len(tr_i)), test_size=0.2, stratify=y, random_state=split_seed)
    lo, hi = Xtr_raw[idx_tr].min(0), Xtr_raw[idx_tr].max(0)
    kept = np.where(hi > lo)[0]  # constant columns carry no information
    cols = [feat[i] for i in kept]
    scaler = MinMaxScaler(clip=True).fit(Xtr_raw[idx_tr][:, kept])
    pre = Preprocessor(cols, scaler)
    groups = np.array([feature_group(k) for k in cols])
    group_names = [g for g in list(_GROUPS) + ["other"] if (groups == g).any()]
    return dict(X=scaler.transform(Xtr_raw[:, kept]).astype(np.float32), y=y, idx_tr=idx_tr, idx_val=idx_val,
                Xte=scaler.transform(Xte_raw[:, kept]).astype(np.float32), yte=yte, unseen=unseen, hard21=None,
                names=cols, groups=groups, group_names=group_names, pre=pre,
                train_labels=df.loc[tr_i, "label"].to_numpy(), test_labels=df.loc[te_i, "label"].to_numpy(),
                frame=df, test_index=te_i,  # the de-duplicated raw rows and which of them form the test split (for exports)
                info=dict(rows_sampled=n_before, rows_after_dedup=len(df), dropped_duplicates=n_before - len(df),
                          constant_columns_dropped=len(feat) - len(cols), holdout=list(holdout)))


def prepare():
    """Sample (cached), split, scale: the CICIoT2023 counterpart of `common.prepare()`."""
    return build(_cached_sample())

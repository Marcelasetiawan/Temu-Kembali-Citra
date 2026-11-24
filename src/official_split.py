"""Percobaan tambahan: pembagian folder bawaan dataset Kaggle (train / val / test).

- Uji  : seluruh citra folder `test` bawaan.
- Latih: citra folder `train` bawaan, dikurangi 10% pasien untuk validasi (dibagi per pasien).
- Validasi: 10% pasien folder `train` + seluruh citra folder `val` bawaan (hanya 16 citra, terlalu sedikit sendiri).

Metode sama dengan proposed model (DenseNet121, Skenario 2 augmentasi, seed 42).
Hasil disimpan terpisah dan tidak mengubah hasil utama penelitian:
  data/splits_official.csv, outputs/runs/s2_augment__densenet121__official_split/, outputs/extra/official_split.csv
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import CLASSES, DATA_DIR, ROOT, load_splits, run_dir
from train import train

SPLIT_NAME = "splits_official.csv"
OUT = ROOT / "outputs" / "extra"
TAG = "official_split"


def make_split(seed=42):
    df = load_splits()                                   # citra unik (duplikat identik sudah dihapus)
    df["split"] = df["orig_split"]                       # train / val / test bawaan
    tr = df[df["orig_split"] == "train"]
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    _, va_idx = next(sgkf.split(tr, tr["label"], groups=tr["patient"]))
    df.loc[tr.index[va_idx], "split"] = "val"
    df.to_csv(DATA_DIR / SPLIT_NAME, index=False)
    return df


def patient_overlap(df):
    latih = set(df.loc[df["split"] == "train", "patient"])
    uji = df[df["split"] == "test"]
    bocor = uji["patient"].isin(latih)
    return int(bocor.sum()), len(uji)


if __name__ == "__main__":
    df = make_split()
    print(df.groupby(["split", "label"]).size().unstack()[CLASSES], flush=True)
    n_bocor, n_uji = patient_overlap(df)
    print(f"Citra uji yang pasiennya juga ada di data latih: {n_bocor} dari {n_uji}", flush=True)

    m = train("s2_augment", "densenet121", splits=SPLIT_NAME, tag=TAG, workers=0)
    pc = pd.read_csv(run_dir("s2_augment", "densenet121", TAG) / "per_class.csv").set_index("class")
    row = {"Model": "DenseNet121 + augmentasi, folder bawaan Kaggle",
           "Akurasi": m["test"]["accuracy"], "Macro F1": m["test"]["macro_f1"],
           "F1 Normal": pc.loc["Normal", "f1"], "F1 Bakteri": pc.loc["Bacteria", "f1"],
           "F1 Virus": pc.loc["Virus", "f1"], "P@10": m["retrieval"]["P@10"], "mAP": m["retrieval"]["mAP"],
           "Citra uji": n_uji, "Citra uji pasien bocor": n_bocor}
    res = pd.DataFrame([row])
    OUT.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(OUT / "official_split.csv", index=False)
    print(res.set_index("Model").T.to_string())

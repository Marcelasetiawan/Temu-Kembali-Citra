"""Menyiapkan dataset X-ray dada anak (Kermany) untuk eksperimen.

Langkah:
1. Membaca label 3 kelas (Normal, Bacteria, Virus) dan ID pasien dari nama file.
2. Menghapus citra yang identik (hash MD5 sama).
3. Membagi data per pasien (stratified group split 70/15/15) agar tidak ada kebocoran data.
4. Menyimpan cache citra grayscale 256x256: versi asli dan versi CLAHE (skenario 3).
"""
import hashlib
import re
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data pneumonia" / "chest_xray"
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
CLASSES = ["Normal", "Bacteria", "Virus"]
CACHE_SIZE = 256
SEED = 42
N_FOLDS = 20  # 3 fold uji (15%), 3 fold validasi (15%), 14 fold latih (70%)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def parse_name(path):
    """Mengembalikan (label, id_pasien) dari nama file Kermany."""
    if path.parent.name == "NORMAL":
        m = re.match(r"((?:NORMAL2-)?IM-\d+)-", path.name)
        return "Normal", "N_" + m.group(1)
    m = re.match(r"person(\d+)_(bacteria|virus)_", path.name)
    # Nomor pasien yang sama dipakai pada file bakteri dan virus. Keduanya disatukan
    # dalam satu grup agar tetap aman apa pun arti penomoran tersebut.
    return m.group(2).capitalize(), "P_" + m.group(1)


def collect():
    rows = []
    for path in sorted(RAW_DIR.glob("*/*/*.jpeg")):
        label, patient = parse_name(path)
        rows.append({
            "raw_path": str(path.relative_to(ROOT)),
            "orig_split": path.parts[-3],
            "label": label,
            "patient": patient,
            "md5": hashlib.md5(path.read_bytes()).hexdigest(),
        })
    return pd.DataFrame(rows)


def split_by_patient(df):
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = pd.Series(-1, index=df.index)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["patient"])):
        fold.iloc[idx] = k
    df["split"] = fold.map(lambda k: "test" if k < 3 else "val" if k < 6 else "train")
    assert df.groupby("patient")["split"].nunique().max() == 1, "pasien bocor antar split"
    return df


def build_cache(df):
    for sub in ("original", "clahe"):
        (CACHE_DIR / sub).mkdir(parents=True, exist_ok=True)
    for row in df.itertuples():
        orig = CACHE_DIR / "original" / row.cache_name
        clahe = CACHE_DIR / "clahe" / row.cache_name
        if orig.exists() and clahe.exists():
            continue
        img = cv2.imread(str(ROOT / row.raw_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(orig), img)
        cv2.imwrite(str(clahe), CLAHE.apply(img))


def main():
    df = collect()
    n_raw = len(df)
    df = df.drop_duplicates("md5", keep="first").reset_index(drop=True)
    print(f"Citra mentah: {n_raw} | duplikat identik dihapus: {n_raw - len(df)} | tersisa: {len(df)}")

    df["label_idx"] = df["label"].map(CLASSES.index)
    df["cache_name"] = df["orig_split"] + "_" + df["raw_path"].map(lambda p: Path(p).stem) + ".png"
    df = split_by_patient(df)

    DATA_DIR.mkdir(exist_ok=True)
    df.drop(columns="md5").to_csv(DATA_DIR / "splits.csv", index=False)
    build_cache(df)

    counts = pd.crosstab(df["split"], df["label"])[CLASSES]
    counts["Total"] = counts.sum(axis=1)
    counts["Pasien"] = df.groupby("split")["patient"].nunique()
    counts.loc["Total"] = counts.sum()
    counts.to_csv(DATA_DIR / "split_summary.csv")
    print(counts.loc[["train", "val", "test", "Total"]])


if __name__ == "__main__":
    main()

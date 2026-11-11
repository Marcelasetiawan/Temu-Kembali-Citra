"""Perbandingan pada protokol yang sama dengan Benmalek dkk. (2025).

Protokol referensi (dari naskah lengkap): hanya citra folder latih Kaggle, pembagian acak per citra 80/20,
validasi dari data latih, input 227x227, normalisasi ImageNet, Adam lr 1e-4, cross-entropy tanpa bobot,
batch 32, 30 epoch dengan early stopping berdasarkan akurasi validasi, tanpa augmentasi.

1. Reproduksi MobileNetV2 dengan protokol tersebut.
2. Metode penelitian ini (train.py: DenseNet121 + augmentasi, AdamW, bobot kelas) pada split yang sama.
"""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from common import CLASSES, DATA_DIR, MEAN, ROOT, STD, RetrievalNet, XrayDataset, load_splits, run_dir, set_seed
from evaluate import classification_metrics, extract, retrieval_metrics
from train import train

SPLIT_NAME = "splits_benmalek.csv"
OUT = ROOT / "outputs" / "extra"


def make_split(seed=42):
    df = load_splits()
    df = df[df["orig_split"] == "train"].reset_index(drop=True)      # hanya folder latih Kaggle
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.20, stratify=df["label"], random_state=seed)
    tr, va = train_test_split(tr, test_size=0.20, stratify=df["label"].iloc[tr], random_state=seed)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    df.to_csv(DATA_DIR / SPLIT_NAME, index=False)
    return df


def reproduce_mobilenet(df, seed=42, epochs=30, patience=5):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([transforms.Resize((227, 227)), transforms.Grayscale(3), transforms.ToTensor(),
                             transforms.Normalize(MEAN, STD)])
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    dl = {s: DataLoader(XrayDataset(d, "original", tf), batch_size=32, shuffle=s == "train", num_workers=0)
          for s, d in parts.items()}
    model = RetrievalNet("mobilenet_v2").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    out_dir = run_dir("benmalek_protocol", "mobilenet_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    def predict(loader):
        model.eval()
        probs = []
        with torch.no_grad():
            for x, _ in loader:
                probs.append(model(x.to(device)).float().softmax(1).cpu())
        return torch.cat(probs).numpy()

    best, wait = -1.0, 0
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in dl["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        acc = (predict(dl["val"]).argmax(1) == parts["val"]["label_idx"].to_numpy()).mean()
        print(f"[reproduksi Benmalek] epoch {ep:02d} val_acc={acc:.4f}", flush=True)
        if acc > best:
            best, wait = acc, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(torch.load(out_dir / "best.pt", weights_only=True))
    prob = predict(dl["test"])
    y = parts["test"]["label_idx"].to_numpy()
    _, per_class, overall = classification_metrics(y, prob)
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    return overall, per_class


def summarize_row(name, overall, per_class, retr=None):
    f1 = per_class.set_index("class")["f1"]
    support = per_class.set_index("class")["support"]
    row = {"Model": name, "Akurasi": overall["accuracy"], "Macro F1": overall["macro_f1"],
           "Weighted F1": float((f1 * support).sum() / support.sum()),
           "F1 Normal": f1["Normal"], "F1 Bakteri": f1["Bacteria"], "F1 Virus": f1["Virus"]}
    if retr:
        row.update({"P@10": retr["P@10"], "mAP": retr["mAP"]})
    return row


if __name__ == "__main__":
    df = make_split()
    print(df.groupby(["split", "label"]).size().unstack()[CLASSES], flush=True)
    rows = []
    overall, per_class = reproduce_mobilenet(df)
    rows.append(summarize_row("MobileNetV2, reproduksi protokol Benmalek", overall, per_class))

    # workers=0: tanpa proses pemuat data terpisah agar hemat memori.
    m = train("s2_augment", "densenet121", splits=SPLIT_NAME, tag="benmalek_split", workers=0)
    pc = pd.read_csv(run_dir("s2_augment", "densenet121", "benmalek_split") / "per_class.csv")
    rows.append(summarize_row("DenseNet121 + augmentasi (metode penelitian ini)", m["test"], pc, m["retrieval"]))

    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "benmalek_protocol.csv", index=False)
    print((res.set_index("Model") * 100).round(2).to_string())

"""Eksperimen pendukung untuk Bab 4.

1. Uji kebocoran data: pembagian acak per citra (mengabaikan pasien) vs pembagian per pasien.
2. Baseline temu kembali dengan fitur pretrained ImageNet tanpa fine-tuning (pendekatan referensi utama 2).
3. Pengulangan seluruh konfigurasi dengan seed lain untuk rata-rata dan simpangan baku.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common import DATA_DIR, MODEL_LABELS, MODELS, ROOT, SCENARIO_LABELS, SCENARIOS, RetrievalNet, load_splits, run_dir
from evaluate import extract, retrieval_metrics
from train import train

EXTRA = ROOT / "outputs" / "extra"
SEEDS = (42, 7, 123)


def make_random_image_split():
    df = load_splits()
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=df["label"], random_state=42)
    va, te = train_test_split(rest, test_size=0.50, stratify=df["label"].iloc[rest], random_state=42)
    df["split"] = "train"
    df.loc[va, "split"] = "val"
    df.loc[te, "split"] = "test"
    df.to_csv(DATA_DIR / "splits_random_image.csv", index=False)
    shared = set(df.loc[df["split"] == "train", "patient"]) & set(df.loc[df["split"] == "test", "patient"])
    test_pat = df.loc[df["split"] == "test", "patient"]
    return {"test_patients": int(test_pat.nunique()), "test_patients_also_in_train": len(shared & set(test_pat)),
            "test_images_with_patient_in_train": int(test_pat.isin(shared).sum()), "test_images": int(len(test_pat))}


def leakage_experiment():
    info = make_random_image_split()
    rows = []
    for arch in ("mobilenet_v2", "efficientnet_b0"):
        if not (run_dir("s2_augment", arch, "randomsplit") / "metrics.json").exists():
            train("s2_augment", arch, splits="splits_random_image.csv", tag="randomsplit")
        for tag, label in (("", "Per pasien (penelitian ini)"), ("randomsplit", "Acak per citra")):
            m = json.loads((run_dir("s2_augment", arch, tag) / "metrics.json").read_text())
            rows.append({"Model": MODEL_LABELS[arch], "Pembagian data": label, "Akurasi": m["test"]["accuracy"],
                         "Macro F1": m["test"]["macro_f1"], "Macro AUC": m["test"]["macro_auc"],
                         "P@10": m["retrieval"]["P@10"], "mAP": m["retrieval"]["mAP"]})
    pd.DataFrame(rows).round(4).to_csv(EXTRA / "leakage_comparison.csv", index=False)
    (EXTRA / "leakage_info.json").write_text(json.dumps(info, indent=2))


def pretrained_baseline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_splits()
    tr, te = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    rows = []
    for arch in MODELS:
        model = RetrievalNet(arch, pretrained=True).to(device)
        g, _ = extract(model, tr, "original", device)
        q, _ = extract(model, te, "original", device)
        r, per_class = retrieval_metrics(q, te["label_idx"].to_numpy(), g, tr["label_idx"].to_numpy())
        rows.append({"Model": MODEL_LABELS[arch], **{k: r[k] for k in ("P@1", "P@5", "P@10", "mAP@10", "mAP")},
                     "mAP Virus": per_class["Virus"]["mAP"]})
    pd.DataFrame(rows).round(4).to_csv(EXTRA / "pretrained_retrieval.csv", index=False)


def seed_repeats():
    for seed in SEEDS[1:]:
        for scenario in SCENARIOS:
            for arch in MODELS:
                if not (run_dir(scenario, arch, f"seed{seed}") / "metrics.json").exists():
                    train(scenario, arch, seed=seed, tag=f"seed{seed}")
    rows = []
    for scenario in SCENARIOS:
        for arch in MODELS:
            for seed in SEEDS:
                m = json.loads((run_dir(scenario, arch, "" if seed == 42 else f"seed{seed}") / "metrics.json").read_text())
                rows.append({"Skenario": SCENARIO_LABELS[scenario], "Model": MODEL_LABELS[arch], "seed": seed,
                             "Akurasi": m["test"]["accuracy"], "Macro F1": m["test"]["macro_f1"],
                             "Macro AUC": m["test"]["macro_auc"], "P@10": m["retrieval"]["P@10"],
                             "mAP": m["retrieval"]["mAP"], "Val Macro F1": m["val_macro_f1"]})
    raw = pd.DataFrame(rows)
    raw.round(4).to_csv(EXTRA / "seed_runs.csv", index=False)
    agg = raw.groupby(["Skenario", "Model"], sort=False).agg(["mean", "std"]).drop(columns="seed")
    agg.columns = [f"{a} {b}" for a, b in agg.columns]
    agg.round(4).to_csv(EXTRA / "seed_summary.csv")


def select_proposed():
    """Proposed model = konfigurasi dengan rata-rata macro F1 validasi 3 seed tertinggi;
    checkpoint yang dipakai aplikasi = seed dengan macro F1 validasi tertinggi pada konfigurasi itu."""
    raw = pd.read_csv(EXTRA / "seed_runs.csv")
    best_cfg = raw.groupby(["Skenario", "Model"])["Val Macro F1"].mean().idxmax()
    runs = raw[(raw["Skenario"] == best_cfg[0]) & (raw["Model"] == best_cfg[1])]
    seed = int(runs.loc[runs["Val Macro F1"].idxmax(), "seed"])
    scenario = next(k for k, v in SCENARIO_LABELS.items() if v == best_cfg[0])
    arch = next(k for k, v in MODEL_LABELS.items() if v == best_cfg[1])
    best = {"scenario": scenario, "model": arch, "tag": "" if seed == 42 else f"seed{seed}", "seed": seed}
    (ROOT / "outputs" / "best_run.json").write_text(json.dumps(best))
    return best


if __name__ == "__main__":
    EXTRA.mkdir(parents=True, exist_ok=True)
    pretrained_baseline()
    print("Baseline pretrained selesai", flush=True)
    leakage_experiment()
    print("Uji kebocoran selesai", flush=True)
    seed_repeats()
    print("Pengulangan seed selesai", flush=True)
    print("Proposed model:", select_proposed(), flush=True)

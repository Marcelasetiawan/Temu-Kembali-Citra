"""Perbandingan setara dengan Saranya & Chinnadurai (2025): tugas biner normal vs pneumonia.

Prediksi 3 kelas digabung menjadi biner (bakteri + virus = pneumonia). Dihitung untuk proposed model
(split per pasien) dan model dengan split acak per citra (protokol yang sama dengan referensi).
Metrik temu kembali biner memakai relevansi normal vs pneumonia, dengan dan tanpa relevance feedback.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score

from common import ROOT, SCENARIOS, RetrievalNet, load_splits, run_dir
from evaluate import extract, retrieval_metrics
from relevance_feedback import rocchio

OUT = ROOT / "outputs" / "extra"


def binary_eval(scenario, arch, tag, splits_name, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir(scenario, arch, tag) / "best.pt", map_location=device, weights_only=True))
    df = load_splits(splits_name)
    tr, te = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    cache_sub = SCENARIOS[scenario][0]
    g, _ = extract(model, tr, cache_sub, device)
    q, prob = extract(model, te, cache_sub, device)
    yb = (te["label_idx"].to_numpy() > 0).astype(int)          # 1 = pneumonia (bakteri atau virus)
    pb = (prob.argmax(1) > 0).astype(int)
    gb = (tr["label_idx"].to_numpy() > 0).astype(int)
    r0, _ = retrieval_metrics(q, yb, g, gb)
    r1, _ = retrieval_metrics(rocchio(q, g, gb, yb, rounds=2), yb, g, gb)
    return {"Model": label, "Akurasi": (pb == yb).mean(), "Presisi": precision_score(yb, pb),
            "Recall": recall_score(yb, pb), "F1": f1_score(yb, pb), "P@10": r0["P@10"], "mAP": r0["mAP"],
            "P@10 + RF": r1["P@10"], "mAP + RF": r1["mAP"]}


if __name__ == "__main__":
    best = json.loads((ROOT / "outputs" / "best_run.json").read_text())
    rows = [binary_eval(best["scenario"], best["model"], best["tag"], "splits.csv",
                        "Proposed (DenseNet121 S2), split per pasien")]
    for arch, name in (("mobilenet_v2", "MobileNetV2 S2"), ("efficientnet_b0", "EfficientNet-B0 S2")):
        rows.append(binary_eval("s2_augment", arch, "", "splits.csv", f"{name}, split per pasien"))
        rows.append(binary_eval("s2_augment", arch, "randomsplit", "splits_random_image.csv",
                                f"{name}, split acak per citra"))
    res = pd.DataFrame(rows)
    res.round(4).to_csv(OUT / "binary_comparison.csv", index=False)
    print((res.set_index("Model") * 100).round(2).to_string())

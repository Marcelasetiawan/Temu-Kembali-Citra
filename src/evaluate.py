"""Evaluasi satu run: metrik klasifikasi (sesuai tabel template) dan metrik temu kembali citra."""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

from common import (CLASSES, MODEL_LABELS, SCENARIO_LABELS, SCENARIOS, RetrievalNet, XrayDataset,
                    build_transform, load_splits, run_dir)

TOP_K = (1, 5, 10)


@torch.no_grad()
def extract(model, df, cache_sub, device, batch=64, workers=4):
    ds = XrayDataset(df, cache_sub, build_transform(augment=False, train=False))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    embs, probs = [], []
    model.eval()
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            e = model.embed(x)
            p = model.head(e).float().softmax(1)
        embs.append(e.float().cpu())
        probs.append(p.cpu())
    return torch.cat(embs).numpy(), torch.cat(probs).numpy()


def classification_metrics(y_true, prob):
    y_pred = prob.argmax(1)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    n = cm.sum()
    rows = []
    for c, name in enumerate(CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = n - tp - fp - fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        rows.append({
            "class": name, "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "accuracy": (tp + tn) / n, "specificity": tn / (tn + fp) if tn + fp else 0.0,
            "auc": roc_auc_score((y_true == c).astype(int), prob[:, c]),
            "support": int(cm[c, :].sum()),
        })
    per_class = pd.DataFrame(rows)
    overall = {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_precision": float(per_class["precision"].mean()),
        "macro_recall": float(per_class["recall"].mean()),
        "macro_f1": float(per_class["f1"].mean()),
        "macro_specificity": float(per_class["specificity"].mean()),
        "macro_auc": float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro")),
    }
    return cm, per_class, overall


def retrieval_metrics(q_emb, q_lab, g_emb, g_lab):
    """Kueri = data uji, galeri = data latih. Relevan jika kelasnya sama. Kemiripan: cosine."""
    q = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    g = g_emb / np.linalg.norm(g_emb, axis=1, keepdims=True)
    order = np.argsort(-(q @ g.T), axis=1)
    rel = (g_lab[order] == q_lab[:, None]).astype(np.float32)

    hits = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)
    ap = (rel * hits / ranks).sum(1) / rel.sum(1)
    k = 10
    ap_k = (rel[:, :k] * hits[:, :k] / ranks[:k]).sum(1) / np.maximum(rel[:, :k].sum(1), 1)
    knn_pred = np.array([np.bincount(row, minlength=len(CLASSES)).argmax() for row in g_lab[order[:, :k]]])

    out = {f"P@{t}": float(rel[:, :t].mean()) for t in TOP_K}
    out.update({"mAP@10": float(ap_k.mean()), "mAP": float(ap.mean()), "kNN10_accuracy": float((knn_pred == q_lab).mean())})
    per_class = {CLASSES[c]: {"mAP": float(ap[q_lab == c].mean()), "P@10": float(rel[q_lab == c, :10].mean())}
                 for c in range(len(CLASSES))}
    return out, per_class


def plot_confusion(cm, title, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES)
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Label sebenarnya")
    ax.set_title(title, fontsize=9)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def evaluate_run(scenario, arch, workers=4, splits="splits.csv", tag=""):
    out_dir = run_dir(scenario, arch, tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device, weights_only=True))
    cache_sub, _ = SCENARIOS[scenario]

    df = load_splits(splits)
    parts = {s: df[df["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    feats = {}
    for s, d in parts.items():
        t0 = time.perf_counter()
        feats[s] = extract(model, d, cache_sub, device, workers=workers)
        if s == "test":
            infer_ms = (time.perf_counter() - t0) / len(d) * 1000
    y = {s: d["label_idx"].to_numpy() for s, d in parts.items()}

    cm, per_class, overall = classification_metrics(y["test"], feats["test"][1])
    _, _, val_overall = classification_metrics(y["val"], feats["val"][1])
    retr, retr_class = retrieval_metrics(feats["test"][0], y["test"], feats["train"][0], y["train"])

    metrics = {
        "scenario": scenario, "model": arch,
        "test": overall, "val_macro_f1": val_overall["macro_f1"],
        "retrieval": retr, "retrieval_per_class": retr_class,
        "inference_ms_per_image": infer_ms,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    per_class.to_csv(out_dir / "per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion(cm, f"{MODEL_LABELS[arch]} - {SCENARIO_LABELS[scenario]}", out_dir / "confusion_matrix.png")
    # Embedding galeri disimpan untuk aplikasi temu kembali.
    np.savez_compressed(out_dir / "gallery.npz", emb=feats["train"][0].astype(np.float16),
                        label=y["train"], cache_name=parts["train"]["cache_name"].to_numpy())
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate_run(args.scenario, args.model), indent=2))

"""Uji relevance feedback (Rocchio) pada embedding hasil fine-tuning dan embedding pretrained.

Pengguna disimulasikan menandai k hasil teratas sebagai relevan/tidak relevan (berdasarkan label galeri),
lalu kueri diperbarui: q' = a*q + b*mean(relevan) - c*mean(tidak relevan), dan pencarian diulang.
"""
import json

import numpy as np
import pandas as pd
import torch

from common import MODEL_LABELS, MODELS, ROOT, SCENARIOS, RetrievalNet, load_splits, run_dir
from evaluate import extract, retrieval_metrics

OUT = ROOT / "outputs" / "extra"


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def rocchio(q, g, gl, ql, k=10, rounds=1, a=1.0, b=0.75, c=0.25):
    q, g = l2(q), l2(g)
    for _ in range(rounds):
        top = np.argsort(-(q @ g.T), axis=1)[:, :k]
        rel = gl[top] == ql[:, None]
        new = []
        for i in range(len(q)):
            pos, neg = g[top[i][rel[i]]], g[top[i][~rel[i]]]
            v = a * q[i]
            if len(pos):
                v = v + b * pos.mean(0)
            if len(neg):
                v = v - c * neg.mean(0)
            new.append(v)
        q = l2(np.stack(new))
    return q


def evaluate(arch, scenario, tag, pretrained, device, tr, te):
    model = RetrievalNet(arch, pretrained=pretrained).to(device)
    if not pretrained:
        model.load_state_dict(torch.load(run_dir(scenario, arch, tag) / "best.pt", map_location=device,
                                         weights_only=True))
    cache_sub = SCENARIOS[scenario][0]
    g, _ = extract(model, tr, cache_sub, device, workers=0)
    q, _ = extract(model, te, cache_sub, device, workers=0)
    gl, ql = tr["label_idx"].to_numpy(), te["label_idx"].to_numpy()
    rows = []
    for name, rounds in (("Tanpa feedback", 0), ("Rocchio 1 putaran", 1), ("Rocchio 2 putaran", 2)):
        qq = rocchio(q, g, gl, ql, rounds=rounds) if rounds else q
        m, pc = retrieval_metrics(qq, ql, g, gl)
        rows.append({"Fitur": "Pretrained" if pretrained else "Fine-tuned", "Model": MODEL_LABELS[arch],
                     "seed": tag or "seed42", "Feedback": name, "P@10": m["P@10"], "mAP": m["mAP"],
                     "mAP Virus": pc["Virus"]["mAP"]})
    return rows


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_splits()
    tr, te = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    best = json.loads((ROOT / "outputs" / "best_run.json").read_text())
    rows = []
    # Proposed model pada ketiga seed untuk rata-rata dan simpangan baku.
    for tag in ("", "seed7", "seed123"):
        rows += evaluate(best["model"], best["scenario"], tag, False, device, tr, te)
    rows += evaluate(best["model"], "s1_original", "", True, device, tr, te)
    res = pd.DataFrame(rows).round(4)
    res.to_csv(OUT / "relevance_feedback.csv", index=False)
    summary = res.groupby(["Fitur", "Feedback"], sort=False)[["P@10", "mAP", "mAP Virus"]].agg(["mean", "std"])
    summary.columns = [f"{a} {b}" for a, b in summary.columns]
    summary.round(4).to_csv(OUT / "relevance_feedback_summary.csv")
    print((summary * 100).round(2).to_string())

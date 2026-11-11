"""Gambar untuk Bab 4 laporan: perbandingan skenario, contoh hasil temu kembali benar/salah."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import CACHE_DIR, CLASSES, MODEL_LABELS, MODELS, ROOT, SCENARIOS, RetrievalNet, load_splits, run_dir
from evaluate import extract

FIG = ROOT / "outputs" / "figures"
# Teks gambar per bahasa: "id" untuk laporan, "en" untuk artikel publikasi.
TEXT = {
    "id": {"cls": {"Normal": "Normal", "Bacteria": "Bakteri", "Virus": "Virus"}, "sc": "Skenario",
           "f1": "Macro F1 (klasifikasi)", "map": "mAP (temu kembali)", "ylabel": "Nilai (%), rata-rata 3 seed",
           "query": "Kueri", "ok": "Benar", "bad": "Salah", "suffix": ""},
    "en": {"cls": {"Normal": "Normal", "Bacteria": "Bacterial", "Virus": "Viral"}, "sc": "Scenario",
           "f1": "Macro F1 (classification)", "map": "mAP (retrieval)", "ylabel": "Value (%), mean of 3 seeds",
           "query": "Query", "ok": "Correct", "bad": "Wrong", "suffix": "_en"},
}
COLORS = ["#4C78A8", "#F58518", "#54A24B"]


def scenario_bars(lang="id"):
    tx = TEXT[lang]
    """Rata-rata ± simpangan baku dari 3 seed untuk setiap model dan skenario."""
    s = pd.read_csv(ROOT / "outputs" / "extra" / "seed_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, metric, title in ((axes[0], "Macro F1", tx["f1"]), (axes[1], "mAP", tx["map"])):
        x = np.arange(len(SCENARIOS))
        for i, m in enumerate(MODELS):
            sub = s[s["Model"] == MODEL_LABELS[m]]
            mean, std = sub[f"{metric} mean"].to_numpy() * 100, sub[f"{metric} std"].to_numpy() * 100
            bars = ax.bar(x + (i - 1) * 0.26, mean, 0.26, yerr=std, capsize=2.5, label=MODEL_LABELS[m],
                          color=COLORS[i], error_kw={"elinewidth": 0.8})
            ax.bar_label(bars, labels=[f"{v:.1f}" for v in mean], fontsize=7, padding=6)
        ax.set_xticks(x, [f"{tx['sc']} {i}" for i in (1, 2, 3)])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(60, 92)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(tx["ylabel"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / f"perbandingan_skenario{tx['suffix']}.png", dpi=200)
    plt.close(fig)


def retrieval_examples(k=5, langs=("id", "en")):
    best = json.loads((ROOT / "outputs" / "best_run.json").read_text())
    scenario, arch = best["scenario"], best["model"]
    cache_sub, _ = SCENARIOS[scenario]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(arch, pretrained=False).to(device)
    model.load_state_dict(torch.load(run_dir(scenario, arch, best.get("tag", "")) / "best.pt", map_location=device,
                                     weights_only=True))
    df = load_splits()
    tr, te = df[df["split"] == "train"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    g, _ = extract(model, tr, cache_sub, device)
    q, _ = extract(model, te, cache_sub, device)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = q @ g.T
    order = np.argsort(-sims, axis=1)[:, :k]
    gl, ql = tr["label_idx"].to_numpy(), te["label_idx"].to_numpy()
    prec = (gl[order] == ql[:, None]).mean(1)

    rng = np.random.default_rng(3)
    picks = []
    for c, good in ((0, True), (1, True), (2, True), (2, False), (1, False)):
        cand = np.where((ql == c) & ((prec == 1.0) if good else (prec <= 0.2)))[0]
        picks.append((rng.choice(cand), good))
    for lang in langs:
        tx = TEXT[lang]
        fig, axes = plt.subplots(len(picks), k + 1, figsize=(1.75 * (k + 1), 1.95 * len(picks)))
        for r, (qi, good) in enumerate(picks):
            ax = axes[r, 0]
            ax.imshow(Image.open(CACHE_DIR / cache_sub / te["cache_name"][qi]), cmap="gray")
            ax.set_title(f"{tx['query']}: {tx['cls'][CLASSES[ql[qi]]]}", fontsize=8, fontweight="bold")
            ax.set_ylabel(tx["ok"] if good else tx["bad"], fontsize=9, color="#2a7d2a" if good else "#c0392b")
            for j, gi in enumerate(order[qi]):
                ax = axes[r, j + 1]
                ax.imshow(Image.open(CACHE_DIR / cache_sub / tr["cache_name"][gi]), cmap="gray")
                ok = gl[gi] == ql[qi]
                ax.set_title(f"#{j + 1} {tx['cls'][CLASSES[gl[gi]]]} ({sims[qi, gi]:.2f})", fontsize=7,
                             color="#2a7d2a" if ok else "#c0392b")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#2a7d2a" if ok else "#c0392b")
                    sp.set_linewidth(2)
            for ax in axes[r]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(FIG / f"contoh_temu_kembali{tx['suffix']}.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    scenario_bars("id")
    scenario_bars("en")
    retrieval_examples()
    print("Gambar Bab 4 tersimpan di", FIG)

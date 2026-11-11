"""Merangkum semua run menjadi tabel untuk laporan (format mengikuti Tabel 1 template)."""
import json

import pandas as pd

from common import MODEL_LABELS, MODELS, ROOT, SCENARIO_LABELS, SCENARIOS, run_dir

OUT = ROOT / "outputs"


def summarize():
    overall, per_class = [], []
    for scenario in SCENARIOS:
        for arch in MODELS:
            d = run_dir(scenario, arch)
            if not (d / "metrics.json").exists():
                continue
            m = json.loads((d / "metrics.json").read_text())
            cfg = json.loads((d / "config.json").read_text())
            overall.append({
                "Skenario": SCENARIO_LABELS[scenario], "Model": MODEL_LABELS[arch],
                "Val Macro F1": m["val_macro_f1"],
                "Akurasi": m["test"]["accuracy"], "Macro Presisi": m["test"]["macro_precision"],
                "Macro Recall": m["test"]["macro_recall"], "Macro F1": m["test"]["macro_f1"],
                "Macro Specificity": m["test"]["macro_specificity"], "Macro AUC": m["test"]["macro_auc"],
                "P@1": m["retrieval"]["P@1"], "P@5": m["retrieval"]["P@5"], "P@10": m["retrieval"]["P@10"],
                "mAP@10": m["retrieval"]["mAP@10"], "mAP": m["retrieval"]["mAP"],
                "Epoch terbaik": cfg["best_epoch"], "Waktu latih (menit)": cfg["train_minutes"],
                "Inferensi (ms/citra)": m["inference_ms_per_image"],
                "scenario_key": scenario, "model_key": arch,
            })
            pc = pd.read_csv(d / "per_class.csv")
            pc.insert(0, "Model", MODEL_LABELS[arch])
            pc.insert(0, "Skenario", SCENARIO_LABELS[scenario])
            per_class.append(pc)
    if not overall:
        return None
    overall = pd.DataFrame(overall)
    OUT.mkdir(exist_ok=True)
    overall.round(4).to_csv(OUT / "summary_overall.csv", index=False)
    pd.concat(per_class).round(4).to_csv(OUT / "summary_per_class.csv", index=False)
    # Model terbaik run seed 42 dipilih berdasarkan macro F1 validasi, bukan data uji. Pilihan final
    # (rata-rata 3 seed) ditulis ke best_run.json oleh extra_experiments.select_proposed().
    best = overall.loc[overall["Val Macro F1"].idxmax()]
    best_seed42 = {"scenario": best["scenario_key"], "model": best["model_key"], "tag": "", "seed": 42}
    (OUT / "best_run_seed42.json").write_text(json.dumps(best_seed42))
    if not (OUT / "best_run.json").exists():
        (OUT / "best_run.json").write_text(json.dumps(best_seed42))
    return overall


if __name__ == "__main__":
    s = summarize()
    if s is not None:
        print(s.drop(columns=["scenario_key", "model_key"]).round(4).to_string(index=False))

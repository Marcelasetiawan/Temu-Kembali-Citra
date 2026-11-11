"""Fine-tuning satu model CNN pada satu skenario, lalu evaluasi."""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import (CLASSES, MODELS, SCENARIOS, RetrievalNet, XrayDataset, build_transform,
                    load_splits, run_dir, set_seed)
from evaluate import evaluate_run


def make_loader(df, cache_sub, augment, train, batch, workers):
    ds = XrayDataset(df, cache_sub, build_transform(augment, train))
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=workers,
                      pin_memory=True, persistent_workers=workers > 0, drop_last=train)


def plot_history(hist, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="Training")
    axes[0].plot(hist["epoch"], hist["val_loss"], label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(hist["epoch"], hist["train_acc"], label="Training")
    axes[1].plot(hist["epoch"], hist["val_acc"], label="Validation")
    axes[1].set_title("Akurasi")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def train(scenario, arch, epochs=20, batch=32, lr=1e-4, patience=5, seed=42, workers=4, splits="splits.csv", tag=""):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = run_dir(scenario, arch, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_sub, augment = SCENARIOS[scenario]

    df = load_splits(splits)
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    dl_tr = make_loader(tr, cache_sub, augment, True, batch, workers)
    dl_va = make_loader(va, cache_sub, augment, False, batch, workers)

    # Bobot kelas untuk mengatasi ketidakseimbangan (dihitung dari data latih saja).
    counts = np.bincount(tr["label_idx"], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    model = RetrievalNet(arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    hist, best_f1, wait = [], -1.0, 0
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            tn += len(y)
        sched.step()

        model.eval()
        vl, preds, ys = 0.0, [], []
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    out = model(x)
                    vl += criterion(out, y).item() * len(y)
                preds.append(out.argmax(1).cpu())
                ys.append(y.cpu())
        preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
        val_f1 = f1_score(ys, preds, average="macro")
        hist.append({"epoch": ep, "train_loss": tl / tn, "train_acc": tc / tn,
                     "val_loss": vl / len(ys), "val_acc": float((preds == ys).mean()), "val_macro_f1": val_f1})
        print(f"[{scenario} | {arch}] epoch {ep:02d} train_loss={tl / tn:.4f} train_acc={tc / tn:.4f} "
              f"val_loss={vl / len(ys):.4f} val_acc={(preds == ys).mean():.4f} val_f1={val_f1:.4f}", flush=True)

        # Model terbaik dipilih dengan data validasi; data uji hanya untuk evaluasi akhir.
        if val_f1 > best_f1:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping pada epoch {ep}", flush=True)
                break

    train_min = (time.perf_counter() - t_start) / 60
    hist = pd.DataFrame(hist)
    hist.to_csv(out_dir / "history.csv", index=False)
    plot_history(hist, out_dir / "curves.png")
    config = {"scenario": scenario, "model": arch, "epochs_max": epochs, "epochs_run": len(hist),
              "best_epoch": int(hist["val_macro_f1"].idxmax() + 1), "batch": batch, "lr": lr,
              "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "CosineAnnealing",
              "patience": patience, "seed": seed, "splits": splits, "train_minutes": train_min,
              "class_weights": weights.tolist(),
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    return evaluate_run(scenario, arch, workers=workers, splits=splits, tag=tag)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=SCENARIOS)
    ap.add_argument("--model", required=True, choices=MODELS)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    m = train(args.scenario, args.model, args.epochs, args.batch, args.lr, workers=args.workers)
    print(json.dumps(m, indent=2))

"""Gambar pendukung proposal/laporan: contoh dataset, distribusi kelas per split, dan efek CLAHE."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from prepare_data import CACHE_DIR, CLASSES, DATA_DIR, ROOT

FIG_DIR = ROOT / "outputs" / "figures"
COLORS = {"Normal": "#2a9d8f", "Bacteria": "#e76f51", "Virus": "#457b9d"}
LABEL_ID = {"Normal": "Normal", "Bacteria": "Pneumonia Bakteri", "Virus": "Pneumonia Virus"}


def dataset_samples(df, n=4):
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(2.2 * n, 2.4 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        sample = df[df["label"] == cls].sample(n, random_state=7)
        for c, name in enumerate(sample["cache_name"]):
            ax = axes[r, c]
            ax.imshow(Image.open(CACHE_DIR / "original" / name), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(f"({chr(97 + r)}) {LABEL_ID[cls]}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "contoh_dataset.png", dpi=200)
    plt.close(fig)


def class_distribution(df):
    counts = pd.crosstab(df["split"], df["label"]).loc[["train", "val", "test"], CLASSES]
    ax = counts.rename(index={"train": "Training", "val": "Validation", "test": "Testing"}).plot(
        kind="bar", color=[COLORS[c] for c in CLASSES], figsize=(6, 3.6), rot=0, width=0.75)
    for cont in ax.containers:
        ax.bar_label(cont, fontsize=8)
    ax.set_ylabel("Jumlah citra")
    ax.set_xlabel("")
    ax.legend([LABEL_ID[c] for c in CLASSES], frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distribusi_kelas.png", dpi=200)
    plt.close()


def clahe_comparison(df):
    picks = [df[df["label"] == c].sample(1, random_state=3)["cache_name"].iloc[0] for c in CLASSES]
    fig, axes = plt.subplots(2, len(CLASSES), figsize=(7.5, 5.2))
    for c, (cls, name) in enumerate(zip(CLASSES, picks)):
        for r, sub in enumerate(("original", "clahe")):
            ax = axes[r, c]
            ax.imshow(Image.open(CACHE_DIR / sub / name), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(LABEL_ID[cls], fontsize=10)
        axes[0, 0].set_ylabel("Sebelum CLAHE")
        axes[1, 0].set_ylabel("Sesudah CLAHE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "perbandingan_clahe.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "splits.csv")
    dataset_samples(df)
    class_distribution(df)
    clahe_comparison(df)
    print("Gambar tersimpan di", FIG_DIR)

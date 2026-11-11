"""Diagram untuk proposal: rancangan penelitian dan arsitektur tiga model."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from figures import FIG_DIR

C_DATA, C_PROC, C_MODEL, C_EVAL, C_HEAD = "#e8f1fb", "#fdf1e3", "#e9f6ef", "#f3ecfa", "#fbe9ea"


def box(ax, x, y, w, h, text, color, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#444444", lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444444", lw=1))


def research_design():
    fig, ax = plt.subplots(figsize=(7.2, 9.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis("off")
    steps = [
        (12.2, "Dataset X-ray dada anak (Kermany et al.)\n5.856 citra: Normal, Pneumonia Bakteri, Pneumonia Virus", C_DATA),
        (10.9, "Pelabelan 3 kelas dari nama file + hapus 32 citra identik\n(tersisa 5.824 citra, 3.117 pasien)", C_PROC),
        (9.6, "Pembagian data per pasien (stratified group split)\nTraining 70% | Validation 15% | Testing 15%", C_PROC),
    ]
    for y, t, c in steps:
        box(ax, 5, y, 8.4, 0.95, t, c)
    for (y1, _, _), (y2, _, _) in zip(steps, steps[1:]):
        arrow(ax, 5, y1 - 0.5, 5, y2 + 0.5)

    xs = [1.75, 5, 8.25]
    sc = ["Skenario 1\nData asli\n(resize 224x224)", "Skenario 2\nAugmentasi data latih\n(rotasi, geser, zoom,\nkecerahan, kontras)",
          "Skenario 3\nCLAHE + augmentasi\ndata latih"]
    for x, t in zip(xs, sc):
        arrow(ax, 5, 9.1, x, 8.55)
        box(ax, x, 7.95, 3.0, 1.15, t, C_PROC, fs=8)
        arrow(ax, x, 7.35, x, 6.85)
    box(ax, 5, 6.35, 9.4, 0.95, "Fine-tuning 3 model CNN pretrained ImageNet\nModel 1: MobileNetV2 | Model 2: EfficientNet-B0 | Model 3: DenseNet121", C_MODEL)
    arrow(ax, 5, 5.85, 5, 5.35)
    box(ax, 5, 4.85, 8.4, 0.95, "Ekstraksi embedding (Global Average Pooling)\nGaleri = data training | Kueri = data testing", C_MODEL)
    arrow(ax, 3, 4.35, 2.6, 3.75)
    arrow(ax, 7, 4.35, 7.4, 3.75)
    box(ax, 2.6, 3.15, 4.4, 1.15, "Klasifikasi (head softmax)\nConfusion matrix, accuracy, precision,\nrecall, F1-score, specificity, AUC", C_EVAL, fs=8)
    box(ax, 7.4, 3.15, 4.4, 1.15, "Temu kembali (cosine similarity)\nPrecision@k (k=1, 5, 10),\nmAP@10, mAP", C_EVAL, fs=8)
    arrow(ax, 2.6, 2.55, 4.4, 1.95)
    arrow(ax, 7.4, 2.55, 5.6, 1.95)
    box(ax, 5, 1.45, 8.4, 0.95, "Model terbaik (macro F1 validasi) + relevance feedback Rocchio\n"
        "-> perbandingan state of the art + aplikasi temu kembali citra", C_HEAD)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rancangan_penelitian.png", dpi=220)
    plt.close(fig)


def architecture(name, blocks, embed_dim, fname):
    n = len(blocks)
    fig, ax = plt.subplots(figsize=(7.2, 1.0 + 0.62 * (n + 5)))
    H = n + 5
    ax.set_xlim(0, 10)
    ax.set_ylim(0, H)
    ax.axis("off")
    y = H - 0.6
    box(ax, 5, y, 6.4, 0.5, "Input citra X-ray 224 x 224 x 3", C_DATA)
    for b in blocks:
        arrow(ax, 5, y - 0.27, 5, y - 0.73)
        y -= 1
        box(ax, 5, y, 6.4, 0.5, b, C_MODEL)
    arrow(ax, 5, y - 0.27, 5, y - 0.73)
    y -= 1
    box(ax, 5, y, 6.4, 0.5, f"Global Average Pooling  ->  embedding {embed_dim} dimensi", C_PROC, bold=True)
    arrow(ax, 4, y - 0.27, 2.6, y - 0.73)
    arrow(ax, 6, y - 0.27, 7.4, y - 0.73)
    y -= 1
    box(ax, 2.6, y, 4.3, 0.55, "Dropout 0,3 -> Dense 3 -> Softmax\n(Normal / Bakteri / Virus)", C_EVAL, fs=8)
    box(ax, 7.4, y, 4.3, 0.55, "Normalisasi L2 -> cosine similarity\nterhadap galeri -> top-k citra mirip", C_EVAL, fs=8)
    ax.set_title(name, fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    research_design()
    architecture("Model 1: MobileNetV2", [
        "Conv 3x3, 32 filter, stride 2",
        "Bottleneck t=1, c=16, n=1",
        "Inverted residual bottleneck t=6: c=24 (n=2), c=32 (n=3)",
        "Inverted residual bottleneck t=6: c=64 (n=4), c=96 (n=3)",
        "Inverted residual bottleneck t=6: c=160 (n=3), c=320 (n=1)",
        "Conv 1x1, 1280 filter",
    ], 1280, "arsitektur_model1_mobilenetv2.png")
    architecture("Model 2: EfficientNet-B0", [
        "Conv 3x3, 32 filter, stride 2",
        "MBConv1 3x3 (16) x1",
        "MBConv6 3x3 (24) x2  |  MBConv6 5x5 (40) x2",
        "MBConv6 3x3 (80) x3  |  MBConv6 5x5 (112) x3",
        "MBConv6 5x5 (192) x4  |  MBConv6 3x3 (320) x1  (+ Squeeze-Excitation)",
        "Conv 1x1, 1280 filter",
    ], 1280, "arsitektur_model2_efficientnetb0.png")
    architecture("Model 3: DenseNet121", [
        "Conv 7x7, 64 filter, stride 2 + Max Pooling 3x3",
        "Dense Block 1 (6 layer) -> Transition (Conv 1x1 + Avg Pool)",
        "Dense Block 2 (12 layer) -> Transition",
        "Dense Block 3 (24 layer) -> Transition",
        "Dense Block 4 (16 layer) -> BatchNorm + ReLU",
    ], 1024, "arsitektur_model3_densenet121.png")
    print("Diagram tersimpan di", FIG_DIR)

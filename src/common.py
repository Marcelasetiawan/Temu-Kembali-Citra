"""Komponen bersama: konfigurasi skenario, transformasi, dataset, dan model."""
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = ROOT / "outputs" / "runs"
CLASSES = ["Normal", "Bacteria", "Virus"]
IMG_SIZE = 224
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# skenario -> (folder cache, pakai augmentasi pada data latih)
SCENARIOS = {
    "s1_original": ("original", False),
    "s2_augment": ("original", True),
    "s3_clahe_augment": ("clahe", True),
}
MODELS = ["mobilenet_v2", "efficientnet_b0", "densenet121"]
MODEL_LABELS = {"mobilenet_v2": "MobileNetV2", "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet121"}
SCENARIO_LABELS = {"s1_original": "Skenario 1 (Original)", "s2_augment": "Skenario 2 (Augmentasi)",
                   "s3_clahe_augment": "Skenario 3 (CLAHE + Augmentasi)"}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform(augment, train):
    steps = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if augment and train:
        # Tanpa flip horizontal: posisi jantung pada X-ray dada tidak simetris.
        steps += [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    steps += [transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)


def load_splits(name="splits.csv"):
    return pd.read_csv(DATA_DIR / name)


class XrayDataset(Dataset):
    def __init__(self, df, cache_sub, transform):
        self.paths = [CACHE_DIR / cache_sub / n for n in df["cache_name"]]
        self.labels = df["label_idx"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.transform(img), self.labels[i]


class RetrievalNet(nn.Module):
    """CNN pretrained ImageNet: backbone menghasilkan embedding untuk temu kembali,
    head menghasilkan prediksi kelas untuk fine-tuning dan metrik klasifikasi."""

    def __init__(self, arch, num_classes=len(CLASSES), pretrained=True):
        super().__init__()
        pool = [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        if arch == "mobilenet_v2":
            m = models.mobilenet_v2(weights="IMAGENET1K_V1" if pretrained else None)
            self.backbone, dim = nn.Sequential(m.features, *pool), m.classifier[1].in_features
        elif arch == "efficientnet_b0":
            m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
            self.backbone, dim = nn.Sequential(m.features, *pool), m.classifier[1].in_features
        elif arch == "densenet121":
            m = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)
            self.backbone, dim = nn.Sequential(m.features, nn.ReLU(inplace=True), *pool), m.classifier.in_features
        else:
            raise ValueError(arch)
        self.embed_dim = dim
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, num_classes))

    def embed(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.embed(x))


def run_dir(scenario, arch, tag=""):
    return RUNS_DIR / (f"{scenario}__{arch}" + (f"__{tag}" if tag else ""))

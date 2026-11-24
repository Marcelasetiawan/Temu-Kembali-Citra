"""Menyiapkan gambar ringan untuk aplikasi versi online (Streamlit Community Cloud).

Folder data/cache (+-450 MB) tidak diunggah ke GitHub, sehingga aplikasi online memakai:
- data/demo/galeri/  : thumbnail JPEG galeri proposed model (hanya untuk ditampilkan, bukan untuk dihitung)
- data/demo/contoh/  : citra contoh data uji ukuran asli cache (dipakai sebagai kueri)
Embedding galeri tetap diambil dari gallery.npz, jadi hasil pencarian sama dengan versi lokal.

Jalankan: python src/make_demo_images.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE_DIR, CLASSES, DATA_DIR, ROOT, SCENARIOS, run_dir  # noqa: E402

DEMO_DIR = DATA_DIR / "demo"
THUMB = 224
N_CONTOH = 4                      # harus sama dengan contoh_uji(n=4) di app.py


def main():
    best = json.loads((ROOT / "outputs" / "best_run.json").read_text())
    sub, _ = SCENARIOS[best["scenario"]]
    names = np.load(run_dir(best["scenario"], best["model"], best.get("tag", "")) / "gallery.npz",
                    allow_pickle=True)["cache_name"]

    galeri = DEMO_DIR / "galeri"
    galeri.mkdir(parents=True, exist_ok=True)
    for n in names:
        img = Image.open(CACHE_DIR / sub / n).convert("L")
        img.thumbnail((THUMB, THUMB))
        img.save(galeri / (Path(n).stem + ".jpg"), quality=82, optimize=True)

    contoh = DEMO_DIR / "contoh"
    contoh.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_DIR / "splits.csv")
    df = df[df["split"] == "test"]
    k = 0
    for c in CLASSES:
        for n in df[df["label"] == c].head(N_CONTOH)["cache_name"]:
            Image.open(CACHE_DIR / "original" / n).save(contoh / n)
            k += 1

    total = sum(f.stat().st_size for f in DEMO_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"Galeri: {len(names)} thumbnail | contoh: {k} citra | total {total:.1f} MB")


if __name__ == "__main__":
    main()

# Temu Kembali Citra X-Ray Dada Berbasis Konten untuk Pneumonia Bakteri dan Virus pada Anak

Final project mata kuliah Temu Kembali Citra, Informatika, Universitas Muhammadiyah Malang (2026).

Sistem ini mencari X-ray dada anak yang paling mirip dengan X-ray yang diunggah, dari basis data kasus yang sudah
diketahui diagnosisnya: **normal, pneumonia bakteri, atau pneumonia virus**. Tujuannya membantu dokter
membandingkan kasus, bukan menggantikan diagnosis.

- **Fine-tuning CNN**: MobileNetV2, EfficientNet-B0, dan DenseNet121 (pretrained ImageNet) dilatih ulang pada X-ray;
  keluaran Global Average Pooling dipakai sebagai vektor fitur (embedding).
- **Pencarian**: cosine similarity antara embedding kueri dan galeri, ditampilkan 10 hasil teratas.
- **Relevance feedback (Rocchio)**: pengguna menandai hasil yang relevan, kueri diperbarui
  (α = 1,0; β = 0,75; γ = 0,25), lalu pencarian diulang.

## Hasil utama

Data uji per pasien, rata-rata 3 seed. Proposed model: **DenseNet121 + augmentasi (Skenario 2)**.

| Ukuran | Hasil |
|---|---|
| Macro F1 (3 kelas) | 81,2 ± 0,9% |
| mAP temu kembali | 81,7 ± 2,1% (fitur pretrained tanpa fine-tuning: 52,5%) |
| P@10 tanpa → dengan 2 putaran relevance feedback | 80,3% → 88,1% |
| mAP tanpa → dengan 2 putaran relevance feedback | 81,7% → 85,7% |
| Normal vs pneumonia (dibandingkan Saranya & Chinnadurai, 2025) | presisi 98,7%, recall 97,7%, F1 98,2% |

Rincian per skenario dan per kelas ada di `outputs/summary_overall.csv`, `outputs/summary_per_class.csv`, dan
`outputs/extra/`.

## Struktur folder

```
app.py                  aplikasi web Streamlit (pencarian + relevance feedback)
src/                    kode penelitian (persiapan data, pelatihan, evaluasi, eksperimen pendukung)
notebook/               versi notebook dari seluruh alur (01–04) beserta keluarannya
data/splits*.csv        pembagian data per pasien yang dipakai pada eksperimen
outputs/                hasil eksperimen, gambar, dan model terbaik
referensi/referensi.bib daftar pustaka (BibTeX)
```

Hanya bobot proposed model yang disertakan (`outputs/runs/s2_augment__densenet121__seed7/best.pt`). Bobot 30
percobaan lain tidak diunggah karena ukurannya besar, tetapi seluruh metriknya tetap ada di folder `outputs/`.

## Cara menjalankan

### 1. Instal library

Python 3.11. Untuk GPU NVIDIA (CUDA 12.4):

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### 2. Unduh dataset

Dataset X-ray dada anak Kermany dkk. (2018):
[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) atau
[Mendeley Data](https://data.mendeley.com/datasets/rscbjbr9sj/2).
Ekstrak sehingga strukturnya menjadi:

```
data pneumonia/chest_xray/train/NORMAL, train/PNEUMONIA, test/..., val/...
```

### 3. Siapkan data

```bash
cd src
python prepare_data.py
```

Langkah ini memberi label 3 kelas dari nama file, menghapus 32 citra duplikat, membagi data per pasien
(70/15/15), dan membuat cache citra di `data/cache/`. Cache ini juga diperlukan aplikasi untuk menampilkan gambar.

### 4. Jalankan aplikasi

Dari folder utama:

```bash
python -m streamlit run app.py
```

### 5. (Opsional) Ulangi eksperimen

Dari folder `src/`:

```bash
python run_all.py                 # 3 model × 3 skenario
python extra_experiments.py       # 3 seed, fitur pretrained, uji kebocoran data
python relevance_feedback.py      # evaluasi relevance feedback
python binary_comparison.py       # perbandingan normal vs pneumonia
python reproduce_benmalek.py      # reproduksi protokol Benmalek dkk. (2025)
```

## Catatan pembagian data

Nomor pasien kelas pneumonia diambil dari nama file (`person23_bacteria_105.jpeg` → pasien 23). Dataset tidak
menyediakan nomor pasien untuk kelas normal, sehingga citra normal dikelompokkan berdasarkan kode pada nama file
(`IM-0011-0001.jpeg` dan `IM-0011-0001-0001.jpeg` → kelompok `IM-0011`). Pengelompokan ini adalah asumsi;
karena hampir setiap kode hanya memiliki satu citra, kelas normal pada praktiknya terbagi per citra.

## Referensi utama

- E. Saranya and M. Chinnadurai, "Medical application driven content based medical image retrieval system for
  enhanced analysis of X-ray images," *Scientific Reports*, vol. 15, 2025, doi: 10.1038/s41598-025-14282-8.
- D. S. Kermany *et al.*, "Identifying medical diagnoses and treatable diseases by image-based deep learning,"
  *Cell*, vol. 172, no. 5, pp. 1122–1131, 2018.

Daftar lengkap ada di `referensi/referensi.bib`.

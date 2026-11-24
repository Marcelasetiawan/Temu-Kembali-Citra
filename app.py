"""Aplikasi temu kembali citra X-ray dada anak (Streamlit).

Jalankan dari folder project:  .venv\\Scripts\\python -m streamlit run app.py
"""
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import CACHE_DIR, CLASSES, DATA_DIR, MODEL_LABELS, ROOT, SCENARIO_LABELS, SCENARIOS, RetrievalNet, build_transform, run_dir  # noqa: E402
from prepare_data import CACHE_SIZE, CLAHE  # noqa: E402

LABEL_ID = {"Normal": "Normal", "Bacteria": "Pneumonia Bakteri", "Virus": "Pneumonia Virus"}
WARNA = {"Normal": "#16A34A", "Bacteria": "#EA580C", "Virus": "#7C3AED"}
OUT = ROOT / "outputs"
FIG = OUT / "figures"
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli", "Agustus", "September", "Oktober",
         "November", "Desember"]

# Ikon garis sederhana (gaya Lucide) untuk kartu statistik.
IKON = {
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/>'
             '<path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/>'
             '<line x1="12" y1="16" x2="12.01" y2="16"/>',
    "dot": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
           '<path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/>'
               '<path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
             '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
}

st.set_page_config(page_title="Temu Kembali X-Ray Pneumonia Anak", page_icon="🩻", layout="wide")
st.markdown("""
<style>
/* ---------- area utama ---------- */
[data-testid="stAppViewContainer"] {background:#EEF3FB;}
[data-testid="stHeader"] {background:transparent;}
.block-container {padding-top:1.4rem; max-width:1280px;}

/* ---------- sidebar biru ---------- */
section[data-testid="stSidebar"] {background:linear-gradient(180deg,#1D4ED8 0%,#1E40AF 100%);}
section[data-testid="stSidebar"] * {color:#FFFFFF;}
section[data-testid="stSidebar"] hr {border-color:rgba(255,255,255,.25);}
section[data-testid="stSidebar"] .stButton button {
  justify-content:flex-start; padding:.65rem 1rem; font-size:1rem; font-weight:600;
  border-radius:12px; border:none; box-shadow:none;}
section[data-testid="stSidebar"] .stButton button[kind="secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {background:transparent;}
section[data-testid="stSidebar"] .stButton button[kind="secondary"]:hover,
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {background:rgba(255,255,255,.12);}
section[data-testid="stSidebar"] .stButton button[kind="primary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] {background:#FFFFFF;}
section[data-testid="stSidebar"] .stButton button[kind="primary"] *,
section[data-testid="stSidebar"] [data-testid="stBaseButton-primary"] * {color:#1D4ED8 !important;}
.brand {display:flex; gap:12px; align-items:center; margin-bottom:.4rem;}
.brand-logo {width:44px; height:44px; border-radius:12px; background:rgba(255,255,255,.18);
  display:flex; align-items:center; justify-content:center; font-size:24px;}
.brand b {font-size:1.05rem;} .brand span {font-size:.8rem; opacity:.8;}
.side-label {font-size:.72rem; font-weight:700; letter-spacing:.08em; opacity:.75; margin:.6rem 0 .2rem .3rem;}
.side-card {background:rgba(255,255,255,.12); border-radius:12px; padding:.8rem 1rem; font-size:.85rem; line-height:1.5;}

/* ---------- kartu putih (hanya blok yang berisi penanda .card-marker langsung) ---------- */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .card-marker),
[data-testid="stVerticalBlockBorderWrapper"]:has(> [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] .card-marker) {
  background:#FFFFFF; border:1px solid #D6E2F5; border-radius:16px; padding:1.1rem 1.3rem;}
[data-testid="stElementContainer"]:has(.card-marker) {display:none;}
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] {display:none;}

/* ---------- komponen ---------- */
.topbar {display:flex; justify-content:space-between; align-items:center; background:#FFFFFF;
  border:1px solid #D6E2F5; border-radius:16px; padding:.9rem 1.3rem; margin-bottom:1.2rem;}
.topbar h2 {margin:0; font-size:1.35rem; color:#0F172A;} .topbar p {margin:0; color:#64748B; font-size:.9rem;}
.pill {display:flex; gap:10px; align-items:center; border:1px solid #D6E2F5; border-radius:999px; padding:.35rem .9rem .35rem .4rem;}
.pill .av {width:34px; height:34px; border-radius:50%; background:#1D4ED8; color:#fff; display:flex;
  align-items:center; justify-content:center; font-weight:700;}
.pill b {font-size:.88rem; color:#0F172A;} .pill span {font-size:.75rem; color:#64748B; display:block;}
.banner {background:linear-gradient(120deg,#1D4ED8 0%,#2563EB 60%,#3B82F6 100%); color:#fff;
  border-radius:18px; padding:1.6rem 1.8rem; margin-bottom:1.2rem; box-shadow:0 8px 24px rgba(29,78,216,.25);}
.banner small {font-weight:700; letter-spacing:.1em; opacity:.85;}
.banner h1 {color:#fff; font-size:1.7rem; margin:.4rem 0 .3rem 0; padding:0;}
.banner p {margin:0; opacity:.9;}
.stat {background:#FFFFFF; border:1px solid #D6E2F5; border-radius:16px; padding:1.1rem 1.2rem; height:100%;}
.stat-top {display:flex; justify-content:space-between; align-items:flex-start;}
.stat-label {color:#475569; font-size:.88rem;}
.stat-icon {width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;}
.stat-value {font-size:1.9rem; font-weight:800; color:#0F172A; margin:.3rem 0 0 0;}
.stat-sub {color:#475569; font-size:.82rem; margin-bottom:.6rem;}
.bar {height:6px; background:#E2E8F0; border-radius:999px; overflow:hidden;}
.bar div {height:100%; border-radius:999px;}
.badge {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff; font-size:.78rem; font-weight:600;}
.small {color:#64748B; font-size:.82rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- model dan data
SEED_TAG = {42: "", 7: "seed7", 123: "seed123"}
BEST = json.loads((OUT / "best_run.json").read_text())


@st.cache_data
def pilihan_model():
    """Kombinasi model x skenario yang filenya tersedia. Untuk tiap kombinasi dipakai seed dengan macro F1
    validasi tertinggi (cara yang sama dengan pemilihan proposed model); data uji tidak dipakai memilih."""
    runs = pd.read_csv(OUT / "extra" / "seed_runs.csv")
    ring = pd.read_csv(OUT / "extra" / "seed_summary.csv").set_index(["Skenario", "Model"])   # sama dengan laporan
    sk_key = {v: k for k, v in SCENARIO_LABELS.items()}
    md_key = {v: k for k, v in MODEL_LABELS.items()}
    hasil = {}
    for (sk, md), g in runs.groupby(["Skenario", "Model"]):
        scen, arch = sk_key[sk], md_key[md]
        seed = int(g.loc[g["Val Macro F1"].idxmax(), "seed"])
        d = run_dir(scen, arch, SEED_TAG[seed])
        if (d / "best.pt").exists() and (d / "gallery.npz").exists():
            r = ring.loc[(sk, md)]
            hasil[(scen, arch)] = {"tag": SEED_TAG[seed], "seed": seed, "val": r["Val Macro F1 mean"],
                                   "f1": r["Macro F1 mean"], "map": r["mAP mean"]}
    if (BEST["scenario"], BEST["model"]) not in hasil:          # cadangan: minimal model terbaik tersedia
        hasil[(BEST["scenario"], BEST["model"])] = {"tag": BEST.get("tag", ""), "seed": BEST.get("seed"),
                                                     "val": np.nan, "f1": np.nan, "map": np.nan}
    return hasil


@st.cache_resource
def load_system(scenario, arch, tag):
    d = run_dir(scenario, arch, tag)
    device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu")
    model = RetrievalNet(arch, pretrained=False)
    # bobot dibaca ke CPU dulu agar tetap bisa dijalankan di server tanpa GPU
    model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=True))
    model = model.to(device).eval()
    g = np.load(d / "gallery.npz", allow_pickle=True)
    emb = g["emb"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return model, device, emb, g["label"], g["cache_name"]


@st.cache_data
def contoh_uji(n=4):
    """Beberapa citra data uji per kelas (tidak ada di galeri) untuk demo tanpa unggah file."""
    df = pd.read_csv(DATA_DIR / "splits.csv")
    df = df[df["split"] == "test"]
    rows = []
    for c in CLASSES:
        for r in df[df["label"] == c].head(n).itertuples():
            rows.append({"nama": Path(r.raw_path).name, "label": c, "cache_name": r.cache_name})
    return pd.DataFrame(rows)


# Versi online tidak memiliki data/cache (terlalu besar untuk GitHub); gambar diambil dari data/demo
# yang dibuat oleh src/make_demo_images.py. Embedding tetap dari gallery.npz, jadi hasilnya sama.
DEMO_DIR = DATA_DIR / "demo"


def path_galeri(sub, name):
    p = CACHE_DIR / sub / name
    return p if p.exists() else DEMO_DIR / "galeri" / (Path(name).stem + ".jpg")


def path_contoh(name):
    p = CACHE_DIR / "original" / name
    return p if p.exists() else DEMO_DIR / "contoh" / name


def preprocess(pil_img, cache_sub):
    img = cv2.resize(np.array(pil_img.convert("L")), (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    if cache_sub == "clahe":
        img = CLAHE.apply(img)
    return Image.fromarray(img)


def card(parent=None):
    """Kartu putih berbingkai: container biasa yang diberi penanda untuk CSS."""
    c = (parent or st).container()
    c.markdown('<span class="card-marker"></span>', unsafe_allow_html=True)
    return c


def badge(c):
    return f'<span class="badge" style="background:{WARNA[c]}">{LABEL_ID[c]}</span>'


def tanggal():
    d = datetime.date.today()
    return f"{HARI[d.weekday()]}, {d.day} {BULAN[d.month - 1]} {d.year}"


def stat_card(label, value, sub, ikon, warna, persen):
    svg = (f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round">{IKON[ikon]}</svg>')
    return (f'<div class="stat"><div class="stat-top"><div class="stat-label">{label}</div>'
            f'<div class="stat-icon" style="background:{warna}">{svg}</div></div>'
            f'<div class="stat-value">{value}</div><div class="stat-sub">{sub}</div>'
            f'<div class="bar"><div style="width:{persen:.1f}%;background:{warna}"></div></div></div>')


def topbar(judul, sub):
    st.markdown(f"""<div class="topbar"><div><h2>{judul}</h2><p>{sub}</p></div>
<div class="pill"><div class="av">AI</div><div><b>{MODEL_LABELS[arch]}</b>
<span>{SCENARIO_LABELS[scenario]}</span></div></div></div>""", unsafe_allow_html=True)


# Pilihan model dibaca dari session_state (widget di halaman Cari) sebelum halaman digambar.
TERSEDIA = pilihan_model()
st.session_state.setdefault("pilih_model", BEST["model"])
st.session_state.setdefault("pilih_skenario", BEST["scenario"])
if (st.session_state.pilih_skenario, st.session_state.pilih_model) not in TERSEDIA:
    st.session_state.pilih_model, st.session_state.pilih_skenario = BEST["model"], BEST["scenario"]
scenario, arch = st.session_state.pilih_skenario, st.session_state.pilih_model
CFG = TERSEDIA[(scenario, arch)]
model, device, g_emb, g_lab, g_names = load_system(scenario, arch, CFG["tag"])
cache_sub, _ = SCENARIOS[scenario]
IS_BEST = (scenario, arch) == (BEST["scenario"], BEST["model"])

# ---------------------------------------------------------------- menu samping
MENU = [("beranda", "Beranda", ":material/home:"), ("cari", "Cari Kasus Mirip", ":material/image_search:"),
        ("hasil", "Hasil Penelitian", ":material/bar_chart:"), ("tentang", "Tentang Sistem", ":material/info:")]
st.session_state.setdefault("page", "beranda")


def pindah(page):
    st.session_state.page = page
    st.rerun()


with st.sidebar:
    st.markdown("""<div class="brand"><div class="brand-logo">🩻</div>
<div><b>X-Ray Pneumonia</b><br><span>Temu Kembali Citra Anak</span></div></div>""", unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="side-label">MENU UTAMA</div>', unsafe_allow_html=True)
    for key, label, icon in MENU:
        if st.button(label, key=f"menu_{key}", icon=icon, use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            pindah(key)
    st.divider()
    st.markdown(f"""<div class="side-card"><b>Model aktif</b><br>{MODEL_LABELS[arch]}<br>
{SCENARIO_LABELS[scenario]}<br><span style="opacity:.8">Arsip: {len(g_lab):,} X-ray berlabel</span></div>"""
                .replace(",", "."), unsafe_allow_html=True)
    st.write("")
    st.caption("Alat bantu pembanding kasus, bukan pengganti diagnosis dokter.")


# ---------------------------------------------------------------- halaman: beranda
def halaman_beranda():
    topbar("Beranda", "Ringkasan sistem temu kembali X-ray dada anak")
    st.markdown(f"""<div class="banner"><small>{tanggal().upper()}</small>
<h1>Selamat datang di sistem temu kembali X-ray</h1>
<p>Cari kasus X-ray dada anak yang paling mirip beserta diagnosisnya: normal, pneumonia bakteri, atau pneumonia virus.</p>
</div>""", unsafe_allow_html=True)

    s = pd.read_csv(DATA_DIR / "split_summary.csv").set_index("split").loc["Total"]
    tot = s["Total"]
    kartu = [
        ("Total X-ray", f"{tot:,}".replace(",", "."), "dataset Kermany dkk. (2018)", "image", "#4F46E5", 100),
        ("Normal", f"{s['Normal']:,}".replace(",", "."), f"{s['Normal'] / tot * 100:.1f}% dari total".replace(".", ","), "shield",
         WARNA["Normal"], s["Normal"] / tot * 100),
        ("Pneumonia Bakteri", f"{s['Bacteria']:,}".replace(",", "."), f"{s['Bacteria'] / tot * 100:.1f}% dari total".replace(".", ","),
         "alert", WARNA["Bacteria"], s["Bacteria"] / tot * 100),
        ("Pneumonia Virus", f"{s['Virus']:,}".replace(",", "."), f"{s['Virus'] / tot * 100:.1f}% dari total".replace(".", ","), "dot",
         WARNA["Virus"], s["Virus"] / tot * 100),
        ("Ketepatan menebak kelas", "81,2%", "DenseNet121, skenario 2", "cpu", "#0EA5E9", 81.2),
        ("10 hasil teratas benar", "88,1%", "naik dari 80,3% dengan umpan balik", "refresh", "#2563EB", 88.1),
    ]
    for start in (0, 3):
        cols = st.columns(3)
        for col, k in zip(cols, kartu[start:start + 3]):
            col.markdown(stat_card(*k), unsafe_allow_html=True)
        st.write("")

    with card():
        st.markdown("**Cara memakai**")
        c = st.columns(3)
        langkah = [("1. Masukkan X-ray", "Unggah foto X-ray dada atau pilih contoh foto yang tersedia."),
                   ("2. Lihat kasus mirip", "Sistem menampilkan X-ray lama yang paling mirip beserta diagnosisnya."),
                   ("3. Tandai yang cocok", "Centang ✓ foto yang sesuai, lalu klik Cari ulang agar hasil makin tepat.")]
        for col, (judul, isi) in zip(c, langkah):
            col.markdown(f"**{judul}**  \n{isi}")
        if st.button("Mulai cari kasus mirip", icon=":material/arrow_forward:", type="primary"):
            pindah("cari")


# ---------------------------------------------------------------- halaman: pencarian
def rocchio(q, relevant, nonrelevant, alpha, beta, gamma):
    """q' = α·q + β·rata-rata(relevan) − γ·rata-rata(tidak relevan), lalu dinormalisasi L2."""
    v = alpha * q
    if len(relevant):
        v = v + beta * g_emb[relevant].mean(0)
    if len(nonrelevant):
        v = v - gamma * g_emb[nonrelevant].mean(0)
    return v / np.linalg.norm(v)


def halaman_cari():
    topbar("Cari Kasus Mirip", "Masukkan X-ray, lihat kasus termirip, lalu tandai yang cocok")
    kiri, kanan = st.columns([1, 2], gap="medium")

    with kiri:
        with card():
            st.markdown("**Model yang dipakai**")
            arch_ops = sorted({a for _, a in TERSEDIA}, key=list(MODEL_LABELS).index)
            scen_ops = sorted({s for s, _ in TERSEDIA}, key=list(SCENARIO_LABELS).index)
            st.selectbox("Model CNN", arch_ops, key="pilih_model",
                         format_func=lambda a: MODEL_LABELS[a] + ("  ⭐ terbaik" if a == BEST["model"] else ""))
            st.selectbox("Skenario", scen_ops, key="pilih_skenario",
                         format_func=lambda s: SCENARIO_LABELS[s] + ("  ⭐ terbaik" if s == BEST["scenario"] else ""))
            if (st.session_state.pilih_skenario, st.session_state.pilih_model) not in TERSEDIA:
                st.warning("File model untuk kombinasi ini tidak tersedia. Dipakai model terbaik.")
            elif not np.isnan(CFG["f1"]):
                fmt = lambda v: f"{v * 100:.1f}".replace(".", ",")
                st.caption(f"Ketepatan menebak kelas {fmt(CFG['f1'])}% · ketepatan pencarian (mAP) "
                           f"{fmt(CFG['map'])}% · rata-rata 3 kali ulang" + (" · ⭐ model terbaik" if IS_BEST else ""))
            if len(TERSEDIA) == 1:
                st.caption("Hanya model terbaik yang tersedia di komputer ini.")

        with card():
            st.markdown("**1. Masukkan X-ray**")
            sumber = st.radio("Sumber gambar", ["Pilih contoh foto", "Unggah file sendiri"], horizontal=True,
                              label_visibility="collapsed")
            query_img, query_id, label_asli = None, None, None
            if sumber == "Unggah file sendiri":
                up = st.file_uploader("Unggah X-ray dada (JPG/PNG)", type=["jpg", "jpeg", "png"])
                if up:
                    query_img, query_id = Image.open(up), f"up_{up.file_id}"
            else:
                ex = contoh_uji()
                pilihan = st.selectbox("Contoh dari data uji (tidak ada di arsip)", ex.index,
                                       format_func=lambda i: f"{LABEL_ID[ex.loc[i, 'label']]} · {ex.loc[i, 'nama']}")
                row = ex.loc[pilihan]
                query_img = Image.open(path_contoh(row["cache_name"]))
                query_id, label_asli = f"ex_{row['cache_name']}", row["label"]
            with st.expander("Pengaturan", icon=":material/tune:"):
                k = st.slider("Jumlah foto mirip yang ditampilkan", 5, 20, 10, step=5)
                st.caption("Bobot umpan balik (algoritma Rocchio)")
                alpha = st.number_input("α · bobot pencarian awal", 0.0, 2.0, 1.0, 0.05)
                beta = st.number_input("β · bobot foto yang ditandai cocok", 0.0, 2.0, 0.75, 0.05)
                gamma = st.number_input("γ · bobot foto yang tidak ditandai", 0.0, 2.0, 0.25, 0.05)

    if query_img is None:
        with kanan:
            st.info("Pilih contoh foto atau unggah X-ray di sebelah kiri untuk memulai.")
        return

    query = preprocess(query_img, cache_sub)
    x = build_transform(augment=False, train=False)(query).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model.embed(x)
        prob = model.head(e).softmax(1)[0].cpu().numpy()
    e = e[0].cpu().numpy()
    e = e / np.linalg.norm(e)

    # Kueri disimpan per gambar agar putaran umpan balik tidak hilang saat halaman dimuat ulang.
    query_id = f"{scenario}|{arch}|{query_id}"             # ganti model = mulai pencarian baru
    if st.session_state.get("query_id") != query_id:
        st.session_state.update(query_id=query_id, q=e, round=0)
    sims = g_emb @ st.session_state.q
    top = np.argsort(-sims)[:k]
    rnd = st.session_state["round"]

    with kanan:
        with card():
            st.markdown("**2. Perkiraan sistem**")
            a, b = st.columns([1, 2], gap="medium")
            a.image(query, caption="X-ray yang dicari", use_container_width=True)
            with b:
                if label_asli:
                    st.markdown(f"Diagnosis asli: {badge(label_asli)}", unsafe_allow_html=True)
                utama = int(np.argmax(prob))
                st.markdown(f"Perkiraan sistem: {badge(CLASSES[utama])} &nbsp; **{prob[utama] * 100:.1f}%**".replace(".", ","),
                            unsafe_allow_html=True)
                for c in np.argsort(-prob):
                    st.progress(float(prob[c]), text=f"{LABEL_ID[CLASSES[c]]}: {prob[c] * 100:.1f}%".replace(".", ","))
                votes = np.bincount(g_lab[top], minlength=len(CLASSES))
                st.caption(f"Dari {k} kasus termirip, {votes.max()} berdiagnosis "
                           f"{LABEL_ID[CLASSES[votes.argmax()]]}.")

    with card():
        judul = f"**3. {k} kasus paling mirip dari arsip**"
        if rnd:
            judul += f" · setelah {rnd} kali umpan balik"
        st.markdown(judul)
        st.caption("Centang ✓ foto yang menurut Anda cocok, lalu klik **Cari ulang**. Foto yang tidak dicentang "
                   "dianggap tidak cocok.")
        n_kol = 5
        marked = []
        for start in range(0, len(top), n_kol):
            cols = st.columns(n_kol)
            for j, idx in enumerate(top[start:start + n_kol]):
                with cols[j]:
                    st.image(str(path_galeri(cache_sub, g_names[idx])), use_container_width=True)
                    mirip = f"{sims[idx]:.3f}".replace(".", ",")
                    st.markdown(f"**#{start + j + 1}** {badge(CLASSES[g_lab[idx]])}<br>"
                                f'<span class="small">Kemiripan {mirip}</span>', unsafe_allow_html=True)
                    marked.append(st.checkbox("Cocok ✓", key=f"rel_{scenario}_{arch}_{rnd}_{idx}"))

        b1, b2, _ = st.columns([2, 2, 3])
        if b1.button("Cari ulang dengan umpan balik", icon=":material/refresh:", type="primary",
                     use_container_width=True):
            marked = np.array(marked)
            st.session_state.q = rocchio(st.session_state.q, top[marked], top[~marked], alpha, beta, gamma)
            st.session_state["round"] = rnd + 1
            st.rerun()
        if b2.button("Kembali ke pencarian awal", icon=":material/undo:", use_container_width=True):
            st.session_state.update(q=e, round=0)
            st.rerun()


# ---------------------------------------------------------------- halaman: hasil penelitian
def halaman_hasil():
    topbar("Hasil Penelitian", "Dataset, 9 percobaan, umpan balik, dan contoh hasil")
    with card():
        tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "9 percobaan", "Umpan balik", "Contoh hasil"])

        with tab1:
            s = pd.read_csv(DATA_DIR / "split_summary.csv").set_index("split")
            s = s.rename(index={"train": "Latih (70%)", "val": "Validasi (15%)", "test": "Uji (15%)"},
                         columns={"Bacteria": "Bakteri", "Pasien": "Kelompok pasien"})
            st.markdown("5.824 X-ray dada anak (Kermany dkk., *Cell*, 2018), dibagi **per anak** sehingga foto "
                        "dari anak yang sama hanya ada di satu bagian.")
            s = s.loc[["Latih (70%)", "Validasi (15%)", "Uji (15%)", "Total"]].map(lambda v: f"{v:,}".replace(",", "."))
            s.index.name = "Bagian"
            st.dataframe(s, use_container_width=True)
            if (FIG / "contoh_dataset.png").exists():
                st.image(str(FIG / "contoh_dataset.png"), caption="Contoh X-ray tiap kelas", use_container_width=True)

        with tab2:
            ring = pd.read_csv(OUT / "extra" / "seed_summary.csv")
            tabel = ring.pivot(index="Skenario", columns="Model", values="Macro F1 mean") * 100
            tabel = tabel[["MobileNetV2", "EfficientNet-B0", "DenseNet121"]]
            st.markdown("Ketepatan menebak kelas (macro F1, %), rata-rata 3 kali ulang. "
                        "Terbaik: **DenseNet121 dengan skenario 2 (augmentasi)**.")
            st.dataframe(tabel.style.highlight_max(axis=None, color="#DBEAFE").format(lambda v: f"{v:.1f}".replace(".", ",")),
                         use_container_width=True)
            if (FIG / "perbandingan_skenario.png").exists():
                st.image(str(FIG / "perbandingan_skenario.png"), use_container_width=True)

        with tab3:
            rf = pd.read_csv(OUT / "extra" / "relevance_feedback_summary.csv")
            rf = rf[rf["Fitur"] == "Fine-tuned"]
            c = st.columns(3)
            warna = ["#93C5FD", "#3B82F6", "#1D4ED8"]
            for col, w, (_, r) in zip(c, warna, rf.iterrows()):
                label = r["Feedback"].replace("Rocchio", "Setelah umpan balik")
                col.markdown(stat_card(label, f"{r['P@10 mean'] * 100:.1f}%".replace(".", ","), "10 hasil teratas benar", "check", w,
                                       r["P@10 mean"] * 100), unsafe_allow_html=True)
            st.write("")
            st.caption("Penandaan ✓ disimulasikan oleh komputer sesuai diagnosis asli (seolah-olah dokter "
                       "selalu benar).")

        with tab4:
            if (FIG / "contoh_temu_kembali.png").exists():
                st.image(str(FIG / "contoh_temu_kembali.png"), use_container_width=True,
                         caption="Bingkai hijau = diagnosis sama dengan foto yang dicari, merah = berbeda")


# ---------------------------------------------------------------- halaman: tentang
def halaman_tentang():
    topbar("Tentang Sistem", "Judul penelitian, cara kerja, dan sumber")
    st.markdown("""<div class="banner"><small>FINAL PROJECT TEMU KEMBALI CITRA · INFORMATIKA UMM</small>
<h1 style="font-size:1.35rem">Temu Kembali Citra X-Ray Dada Berbasis Konten untuk Pendukung Diagnosis Pneumonia
Bakteri dan Virus pada Anak Menggunakan Fine-Tuning CNN dan Relevance Feedback</h1></div>""",
                unsafe_allow_html=True)
    c = st.columns(2)
    with card(c[0]):
        st.markdown("""**Cara kerja**
1. X-ray diubah ukurannya menjadi 224×224 piksel.
2. Model CNN yang sudah dilatih ulang mengambil ciri gambar menjadi deretan angka.
3. Angka itu dibandingkan dengan seluruh arsip (cosine similarity), lalu kasus termirip ditampilkan.
4. Foto yang ditandai cocok dipakai untuk memperbaiki pencarian (algoritma Rocchio).""")
    with card(c[1]):
        st.markdown("""**Sumber utama**
- Dataset: Kermany dkk., *Cell*, 2018
- Referensi utama: Saranya & Chinnadurai, *Scientific Reports*, 2025
- Pembanding 3 kelas: Benmalek dkk., *Research on Biomedical Engineering*, 2025
- Model: MobileNetV2 (2018), EfficientNet (2019), DenseNet (2017)""")
    st.warning("Sistem ini adalah alat bantu pembanding kasus, bukan pengganti diagnosis dokter.")


{"beranda": halaman_beranda, "cari": halaman_cari, "hasil": halaman_hasil,
 "tentang": halaman_tentang}[st.session_state.page]()

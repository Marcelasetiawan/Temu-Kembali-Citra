"""Aplikasi temu kembali citra X-ray dada anak (Streamlit).

Jalankan dari folder project:  .venv\\Scripts\\python -m streamlit run app.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import CACHE_DIR, CLASSES, MODEL_LABELS, ROOT, SCENARIO_LABELS, SCENARIOS, RetrievalNet, build_transform, run_dir  # noqa: E402
from prepare_data import CACHE_SIZE, CLAHE  # noqa: E402

LABEL_ID = {"Normal": "Normal", "Bacteria": "Pneumonia Bakteri", "Virus": "Pneumonia Virus"}


@st.cache_resource
def load_system():
    best = json.loads((ROOT / "outputs" / "best_run.json").read_text())
    scenario, arch = best["scenario"], best["model"]
    d = run_dir(scenario, arch, best.get("tag", ""))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RetrievalNet(arch, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(d / "best.pt", map_location=device, weights_only=True))
    g = np.load(d / "gallery.npz", allow_pickle=True)
    emb = g["emb"].astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return scenario, arch, model, device, emb, g["label"], g["cache_name"]


def preprocess(pil_img, cache_sub):
    img = cv2.resize(np.array(pil_img.convert("L")), (CACHE_SIZE, CACHE_SIZE), interpolation=cv2.INTER_AREA)
    if cache_sub == "clahe":
        img = CLAHE.apply(img)
    return Image.fromarray(img)


st.set_page_config(page_title="Temu Kembali X-Ray Pneumonia Anak", layout="wide")
st.title("Temu Kembali Citra X-Ray Dada Anak")
st.caption("Sistem menampilkan kasus X-ray paling mirip dari arsip berlabel (Normal, Pneumonia Bakteri, "
           "Pneumonia Virus) sebagai bahan pembanding. **Bukan pengganti diagnosis dokter.**")

scenario, arch, model, device, g_emb, g_lab, g_names = load_system()
cache_sub, _ = SCENARIOS[scenario]
st.sidebar.markdown(f"**Model:** {MODEL_LABELS[arch]}  \n**Konfigurasi:** {SCENARIO_LABELS[scenario]}")
k = st.sidebar.slider("Jumlah citra mirip yang ditampilkan", 3, 20, 8)

st.sidebar.markdown("**Relevance feedback (Rocchio)**")
alpha = st.sidebar.number_input("α (bobot kueri awal)", 0.0, 2.0, 1.0, 0.05)
beta = st.sidebar.number_input("β (bobot hasil relevan)", 0.0, 2.0, 0.75, 0.05)
gamma = st.sidebar.number_input("γ (bobot hasil tidak relevan)", 0.0, 2.0, 0.25, 0.05)


def rocchio(q, relevant, nonrelevant):
    """q' = α·q + β·rata-rata(relevan) − γ·rata-rata(tidak relevan), lalu dinormalisasi L2."""
    v = alpha * q
    if len(relevant):
        v = v + beta * g_emb[relevant].mean(0)
    if len(nonrelevant):
        v = v - gamma * g_emb[nonrelevant].mean(0)
    return v / np.linalg.norm(v)


uploaded = st.file_uploader("Unggah citra X-ray dada (JPG/PNG)", type=["jpg", "jpeg", "png"])
if uploaded:
    query = preprocess(Image.open(uploaded), cache_sub)
    x = build_transform(augment=False, train=False)(query).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model.embed(x)
        prob = model.head(e).softmax(1)[0].cpu().numpy()
    e = e[0].cpu().numpy()

    # Kueri disimpan per file agar putaran relevance feedback tidak hilang saat halaman dimuat ulang.
    if st.session_state.get("file_id") != uploaded.file_id:
        st.session_state.update(file_id=uploaded.file_id, q=e / np.linalg.norm(e), round=0)
    sims = g_emb @ st.session_state.q
    top = np.argsort(-sims)[:k]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(query, caption="Citra kueri (setelah praproses)", use_container_width=True)
    with c2:
        st.subheader("Probabilitas kelas")
        for c in np.argsort(-prob):
            st.progress(float(prob[c]), text=f"{LABEL_ID[CLASSES[c]]}: {prob[c] * 100:.1f}%")
        votes = np.bincount(g_lab[top], minlength=len(CLASSES))
        st.write(f"Mayoritas {k} kasus termirip: **{LABEL_ID[CLASSES[votes.argmax()]]}** "
                 f"({votes.max()} dari {k})")

    rnd = st.session_state["round"]
    st.subheader(f"{k} kasus paling mirip dari arsip" + (f" (setelah {rnd} putaran feedback)" if rnd else ""))
    st.caption("Centang kasus yang menurut Anda relevan (mirip secara klinis), lalu klik **Cari ulang**. "
               "Kasus yang tidak dicentang dianggap tidak relevan.")
    cols = st.columns(4)
    marked = []
    for i, idx in enumerate(top):
        with cols[i % 4]:
            st.image(str(CACHE_DIR / cache_sub / g_names[idx]), use_container_width=True,
                     caption=f"#{i + 1} {LABEL_ID[CLASSES[g_lab[idx]]]} | kemiripan {sims[idx]:.3f}")
            marked.append(st.checkbox("Relevan", key=f"rel_{rnd}_{idx}"))

    b1, b2 = st.columns(2)
    if b1.button("Cari ulang dengan relevance feedback", type="primary"):
        marked = np.array(marked)
        st.session_state.q = rocchio(st.session_state.q, top[marked], top[~marked])
        st.session_state["round"] = rnd + 1
        st.rerun()
    if b2.button("Reset ke pencarian awal"):
        st.session_state.update(q=e / np.linalg.norm(e), round=0)
        st.rerun()

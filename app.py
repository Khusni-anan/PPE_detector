import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
from PIL import Image
import time

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Monitoring K3 - APD",
    page_icon="👷",
    layout="wide"
)

# --- JUDUL & SIDEBAR ---
st.title("👷 Sistem Deteksi Kepatuhan APD (Real-Time)")
st.markdown("---")

st.sidebar.header("⚙️ Panel Kontrol")

# Pilihan Sumber Video
source_radio = st.sidebar.radio(
    "Sumber Kamera:",
    ["Webcam (Live)", "Upload Video (Tes)"]
)

# Slider Sensitivitas AI
conf_threshold = st.sidebar.slider(
    "Akurasi Minimal (Confidence)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05,
    help="Semakin tinggi, AI semakin 'pemilih' (hanya mendeteksi yang sangat jelas)."
)

# --- LOAD MODEL CUSTOM ---
@st.cache_resource
def load_model():
    # Memuat model hasil training kamu
    # Pastikan file 'best.pt' ada di satu folder dengan app.py
    model = YOLO('best.pt') 
    return model

try:
    model = load_model()
    st.sidebar.success("✅ Model AI Siap!")
    
    # Menampilkan kelas yang bisa dideteksi (Sesuai dataset kamu)
    with st.sidebar.expander("Daftar Objek Deteksi"):
        st.write(model.names)
        # Harusnya isinya: Boots, Gloves, Mask, Safety-Helmet, Safety-Vest, Safety-Wearpack
except Exception as e:
    st.sidebar.error(f"⚠️ Model 'best.pt' tidak ditemukan! Pastikan file ada di folder.")
    st.stop()

# --- FUNGSI PROSES VIDEO ---
def process_video(cap):
    st_frame = st.empty()
    
    # Statistik (KPI) di atas video
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize agar performa cepat
        frame = cv2.resize(frame, (854, 480))
        
        # 1. Deteksi dengan Model Kamu
        results = model(frame, conf=conf_threshold, verbose=False)

        # Variabel Hitungan per Frame
        count_helmet = 0
        count_vest = 0
        count_gloves = 0
        count_boots = 0

        # 2. Gambar Kotak & Label
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label_name = model.names[cls]
                conf = float(box.conf[0])
                
                # --- LOGIKA WARNA & KATEGORI ---
                # Sesuai training log kamu: 
                # Boots, Gloves, Mask, Safety-Helmet, Safety-Vest, Safety-Wearpack
                
                # KELOMPOK 1: APD UTAMA (Hijau Tebal)
                if label_name in ['Safety-Helmet', 'Safety-Vest', 'Safety-Wearpack']:
                    color = (0, 255, 0) # Hijau
                    if label_name == 'Safety-Helmet': count_helmet += 1
                    if label_name == 'Safety-Vest': count_vest += 1
                
                # KELOMPOK 2: APD TAMBAHAN (Kuning/Biru)
                elif label_name in ['Gloves', 'Mask', 'Boots']:
                    color = (255, 255, 0) # Cyan/Kuning
                    if label_name == 'Gloves': count_gloves += 1
                    if label_name == 'Boots': count_boots += 1
                
                else:
                    color = (255, 255, 255) # Putih (Lainnya)

                # Gambar Kotak
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Background Label
                text = f"{label_name} {int(conf*100)}%"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(frame, text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # 3. Peringatan Dini (Logika Sederhana)
        # Jika terdeteksi Rompi/Wearpack TAPI Helm Nol -> Potensi Pelanggaran
        if count_vest > 0 and count_helmet == 0:
            cv2.putText(frame, "PERINGATAN: CEK HELM!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # 4. Update Statistik KPI
        with kpi1:
            st.metric("⛑️ Helm", count_helmet)
        with kpi2:
            st.metric("🦺 Rompi", count_vest)
        with kpi3:
            st.metric("🧤 Sarung Tangan", count_gloves)
        with kpi4:
            st.metric("🥾 Sepatu", count_boots)

        # 5. Tampilkan ke Layar
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

# --- LOGIKA INPUT UTAMA ---
if source_radio == "Webcam (Live)":
    if st.sidebar.button("🔴 Mulai Kamera"):
        cap = cv2.VideoCapture(0) # Index 0 untuk webcam laptop
        process_video(cap)

elif source_radio == "Upload Video (Tes)":
    uploaded_file = st.sidebar.file_uploader("Upload video .mp4", type=['mp4', 'avi'])
    if uploaded_file is not None:
        # Simpan file sementara agar bisa dibaca OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.sidebar.button("▶️ Putar Video"):
            process_video(cap)

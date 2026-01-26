import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Monitoring K3 - Helm & Rompi",
    page_icon="👷",
    layout="wide"
)

# --- JUDUL & SIDEBAR ---
st.title("👷 Sistem Deteksi Kepatuhan APD (Helm & Rompi)")
st.markdown("---")

st.sidebar.header("⚙️ Panel Kontrol")

# Pilihan Sumber Media
source_radio = st.sidebar.radio(
    "Sumber Media:",
    ["Ambil Foto (Webcam)", "Upload Video", "Upload Gambar"]
)

# Slider Sensitivitas AI
conf_threshold = st.sidebar.slider(
    "Akurasi Minimal (Confidence)", 
    min_value=0.0, max_value=1.0, value=0.45, step=0.05
)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file 'best.pt' ada di folder yang sama
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("✅ Model AI Siap!")
except Exception as e:
    st.sidebar.error("⚠️ Model tidak ditemukan!")
    st.stop()

# --- FUNGSI DETEKSI (KHUSUS HELM & ROMPI) ---
def detect_objects(frame):
    # Resize frame agar ringan
    frame = cv2.resize(frame, (854, 480))
    results = model(frame, conf=conf_threshold, verbose=False)
    
    # Init statistik hanya untuk Helm dan Rompi
    stats = {'Helm': 0, 'Rompi': 0}

    # Daftar kelas yang diizinkan (Sesuaikan dengan nama class di model YOLO Anda)
    target_classes = ['Safety-Helmet', 'Safety-Vest']

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label_name = model.names[cls] 
            
            # --- FILTER UTAMA: Lewati jika bukan Helm atau Rompi ---
            if label_name not in target_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Logika Pewarnaan & Hitung
            if label_name == 'Safety-Helmet':
                stats['Helm'] += 1
                color = (0, 255, 0) # Hijau
                label_display = "HELM"
            elif label_name == 'Safety-Vest':
                stats['Rompi'] += 1
                color = (0, 165, 255) # Oranye
                label_display = "ROMPI"
            else:
                color = (255, 255, 255)
                label_display = label_name

            # Gambar Kotak
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Gambar Label Background agar tulisan jelas
            (w, h), _ = cv2.getTextSize(f"{label_display} {int(conf*100)}%", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            
            cv2.putText(frame, f"{label_display} {int(conf*100)}%", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return frame, stats

# --- LOGIKA UTAMA ---

# 1. MODE WEBCAM (SNAPSHOT)
if source_radio == "Ambil Foto (Webcam)":
    st.write("### 📸 Ambil Foto dari Webcam")
    st.info("Sistem akan mendeteksi Helm dan Rompi dari foto yang diambil.")
    
    img_file = st.camera_input("Klik tombol 'Take Photo'")
    
    if img_file is not None:
        image = Image.open(img_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        processed_frame, stats = detect_objects(frame)
        
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Hasil Deteksi", use_column_width=True)
        
        # Tampilkan KPI (Hanya 2 Kolom)
        k1, k2 = st.columns(2)
        k1.metric("⛑️ Helm Terdeteksi", stats['Helm'])
        k2.metric("🦺 Rompi Terdeteksi", stats['Rompi'])

# 2. MODE UPLOAD VIDEO
elif source_radio == "Upload Video":
    st.write("### ▶️ Analisa Video (Helm & Rompi)")
    uploaded_file = st.sidebar.file_uploader("Upload video .mp4", type=['mp4'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        start_button = st.sidebar.button("Mulai Putar")
        
        if start_button:
            st_frame = st.empty()
            # KPI Container di atas video
            k1, k2 = st.columns(2)
            kpi1 = k1.empty()
            kpi2 = k2.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame, stats = detect_objects(frame)
                
                # Update Gambar
                st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                
                # Update Angka Realtime
                kpi1.metric("⛑️ Helm", stats['Helm'])
                kpi2.metric("🦺 Rompi", stats['Rompi'])
            cap.release()

# 3. MODE UPLOAD GAMBAR
elif source_radio == "Upload Gambar":
    st.write("### 🖼️ Analisa Gambar")
    uploaded_file = st.sidebar.file_uploader("Upload gambar", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        processed_frame, stats = detect_objects(frame)
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
        
        k1, k2 = st.columns(2)
        k1.metric("⛑️ Helm", stats['Helm'])
        k2.metric("🦺 Rompi", stats['Rompi'])

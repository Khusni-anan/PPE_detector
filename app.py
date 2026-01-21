import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Dashboard Monitoring K3 - APD",
    page_icon="👷",
    layout="wide"
)

# --- JUDUL & SIDEBAR ---
st.title("👷 Sistem Deteksi Kepatuhan APD")
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
    # Pastikan file 'best.pt' ada di github
    return YOLO('best.pt')

try:
    model = load_model()
    st.sidebar.success("✅ Model AI Siap!")
except Exception as e:
    st.sidebar.error("⚠️ Model tidak ditemukan!")
    st.stop()

# --- FUNGSI DETEKSI (UMUM) ---
def detect_objects(frame):
    # Resize frame
    frame = cv2.resize(frame, (854, 480))
    results = model(frame, conf=conf_threshold, verbose=False)
    
    stats = {'Helm': 0, 'Rompi': 0, 'Gloves': 0, 'Boots': 0}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            conf = float(box.conf[0])
            
            # Pewarnaan & Hitung
            if label_name in ['Safety-Helmet', 'Safety-Vest', 'Safety-Wearpack']:
                color = (0, 255, 0) # Hijau
                if label_name == 'Safety-Helmet': stats['Helm'] += 1
                if label_name == 'Safety-Vest': stats['Rompi'] += 1
            elif label_name in ['Gloves', 'Boots', 'Mask']:
                color = (255, 255, 0) # Kuning
                if label_name == 'Gloves': stats['Gloves'] += 1
                if label_name == 'Boots': stats['Boots'] += 1
            else:
                color = (255, 255, 255) # Putih

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {int(conf*100)}%", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame, stats

# --- LOGIKA UTAMA ---

# 1. MODE WEBCAM (FITUR FOTO) - COCOK BUAT CLOUD
if source_radio == "Ambil Foto (Webcam)":
    st.write("### 📸 Ambil Foto dari Webcam")
    st.info("Karena berjalan di Cloud, sistem menggunakan metode 'Snapshot' (Bukan Video Stream).")
    
    # Widget Kamera Resmi Streamlit (Bekerja di Browser/HP)
    img_file = st.camera_input("Klik tombol 'Take Photo'")
    
    if img_file is not None:
        # Konversi Foto ke Format OpenCV
        image = Image.open(img_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Proses Deteksi
        processed_frame, stats = detect_objects(frame)
        
        # Tampilkan Hasil
        st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Hasil Analisa", use_column_width=True)
        
        # Tampilkan KPI
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⛑️ Helm", stats['Helm'])
        k2.metric("🦺 Rompi", stats['Rompi'])
        k3.metric("🧤 Sarung Tangan", stats['Gloves'])
        k4.metric("🥾 Sepatu", stats['Boots'])

# 2. MODE UPLOAD VIDEO
elif source_radio == "Upload Video":
    st.write("### ▶️ Analisa Video")
    uploaded_file = st.sidebar.file_uploader("Upload video .mp4", type=['mp4'])
    
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        start_button = st.sidebar.button("Mulai Putar")
        
        if start_button:
            st_frame = st.empty()
            k1, k2, k3, k4 = st.columns(4)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                processed_frame, stats = detect_objects(frame)
                
                # Update Gambar & Angka
                st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_column_width=True)
                k1.metric("⛑️ Helm", stats['Helm'])
                k2.metric("🦺 Rompi", stats['Rompi'])
                k3.metric("🧤 Sarung Tangan", stats['Gloves'])
                k4.metric("🥾 Sepatu", stats['Boots'])
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
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("⛑️ Helm", stats['Helm'])
        k2.metric("🦺 Rompi", stats['Rompi'])
        k3.metric("🧤 Sarung Tangan", stats['Gloves'])
        k4.metric("🥾 Sepatu", stats['Boots'])

import streamlit as st
import cv2
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Live K3 Detection", page_icon="👷", layout="wide")
st.title("👷 Live CCTV Kepatuhan APD (Helm & Rompi)")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model tidak ditemukan!")
    st.stop()

# --- CLASS PEMROSES VIDEO (WEBRTC) ---
# Ini adalah inti agar bisa Live Streaming
class VideoProcessor:
    def __init__(self):
        # Setting confidence (bisa diambil dari slider kalau mau canggih, tapi kita hardcode dulu biar simpel)
        self.conf_threshold = 0.5

    def recv(self, frame):
        # 1. Konversi format frame dari WebRTC ke OpenCV
        img = frame.to_ndarray(format="bgr24")

        # 2. Deteksi YOLO
        results = model(img, conf=self.conf_threshold, verbose=False)
        
        target_classes = ['Safety-Helmet', 'Safety-Vest']

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                label_name = model.names[cls]

                # Filter hanya Helm & Rompi
                if label_name not in target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Warna & Label
                if label_name == 'Safety-Helmet':
                    color = (0, 255, 0) # Hijau
                    label_display = f"HELM {int(conf*100)}%"
                else: # Safety-Vest
                    color = (0, 165, 255) # Oranye
                    label_display = f"ROMPI {int(conf*100)}%"

                # Gambar Kotak
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Label Background
                (w, h), _ = cv2.getTextSize(label_display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(img, label_display, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 3. Kembalikan frame yang sudah digambar ke browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- UI UTAMA ---
st.write("### 🎥 Live Camera Feed")
st.info("Pastikan memberikan izin akses kamera pada browser.")

# Panggil Widget WebRTC
webrtc_streamer(
    key="k3-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False}, # Hanya video, tanpa suara
    async_processing=True,
)

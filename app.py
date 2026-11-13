from pathlib import Path
from io import BytesIO
from threading import Lock
import os

import av
import streamlit as st
from PIL import Image, ImageOps
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from ultralytics import YOLO

st.set_page_config(page_title='Wall Crack Detection', page_icon='🧱', layout='wide')
st.title('Wall Crack Detection Using YOLO11')
st.caption('Alat bantu visual, bukan penilaian keamanan atau tingkat bahaya bangunan.')

MODEL_PATH = Path(__file__).resolve().parent / 'models' / 'best.pt'

@st.cache_resource
def load_model(path, modified):
    # Serialize access to the shared Ultralytics model across video and photo sessions.
    return YOLO(path), Lock()

if not MODEL_PATH.is_file():
    st.error('Masukkan best.pt ke folder models/ terlebih dahulu.')
    st.stop()

with st.sidebar:
    st.header('Pengaturan')
    mode = st.radio('Mode deteksi', ['Image Detection', 'Real-Time Detection'])
    conf = st.slider('Confidence threshold', 0.05, 0.95, 0.25, 0.05)
    st.caption('Ambang lebih rendah dapat meningkatkan deteksi keliru.')
    st.write('Model: YOLO11n • CPU')
    if mode == 'Real-Time Detection':
        live_size = st.selectbox('Ukuran inferensi video', [320, 480, 640], index=1)
        camera = st.radio('Kamera', ['Belakang (HP)', 'Depan / webcam'])
        st.caption('Ukuran 480 membantu mengurangi beban CPU. Gunakan 640 untuk retakan kecil jika cukup cepat.')

try:
    model, model_lock = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f'Model tidak dapat dimuat: {exc}')
    st.stop()

def predict(image, threshold, size):
    with model_lock:
        result = model.predict(image, conf=threshold, imgsz=size, device='cpu', verbose=False)[0]
        scores = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
        annotated = result.plot()
    return annotated, scores

if mode == 'Real-Time Detection':
    st.subheader('Real-Time Detection')
    st.write('Tekan START, izinkan akses kamera, lalu arahkan kamera ke permukaan dinding.')
    st.caption('Video dikirim ke komputer/server yang menjalankan aplikasi untuk diproses. Jangan arahkan ke informasi sensitif.')

    def video_callback(frame):
        image = frame.to_ndarray(format='bgr24')
        annotated, _ = predict(image, conf, live_size)
        return av.VideoFrame.from_ndarray(annotated, format='bgr24')

    # STUN for discovery; optional TURN for restrictive networks / cloud hosting.
    ice_servers = [{'urls': ['stun:stun.l.google.com:19302']}]
    turn_url = os.getenv('TURN_URL')
    turn_username = os.getenv('TURN_USERNAME')
    turn_credential = os.getenv('TURN_CREDENTIAL')
    if turn_url and turn_username and turn_credential:
        ice_servers.append({'urls': [turn_url], 'username': turn_username, 'credential': turn_credential})

    facing = 'environment' if camera == 'Belakang (HP)' else 'user'
    webrtc_streamer(
        key=f'wall-crack-live-{facing}',
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={'iceServers': ice_servers},
        media_stream_constraints={
            'video': {
                'facingMode': {'ideal': facing},
                'width': {'ideal': 640},
                'height': {'ideal': 480},
                'frameRate': {'ideal': 10, 'max': 15},
            },
            'audio': False,
        },
        video_frame_callback=video_callback,
        async_processing=True,
    )
    st.info('Jika kamera HP tidak dapat dibuka, pastikan alamat website menggunakan HTTPS dan izin kamera telah diberikan. Jika video tidak tersambung, periksa jaringan atau konfigurasi TURN.')
else:
    st.subheader('Image Detection')
    upload = st.file_uploader('Pilih foto dinding', type=['jpg', 'jpeg', 'png', 'webp'])
    if upload is None:
        st.info('Silakan unggah gambar untuk memulai.')
        st.stop()
    try:
        im = ImageOps.exif_transpose(Image.open(upload)).convert('RGB')
    except Exception:
        st.error('File gambar tidak dapat dibaca.')
        st.stop()
    if im.width * im.height > 24_000_000:
        im.thumbnail((3000, 3000))
    with st.spinner('Mendeteksi retakan...'):
        try:
            annotated_bgr, scores = predict(im, conf, 640)
            annotated = Image.fromarray(annotated_bgr[:, :, ::-1])
        except Exception as exc:
            st.error(f'Prediksi gagal: {exc}')
            st.stop()
    a, b = st.columns(2)
    a.image(im, caption='Gambar asli', use_container_width=True)
    b.image(annotated, caption='Hasil deteksi', use_container_width=True)
    a, b, c = st.columns(3)
    a.metric('Status', 'Retak terdeteksi' if scores else 'Tidak terdeteksi')
    b.metric('Jumlah bounding box', len(scores))
    c.metric('Confidence tertinggi', f'{max(scores):.1%}' if scores else '—')
    if scores:
        st.dataframe([{'Deteksi': i+1, 'Kelas': 'crack', 'Confidence': f'{score:.1%}'} for i, score in enumerate(scores)], hide_index=True)
    else:
        st.info('Tidak ada retakan terdeteksi pada ambang ini. Hal ini tidak membuktikan dinding bebas retak.')
    buffer = BytesIO()
    annotated.save(buffer, format='PNG')
    st.download_button('Unduh hasil deteksi', buffer.getvalue(), 'wall_crack_result.png', 'image/png')

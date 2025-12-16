import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.signal import butter, filtfilt, decimate
import plotly.graph_objects as go
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Grasp-and-Lift EEG Detector", layout="wide")

# --- KONSTANTA (Sesuai Notebook) ---
WINDOW_SIZE = 250
STEP = 50  # Step untuk sliding window saat inferensi (bisa diperkecil untuk resolusi lebih tinggi)
ORIG_FS = 500
TARGET_FS = 250
CHANNELS = 32
EVENT_ORDER = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']
OPTIMAL_THRESHOLDS = {
    'HandStart': 0.60,
    'FirstDigitTouch': 0.68,
    'BothStartLoadPhase': 0.67,
    'LiftOff': 0.57,
    'Replace': 0.26,
    'BothReleased': 0.24
}

# --- DEFINISI CUSTOM LAYER (Wajib untuk load model) ---
@keras.utils.register_keras_serializable()
def se_block(input_tensor, ratio=16):
    """Squeeze and Excitation Block (Re-defined for loading context)"""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    se = layers.Reshape((1, filters))(se)
    x = layers.Multiply()([input_tensor, se])
    return x

# --- FUNGSI PREPROCESSING (Sesuai Notebook) ---
def bandpass_filter(X, fs=500, low=1.0, high=45.0, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, X, axis=0)

def downsample_data(X, orig_fs=500, target_fs=250):
    if target_fs >= orig_fs:
        return X
    factor = int(orig_fs / target_fs)
    return decimate(X, factor, axis=0, zero_phase=True)

def standardize(X, eps=1e-9):
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + eps
    return (X - mu) / sigma

def make_windows(X, window_size=250, step=50):
    T, C = X.shape
    starts = list(range(0, T - window_size + 1, step))
    Xw = np.zeros((len(starts), C, window_size), dtype=np.float32)
    idxs = []
    for i, s in enumerate(starts):
        e = s + window_size
        Xw[i] = X[s:e].T # Transpose agar shape (Channel, Time) sesuai model
        idxs.append(s) # Simpan index waktu mulai (dalam skala downsampled)
    return Xw, np.array(idxs)

# --- LOAD MODEL ---
@st.cache_resource
def load_trained_model(path='best_model.keras'):
    if not os.path.exists(path):
        return None
    try:
        # Load model dengan custom objects jika diperlukan
        model = keras.models.load_model(path, custom_objects={'se_block': se_block})
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

# --- UI UTAMA ---
st.title("🧠 Grasp-and-Lift EEG Event Detection")
st.markdown("""
Aplikasi ini mendeteksi tahapan gerakan tangan (Grasp & Lift) berdasarkan sinyal EEG 32-channel.
Model menggunakan arsitektur **CNN + LSTM + SE Block**.
""")

# Sidebar
st.sidebar.header("Konfigurasi")
model = load_trained_model('best_model.keras')

if model is None:
    st.sidebar.warning("⚠️ File model tidak ditemukan. Menggunakan mode simulasi prediksi (output acak).")

data_source = st.sidebar.radio("Sumber Data:", ["Simulasi (Random EEG)", "Upload CSV"])

# --- LOAD DATA ---
raw_data = None
fs_current = ORIG_FS

if data_source == "Simulasi (Random EEG)":
    if st.sidebar.button("Generate Data Baru"):
        # Buat dummy data: 5 detik @ 500Hz = 2500 sampel, 32 channel
        n_samples = 5000 
        # Membuat sinyal sinus + noise agar terlihat seperti EEG
        t = np.linspace(0, 10, n_samples)
        raw_data = np.zeros((n_samples, CHANNELS))
        for ch in range(CHANNELS):
            raw_data[:, ch] = np.sin(2 * np.pi * (ch+1) * t) * 0.5 + np.random.normal(0, 0.2, n_samples)
        
        st.session_state['raw_data'] = raw_data
    
    if 'raw_data' in st.session_state:
        raw_data = st.session_state['raw_data']
    else:
        st.info("Klik tombol 'Generate Data Baru' di sidebar.")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload file *_data.csv", type=["csv"])
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Ambil hanya kolom data (biasanya dimulai dari kolom ke-1, kolom 0 adalah ID)
            # Asumsi format kompetisi: kolom pertama adalah 'id', sisanya channel
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            # Filter kolom yang benar-benar EEG (Fp1, etc.. biasanya ada 32)
            # Di sini kita ambil semua numerik, lalu pastikan jumlahnya 32
            data_vals = df[numeric_cols].values
            
            # Jika ada kolom ID (biasanya index urut), buang
            if data_vals.shape[1] > 32:
                data_vals = data_vals[:, 1:] # Skip kolom pertama
            
            if data_vals.shape[1] != 32:
                st.error(f"File harus memiliki 32 kolom EEG. Terdeteksi: {data_vals.shape[1]}")
                raw_data = None
            else:
                raw_data = data_vals
        except Exception as e:
            st.error(f"Error membaca file: {e}")

# --- PROCESSING & PREDICTION ---
if raw_data is not None:
    st.subheader("1. Sinyal EEG Input")
    
    # Plot sebagian channel saja agar tidak berat
    channels_to_plot = st.multiselect("Pilih Channel untuk dilihat (Raw)", 
                                      options=[f"Ch{i+1}" for i in range(CHANNELS)], 
                                      default=["Ch1", "Ch2", "Ch3"])
    
    fig_raw = go.Figure()
    x_axis = np.arange(len(raw_data)) / ORIG_FS
    for ch_name in channels_to_plot:
        ch_idx = int(ch_name.replace("Ch", "")) - 1
        fig_raw.add_trace(go.Scatter(x=x_axis, y=raw_data[:, ch_idx], name=ch_name, opacity=0.8))
    fig_raw.update_layout(title="Raw EEG Signal (Time Domain)", xaxis_title="Waktu (detik)", yaxis_title="Amplitudo", height=300)
    st.plotly_chart(fig_raw, use_container_width=True)

    if st.button("Jalankan Prediksi"):
        with st.spinner("Preprocessing & Predicting..."):
            # 1. Preprocessing
            X = raw_data.astype(np.float32)
            
            # Filter
            try:
                X = bandpass_filter(X, fs=ORIG_FS)
            except Exception as e:
                st.warning("Filter gagal, menggunakan raw data.")
            
            # Downsample
            if TARGET_FS != ORIG_FS:
                X = downsample_data(X, orig_fs=ORIG_FS, target_fs=TARGET_FS)
            
            # Standardize
            X = standardize(X)
            
            # Windowing
            Xw, time_idxs = make_windows(X, window_size=WINDOW_SIZE, step=STEP)
            
            if len(Xw) == 0:
                st.error("Data terlalu pendek untuk dibuat window.")
            else:
                # 2. Prediction
                if model is not None:
                    preds = model.predict(Xw)
                else:
                    # Dummy prediction jika model tidak ada
                    preds = np.random.rand(len(Xw), 6)
                
                # Mapping waktu prediksi kembali ke detik
                # time_idxs adalah index awal window di data yang sudah di-downsample
                # Karena window merepresentasikan satu segmen, kita plot titik waktunya di tengah atau akhir window
                time_in_seconds = (time_idxs + WINDOW_SIZE) / TARGET_FS
                
                # 3. Visualization Results
                st.subheader("2. Hasil Prediksi Probabilitas")
                
                fig_pred = go.Figure()
                
                # Plot probabilitas tiap event
                for i, event_name in enumerate(EVENT_ORDER):
                    fig_pred.add_trace(go.Scatter(
                        x=time_in_seconds, 
                        y=preds[:, i], 
                        name=event_name,
                        mode='lines+markers'
                    ))
                    
                    # Tambahkan garis threshold (opsional, putus-putus tipis)
                    th = OPTIMAL_THRESHOLDS[event_name]
                    fig_pred.add_shape(
                        type="line",
                        x0=time_in_seconds[0], x1=time_in_seconds[-1],
                        y0=th, y1=th,
                        line=dict(color="gray", width=1, dash="dot"),
                        opacity=0.3
                    )

                fig_pred.update_layout(
                    title="Probabilitas Event Seiring Waktu",
                    xaxis_title="Waktu (detik)",
                    yaxis_title="Probabilitas",
                    yaxis_range=[0, 1.1],
                    height=500
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # 4. Binary Output Table (Active Events)
                st.subheader("3. Deteksi Event (Thresholded)")
                
                # Buat DataFrame hasil
                res_df = pd.DataFrame(preds, columns=EVENT_ORDER)
                res_df.insert(0, "Waktu (s)", time_in_seconds)
                
                # Apply thresholds untuk display teks
                binary_res = res_df.copy()
                for evt in EVENT_ORDER:
                    th = OPTIMAL_THRESHOLDS[evt]
                    # Ubah jadi string "ACTIVE" atau ""
                    binary_res[evt] = binary_res[evt].apply(lambda x: "✅ ACTIVE" if x > th else "-")
                
                st.dataframe(binary_res.style.applymap(
                    lambda v: 'background-color: #d4edda; color: green;' if "ACTIVE" in str(v) else '', 
                    subset=EVENT_ORDER
                ))

else:
    st.info("Silakan pilih sumber data atau upload file CSV untuk memulai.")
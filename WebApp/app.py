import streamlit as st
import os
import pickle
import pandas as pd

# ==========================================
# 1. KONFIGURASI HALAMAN WEB & FAVICON
# ==========================================
st.set_page_config(
    page_title="MotoGP Tyre Predictor AI",
    page_icon="🛞",
    layout="centered"
)

# Jalur Folder Relatif (Aman untuk Lokal Windows & Cloud Server)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "Datasets", "Dataset_Final_XGBoost.csv")
MODEL_DIR = os.path.join(BASE_DIR, "Models")

# ==========================================
# UI DESIGNER: DASHBOARD TELEMETRY PIT-BOARD
# ==========================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">

<style>
/* ---------- TOKEN WARNA ---------- */
:root{
    --asphalt-900:#0B0E11;
    --asphalt-800:#14181D;
    --asphalt-700:#1C2128;
    --line-faint: rgba(255,255,255,0.06);
    --text-main:#F2F1EC;
    --text-muted:#8A9099;
    --racing-blue:#0057B8;
    --racing-blue-glow: rgba(0,87,184,0.35);
    --pit-yellow:#FFD400;
    --flag-red:#E63946;
    --compound-medium:#FFC72C;
    --compound-hard:#E8E8E8;
    --compound-wet:#00A3FF;
}

/* ---------- BASE ---------- */
html, body, [class*="css"]{
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}
.stApp{
    background:
        repeating-linear-gradient(115deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 4px),
        radial-gradient(circle at 15% -10%, #1a2027 0%, var(--asphalt-900) 45%);
}
h1,h2,h3, .rajdhani{
    font-family:'Rajdhani', sans-serif;
    letter-spacing: 0.5px;
}
#MainMenu, footer, header{visibility:hidden;}

/* ---------- HEADER / PIT BOARD ---------- */
.pitboard{
    position:relative;
    background: linear-gradient(180deg, var(--asphalt-800), var(--asphalt-700));
    border:1px solid var(--line-faint);
    border-left: 6px solid var(--racing-blue);
    border-radius: 6px;
    padding: 22px 26px;
    margin-bottom: 22px;
    overflow:hidden;
}
.pitboard::after{
    content:"";
    position:absolute; top:0; right:0; bottom:0; width:46px;
    background: repeating-linear-gradient(45deg, #0d0f12 0 8px, #e7e7e2 8px 16px);
    opacity:0.12;
}
.pitboard h1{ color:#fff; margin:0; font-size:34px; font-weight:700; }
.pitboard h1 span.hl{ color: var(--pit-yellow); }
.pitboard h1 span.blue{ color:#5AA9FF; }
.pitboard .sub{
    font-family:'Rajdhani', sans-serif;
    color: var(--text-muted);
    font-size:14px; letter-spacing:2px; text-transform:uppercase;
    margin-top:6px;
}
.pitboard .lede{
    color: var(--text-muted); font-size:14.5px; margin-top:14px; max-width:560px;
    font-family:'Inter', sans-serif;
}

/* ---------- SECTION LABEL (PLAQUE) ---------- */
.plaque{
    display:inline-block;
    font-family:'Rajdhani', sans-serif;
    font-weight:700; font-size:13px; letter-spacing:3px; text-transform:uppercase;
    color: var(--asphalt-900);
    background: var(--pit-yellow);
    padding: 4px 12px;
    border-radius: 3px;
    margin-bottom: 10px;
}
.plaque.blue{ background: var(--racing-blue); color:#fff; }

/* ---------- PANEL ---------- */
.panel{
    background: var(--asphalt-800);
    border: 1px solid var(--line-faint);
    border-radius: 8px;
    padding: 18px 20px 6px 20px;
    margin-bottom: 18px;
}

/* ---------- STREAMLIT WIDGET RESKIN ---------- */
div[data-baseweb="select"] > div{
    background-color: var(--asphalt-700) !important;
    border-color: var(--line-faint) !important;
    border-radius:6px !important;
}
.stSlider > div > div > div > div{ background: var(--racing-blue) !important; }
.stSlider [data-baseweb="slider"] > div:nth-child(2){ background: var(--racing-blue) !important; }
label, .stMarkdown p{ color: var(--text-main) !important; }
hr{ border-color: var(--line-faint) !important; }

/* ---------- BUTTON: LIGHTS OUT ---------- */
.stButton > button{
    background: linear-gradient(180deg, var(--flag-red), #B3212C);
    color:#fff; border:none; border-radius:6px;
    font-family:'Rajdhani', sans-serif; font-weight:700; letter-spacing:2px;
    font-size:17px; text-transform:uppercase;
    padding: 14px 0;
    box-shadow: 0 4px 14px rgba(230,57,70,0.35);
    transition: all 0.15s ease;
}
.stButton > button:hover{
    background: linear-gradient(180deg, #ff4b58, var(--flag-red));
    box-shadow: 0 6px 18px rgba(230,57,70,0.5);
    transform: translateY(-1px);
}

/* ---------- TYRE GAUGE (SIGNATURE ELEMENT) ---------- */
.tyre-wrap{ text-align:center; padding: 6px 0 18px 0; }
.tyre-ring{
    width:128px; height:128px; margin:0 auto 12px auto;
    border-radius:50%;
    background: repeating-conic-gradient(#111417 0deg 6deg, #1c2128 6deg 12deg);
    display:flex; align-items:center; justify-content:center;
    box-shadow: inset 0 0 0 6px #05070a, 0 0 0 1px var(--line-faint);
}
.tyre-core{
    width:78px; height:78px; border-radius:50%;
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    font-family:'Rajdhani', sans-serif; font-weight:700;
    box-shadow: inset 0 2px 6px rgba(0,0,0,0.25);
}
.tyre-core .letter{ font-size:26px; line-height:1; }
.tyre-core .compound{ font-size:10.5px; letter-spacing:1px; margin-top:2px; }
.tyre-soft   .tyre-core{ background: var(--flag-red); color:#fff; }
.tyre-medium .tyre-core{ background: var(--compound-medium); color:#1a1400; }
.tyre-hard   .tyre-core{ background: var(--compound-hard); color:#1a1a1a; }
.tyre-wet    .tyre-core{ background: var(--compound-wet); color:#ffffff; }
.tyre-label{
    font-family:'Rajdhani', sans-serif; font-size:13px; letter-spacing:2px;
    color: var(--text-muted); text-transform:uppercase; margin-bottom:2px;
}
.tyre-note{ font-size:12.5px; color: var(--text-muted); margin-top:4px; }

/* ---------- BADGE / FOOTER ---------- */
.accuracy-badge{
    display:inline-flex; align-items:center; gap:8px;
    background: rgba(0,87,184,0.12); border:1px solid var(--racing-blue-glow);
    color:#8FC1FF; font-family:'Rajdhani', sans-serif; font-weight:600;
    letter-spacing:1px; padding:6px 14px; border-radius:20px; font-size:13px;
}
            
/* ---------- SLIDER OVERRIDE (ANTI AUTO-DARK BROWSER) ---------- */
div[data-baseweb="slider"] {
    margin-top: 10px !important;
}

/* Warna Angka Minimum, Maksimum, dan Angka Slider yang Dipilih */
div[data-testid="stSliderTickBar"] + div,
div[aria-valuenow] + div,
div[data-baseweb="slider"] div {
    color: #F2F1EC !important;
    background-color: transparent !important;
}

/* Mengubah Kotak Tooltip Angka Slider Biar Gelap & Teks Putih */
div[data-baseweb="tooltip"] > div {
    background-color: #1C2128 !important;
    color: #FFD400 !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    font-weight: bold !important;
}

/* Mengubah Lingkaran Pegangan Slider (Thumb) Jadi Merah Balap */
div[role="slider"] {
    background-color: #E63946 !important;
    border: 2px solid #FFFFFF !important;
    box-shadow: 0 0 10px rgba(230, 57, 70, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
    <div class="pitboard">
        <h1>MOTO<span class="hl">TYRE</span><span class="blue">.AI</span></h1>
        <div class="sub">// Pit Wall Strategy Simulator — Season 2026</div>
        <p class="lede">Halo Fans! Atur Skenario Kondisi Sirkuit favoritmu di bawah ini,
        dan biarkan MOTOTYRE membaca datanya seperti insinyur di paddock untuk merekomendasikan
        kompon ban terbaik untuk tim kesayanganmu.</p>
    </div>
""", unsafe_allow_html=True)

# Fungsi Memuat Model Keamanan
@st.cache_resource
def load_ai_models():
    try:
        model_front = pickle.load(open(os.path.join(MODEL_DIR, "xgb_model_front.pkl"), "rb"))
        model_rear = pickle.load(open(os.path.join(MODEL_DIR, "xgb_model_rear.pkl"), "rb"))
        feature_encoders = pickle.load(open(os.path.join(MODEL_DIR, "feature_encoders.pkl"), "rb"))
        target_encoder = pickle.load(open(os.path.join(MODEL_DIR, "target_encoder.pkl"), "rb"))
        feature_columns = pickle.load(open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb"))
        return model_front, model_rear, feature_encoders, target_encoder, feature_columns
    except Exception as e:
        st.error(f"Gagal memuat otak AI (.pkl). Error: {e}")
        return None

# Ekstrak Seluruh Sirkuit dari CSV Master
@st.cache_data
def ekstrak_data_master(path):
    if not os.path.exists(path):
        st.error(f"File dataset tidak ditemukan di {path}")
        return {}, [], []
    
    df = pd.read_csv(path)
    
    # 1. Ambil SEMUA jenis cuaca asli dari dataset (Clear, Cloudy, dll)
    list_cuaca = sorted(df['Weather'].dropna().unique().tolist())
    
    # 2. Tambahkan opsi 'Rain (Wet)' di akhir list jika belum ada
    if not any("rain" in c.lower() or "wet" in c.lower() for c in list_cuaca):
        list_cuaca.append("Rain (Wet)")
        
    # 3. Ekstrak sirkuit
    kamus_sirkuit = {}
    grouped = df.groupby('Circuit').last().reset_index()
    
    for _, row in grouped.iterrows():
        nama_sirkuit = row['Circuit']
        kamus_sirkuit[nama_sirkuit] = {
            'Surface_Abrasion': int(row['Surface_Abrasion']),
            'Grip_Level': int(row['Grip_Level']),
            'Layout_Type': str(row['Layout_Type']),
            'Symmetry': str(row['Symmetry']),
            'Dominant': str(row['Dominant']),
            'Round': int(row['Round'])
        }
        
    list_sirkuit = sorted(list(kamus_sirkuit.keys()))
    return kamus_sirkuit, list_sirkuit, list_cuaca

models = load_ai_models()
kamus_sirkuit, list_sirkuit, list_cuaca = ekstrak_data_master(DATASET_PATH)

if models is not None and kamus_sirkuit:
    model_front, model_rear, feature_encoders, target_encoder, feature_columns = models

    # SIDEBAR INFO
    # SIDEBAR INFO
    with st.sidebar:
        st.markdown('<div class="plaque blue">Tentang Model</div>', unsafe_allow_html=True)
        st.markdown("""
        **MotoTyre.AI** memprediksi rekomendasi kompon ban depan & belakang
        menggunakan **XGBoost**, dilatih dari data historis balapan MotoGP.""")
        st.markdown(
            '<div class="accuracy-badge">⚙️ Akurasi Model: 85–88%</div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption("Dibuat untuk keperluan riset & simulasi fans. Bukan alat pengganti keputusan tim/insinyur resmi.")

    KAMUS_MOTOR = {
        'Ducati': {'Engine_Config': 'V4', 'Description': 'First Team'},
        'KTM': {'Engine_Config': 'V4', 'Description': 'First Team'},
        'Aprilia': {'Engine_Config': 'V4', 'Description': 'First Team'},
        'Yamaha': {'Engine_Config': 'Inline-4', 'Description': 'First Team'},
        'Honda': {'Engine_Config': 'V4', 'Description': 'First Team'},
        'Gresini (Ducati)': {'Engine_Config': 'V4', 'Description': 'Satellite Team'},
        'VR46 (Ducati)': {'Engine_Config': 'V4', 'Description': 'Satellite Team'},
        'Pramac (Yamaha)': {'Engine_Config': 'Inline-4', 'Description': 'Satellite Team'},
        'Trackhouse (Aprilia)': {'Engine_Config': 'V4', 'Description': 'Satellite Team'},
        'LCR (Honda)': {'Engine_Config': 'V4', 'Description': 'Satellite Team'},
        'Tech3 (KTM)': {'Engine_Config': 'V4', 'Description': 'Satellite Team'}
    }

    # AREA INTERFACE
    st.markdown('<div class="plaque">Track Setup</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        pilihan_sirkuit = st.selectbox("🏟️ Pilih Sirkuit Balapan:", list_sirkuit)
        pilihan_motor = st.selectbox("🏍️ Pilih Tim / Motor Favorit:", list(KAMUS_MOTOR.keys()))
        # DROPDOWN CUACA LENGKAP DARI CSV + OPSI RAIN
        pilihan_cuaca = st.selectbox("🌤️ Kondisi Cuaca di Sirkuit:", list_cuaca)
    with col2:
        suhu_aspal = st.slider("🌡️ Suhu Lintasan Aspal (°C):", min_value=15, max_value=60, value=35)
        suhu_udara = st.slider("🌡️ Suhu Udara Sekitar (°C):", min_value=10, max_value=45, value=27)
        kelembapan = st.slider("💧 Tingkat Kelembapan Udara (%):", min_value=10, max_value=100, value=50)

    st.markdown('</div>', unsafe_allow_html=True)

    # PROSES TOMBOL
    hitung = st.button("🚦 HITUNG REKOMENDASI BAN AI", use_container_width=True)

    if hitung:
        with st.spinner("Sistem sedang menganalisis telemetri lintasan..."):

            # KONDISI ABSOLUT HUJAN / RAIN
            if "rain" in pilihan_cuaca.lower() or "wet" in pilihan_cuaca.lower():
                ban_depan = "WET"
                ban_belakang = "WET"
                catatan_footer = "🌧️ Skenario Hujan (Wet): Sesuai regulasi MotoGP, pembalap mutlak menggunakan Michelin Wet Tyres dengan alur pembuangan air."

            else:
                # KONDISI KERING DIHITUNG OLEH XGBOOST
                info_trek = kamus_sirkuit[pilihan_sirkuit]
                nama_konstruktor = pilihan_motor.split()[0] if '(' in pilihan_motor else pilihan_motor
                info_motor = KAMUS_MOTOR[pilihan_motor]

                data_fans = {
                    'Circuit': pilihan_sirkuit, 'Year': 2026, 'Round': info_trek['Round'],
                    'Track': 'Dry', 'Ground': suhu_aspal, 'Humidity': kelembapan, 'Weather': pilihan_cuaca,
                    'Air Temp': suhu_udara, 'Surface_Abrasion': info_trek['Surface_Abrasion'], 'Grip_Level': info_trek['Grip_Level'],
                    'Layout_Type': info_trek['Layout_Type'], 'Symmetry': info_trek['Symmetry'], 'Dominant': info_trek['Dominant'],
                    'Constructor': nama_konstruktor, 'Engine_Config': info_motor['Engine_Config'], 'Description': info_motor['Description']
                }

                df_fans = pd.DataFrame([data_fans])
                df_encoded = df_fans.copy()
                for col, encoder in feature_encoders.items():
                    if col in df_encoded.columns:
                        df_encoded[col] = encoder.transform(df_encoded[col].astype(str))

                df_encoded = df_encoded[feature_columns]
                pred_f_num = model_front.predict(df_encoded)
                pred_r_num = model_rear.predict(df_encoded)

                ban_depan = target_encoder.inverse_transform(pred_f_num)[0]
                ban_belakang = target_encoder.inverse_transform(pred_r_num)[0]
                catatan_footer = "🎯 Analisis berbasis data historis riil murni balapan kering · Akurasi XGBoost: 85–88%"

            # DISPLAY VISUAL TYRE GAUGE
            def render_tyre(label_posisi, kompon):
                k = kompon.lower()
                css_class = f"tyre-{k}" if k in ("soft", "medium", "hard", "wet") else "tyre-medium"
                deskripsi = {
                    "soft":   "Grip instan, cepat menurun — cocok stint pendek.",
                    "medium": "Keseimbangan grip & daya tahan.",
                    "hard":   "Daya tahan tinggi, ideal balapan panjang.",
                    "wet":    "Alur basah khusus — evakuasi air maksimal."
                }.get(k, "")
                st.markdown(f"""
                    <div class="tyre-wrap {css_class}">
                        <div class="tyre-label">{label_posisi}</div>
                        <div class="tyre-ring">
                            <div class="tyre-core">
                                <div class="letter">{kompon[0].upper()}</div>
                                <div class="compound">{kompon.upper()}</div>
                            </div>
                        </div>
                        <div class="tyre-note">{deskripsi}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('<div class="plaque blue">Hasil Rekomendasi</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            col_ban_f, col_ban_r = st.columns(2)
            with col_ban_f:
                render_tyre("Ban Depan · Front", ban_depan)
            with col_ban_r:
                render_tyre("Ban Belakang · Rear", ban_belakang)

            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="accuracy-badge">{catatan_footer}</div>',
                unsafe_allow_html=True
            )
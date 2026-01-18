import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import requests
import io
import json
import os
from PIL import Image
from datetime import datetime

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Turkeller Surfer Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS (HER ŞEY KOYU GRİ/ANTRASİT) ---
st.markdown("""
    <style>
    /* TÜM SAYFA - KOYU GRİ ARKAPLAN */
    .main {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        color: white !important;
        padding: 1rem;
    }
    
    /* SIDEBAR - ANTRASİT */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%) !important;
        color: white !important;
    }
    
    /* TÜM YAZILAR BEYAZ */
    .main *,
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* INPUT ALANLARI - KARBON SİYAHI */
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #0d141c !important;
        color: white !important;
        border: 1px solid #34495e !important;
        border-radius: 8px !important;
        font-size: 16px !important;
        height: 45px !important;
        padding: 5px 10px !important;
    }
    
    /* BUTONLAR - GRİ TONLARI */
    .stButton > button {
        background: linear-gradient(135deg, #5d6d7e 0%, #34495e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        height: 50px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin: 5px 0 !important;
        width: 100% !important;
    }
    
    /* BUTON HOVER - DAHA AÇIK GRİ */
    .stButton > button:hover {
        background: linear-gradient(135deg, #7f8c8d 0%, #5d6d7e 100%) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    /* KARTLAR - ANTRASİT */
    .stat-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #5d6d7e !important;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    .ai-card {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%) !important;
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #5d6d7e !important;
        margin: 10px 0;
    }
    
    /* SLIDER - ANTRASİT */
    .stSlider > div > div {
        background: #1a252f !important;
    }
    
    .stSlider > div > div > div {
        background: #7f8c8d !important;
    }
    
    /* DIVIDER - GRİ */
    hr {
        border-color: #5d6d7e !important;
        margin: 20px 0 !important;
    }
    
    /* EXPANDER - ANTRASİT */
    .streamlit-expanderHeader {
        background: #2c3e50 !important;
        color: white !important;
        border: 1px solid #5d6d7e !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background: #1a252f !important;
        border: 1px solid #5d6d7e !important;
        border-radius: 0 0 8px 8px !important;
    }
    
    /* METRIC KARTLARI - ANTRASİT */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%) !important;
        border: 1px solid #5d6d7e !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    
    /* PLOTLY GRAFİK ARKAPLANI - KOYU */
    .js-plotly-plot {
        background: #1a252f !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* MOBİL UYUMLULUK */
    @media (max-width: 768px) {
        .stButton>button {
            height: 55px !important;
            font-size: 18px !important;
        }
        .stat-card {
            padding: 15px;
            margin: 8px 0;
        }
    }
    
    /* LABELLER - AÇIK GRİ */
    label {
        color: #bdc3c7 !important;
        font-weight: 500 !important;
    }
    
    /* ALERT MESAJLARI - ANTRASİT */
    .stAlert {
        background: #2c3e50 !important;
        border: 1px solid #5d6d7e !important;
        border-radius: 8px !important;
    }
    
    /* CAPTION - AÇIK GRİ */
    .stCaption {
        color: #95a5a6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- GİRİŞ KONTROLÜ ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Turkeller Surfer Pro")
        st.markdown("---")
        pwd = st.text_input("**Erişim Şifresi**", type="password", key="login_pwd")
        
        if st.button("🚀 Giriş Yap", use_container_width=True) or (pwd and pwd == "altin2026"):
            if pwd == "altin2026":
                st.session_state["authenticated"] = True
                st.rerun()
            elif pwd:
                st.error("❌ Hatalı şifre!")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# --- VERİTABANI FONKSİYONLARI ---
DB_FILE = "kayitli_yerler.json"
AI_REPORTS_FILE = "ai_analiz_raporlari.json"

def yerleri_yukle():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w", encoding="utf-8") as f: 
            json.dump([], f)
        return []
    try:
        with open(DB_FILE, "r", encoding="utf-8") as f: 
            return json.load(f)
    except:
        return []

def ai_raporlari_yukle():
    if not os.path.exists(AI_REPORTS_FILE):
        return []
    try:
        with open(AI_REPORTS_FILE, "r", encoding="utf-8") as f:
            content = f.read()
            if content.strip():
                reports = content.split("---\n")
                return [json.loads(r) for r in reports if r.strip()]
            return []
    except:
        return []

def yer_kaydet(isim, lat, lon):
    yerler = yerleri_yukle()
    yerler.append({
        "isim": isim,
        "lat": round(float(lat), 6),
        "lon": round(float(lon), 6)
    })
    with open(DB_FILE, "w", encoding="utf-8") as f: 
        json.dump(yerler, f, ensure_ascii=False, indent=2)

def yer_sil(index):
    yerler = yerleri_yukle()
    if 0 <= index < len(yerler):
        del yerler[index]
        with open(DB_FILE, "w", encoding="utf-8") as f: 
            json.dump(yerler, f, ensure_ascii=False, indent=2)
        return True
    return False

def ai_rapor_kaydet(rapor):
    try:
        with open(AI_REPORTS_FILE, "a", encoding="utf-8") as f:
            json.dump(rapor, f, ensure_ascii=False, indent=2)
            f.write("\n---\n")
        return True
    except:
        return False

# --- SESSION STATE ---
if 'lat' not in st.session_state: 
    st.session_state.lat = 40.104844
if 'lon' not in st.session_state: 
    st.session_state.lon = 27.769064
if 'Z_data' not in st.session_state: 
    st.session_state.Z_data = None
if 'X_data' not in st.session_state: 
    st.session_state.X_data = None
if 'Y_data' not in st.session_state: 
    st.session_state.Y_data = None

# --- YAN MENÜ (KONTROL PANELİ) ---
with st.sidebar:
    st.markdown("## 🎮 Kontrol Paneli")
    
    lat_input = st.number_input("**Enlem**", value=float(st.session_state.lat), format="%.6f", key="lat_in")
    lon_input = st.number_input("**Boylam**", value=float(st.session_state.lon), format="%.6f", key="lon_in")
    
    st.session_state.lat = lat_input
    st.session_state.lon = lon_input
    
    cap = st.slider("**Tarama Çapı (m)**", 20, 300, 50, key="cap_slider")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        analiz_butonu = st.button("🔍 ANALİZ", use_container_width=True, type="primary")
    with col2:
        ai_yorum_butonu = st.button("🤖 AI YORUM", use_container_width=True, 
                                   disabled=st.session_state.Z_data is None,
                                   type="secondary")
    
    st.markdown("---")
    
    maps_url = f"https://www.google.com/maps/search/?api=1&query={st.session_state.lat},{st.session_state.lon}"
    st.markdown(f"""
    <a href="{maps_url}" target="_blank">
        <button style="
            width: 100%;
            height: 50px;
            background: linear-gradient(135deg, #5d6d7e 0%, #34495e 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        ">
        <span>🌍</span>
        <span>HARİTADA GÖSTER</span>
        </button>
    </a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 📍 Kayıtlı Yerler")
    yerler = yerleri_yukle()
    
    if not yerler:
        st.info("Henüz kayıtlı yer yok")
    else:
        for i, y in enumerate(yerler):
            col_btn, col_del = st.columns([4, 1])
            with col_btn:
                if st.button(f"📍 {y['isim']}", key=f"yer_{i}", use_container_width=True):
                    st.session_state.lat = y['lat']
                    st.session_state.lon = y['lon']
                    st.experimental_rerun()
            with col_del:
                if st.button("🗑️", key=f"sil_{i}"):
                    if yer_sil(i): 
                        st.experimental_rerun()
    
    with st.expander("➕ Yeni Yer Kaydet", expanded=False):
        kayit_isim = st.text_input("Yer İsmi", placeholder="Örn: İşyeri")
        if st.button("💾 Kaydet", use_container_width=True):
            if kayit_isim.strip():
                yer_kaydet(kayit_isim, st.session_state.lat, st.session_state.lon)
                st.success(f"✅ '{kayit_isim}' kaydedildi!")
                st.experimental_rerun()
    
    st.markdown("---")
    st.caption("🛰️ Turkeller Surfer Pro v3.0")

# --- ANA EKRAN ---
st.markdown("# 🛰️ Turkeller Surfer Pro")
st.markdown("### 📍 Mevcut Konum")

col_loc1, col_loc2, col_loc3 = st.columns(3)
with col_loc1:
    st.markdown(f'''
    <div class="stat-card">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">ENLEM</div>
        <div style="font-size: 24px; font-weight: bold;">{st.session_state.lat:.6f}</div>
    </div>
    ''', unsafe_allow_html=True)
with col_loc2:
    st.markdown(f'''
    <div class="stat-card">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">BOYLAM</div>
        <div style="font-size: 24px; font-weight: bold;">{st.session_state.lon:.6f}</div>
    </div>
    ''', unsafe_allow_html=True)
with col_loc3:
    st.markdown(f'''
    <div class="stat-card">
        <div style="font-size: 14px; opacity: 0.9; margin-bottom: 10px;">TARAMA ÇAPI</div>
        <div style="font-size: 24px; font-weight: bold;">{cap} m</div>
    </div>
    ''', unsafe_allow_html=True)

# --- ANALİZ KODU ---
if analiz_butonu:
    CLIENT_ID = 'sh-8334dee7-cd0a-412e-8278-ceda2e981f0d'
    CLIENT_SECRET = 'QhUU1AbK8oBk8zBFvjQ0DIL3wjPEdVKN'

    def get_token():
        try:
            auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
            response = requests.post(auth_url, data=data, timeout=30)
            if response.status_code == 200:
                return response.json().get('access_token')
            return None
        except:
            return None

    token = get_token()
    if token:
        with st.spinner('🛰️ Veri çekiliyor...'):
            try:
                lat_f = cap / 111320.0
                lon_f = cap / (40075000.0 * math.cos(math.radians(st.session_state.lat)) / 360.0)
                bbox = [
                    st.session_state.lon - lon_f,
                    st.session_state.lat - lat_f,
                    st.session_state.lon + lon_f,
                    st.session_state.lat + lat_f
                ]
                
                evalscript = """function setup() { return { input: ["VV"], output: { id: "default", bands: 1, sampleType: "FLOAT32" } }; } function evaluatePixel(sample) { return [sample.VV]; }"""
                payload = {
                    "input": { 
                        "bounds": { 
                            "bbox": bbox, 
                            "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"} 
                        }, 
                        "data": [{"type": "sentinel-1-grd"}] 
                    },
                    "output": { 
                        "width": 120, 
                        "height": 120, 
                        "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}] 
                    },
                    "evalscript": evalscript
                }
                
                res = requests.post(
                    "https://sh.dataspace.copernicus.eu/api/v1/process", 
                    headers={"Authorization": f"Bearer {token}"}, 
                    json=payload,
                    timeout=60
                )
                
                if res.status_code == 200:
                    Z = np.array(Image.open(io.BytesIO(res.content)))
                    X, Y = np.meshgrid(
                        np.linspace(bbox[0], bbox[2], 120),
                        np.linspace(bbox[1], bbox[3], 120)
                    )
                    
                    st.session_state.Z_data = Z
                    st.session_state.X_data = X
                    st.session_state.Y_data = Y

                    fig = go.Figure(data=[go.Surface(
                        z=Z, 
                        x=X, 
                        y=Y, 
                        colorscale='Viridis',
                        hovertemplate=(
                            '<b>Boylam</b>: %{x:.6f}<br>' +
                            '<b>Enlem</b>: %{y:.6f}<br>' +
                            '<b>VV Değeri</b>: %{z:.4f}<br>' +
                            '<extra></extra>'
                        ),
                        colorbar=dict(
                            title="VV Değeri",
                            orientation='h', 
                            y=-0.1,
                            len=0.8
                        )
                    )])
                    
                    fig.update_layout(
                        scene=dict(
                            aspectratio=dict(x=1, y=1, z=0.5),
                            xaxis_title="Boylam",
                            yaxis_title="Enlem",
                            zaxis_title="VV Değeri",
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=0.8)
                            )
                        ),
                        margin=dict(l=0, r=0, b=0, t=40),
                        height=500,
                        title=dict(
                            text=f"Sentinel-1 SAR Verisi - {cap}m Çap",
                            font=dict(size=18)
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("📊 Detaylı İstatistikler", expanded=True):
                        if Z.size > 0:
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Ortalama", f"{np.mean(Z):.4f}")
                            with col2:
                                st.metric("Maksimum", f"{np.max(Z):.4f}")
                            with col3:
                                st.metric("Minimum", f"{np.min(Z):.4f}")
                            with col4:
                                st.metric("Standart Sapma", f"{np.std(Z):.4f}")
                            
                            st.subheader("📈 Değer Dağılımı")
                            hist_fig = go.Figure(data=[go.Histogram(
                                x=Z.flatten(),
                                nbinsx=30,
                                marker_color='#7f8c8d'
                            )])
                            hist_fig.update_layout(
                                title="VV Değerleri Histogramı",
                                xaxis_title="VV Değeri",
                                yaxis_title="Frekans",
                                height=300
                            )
                            st.plotly_chart(hist_fig, use_container_width=True)
                            
                    st.success("✅ Analiz tamamlandı!")
                    
                else:
                    st.error(f"❌ Veri alınamadı! Hata kodu: {res.status_code}")
                    
            except Exception as e:
                st.error(f"❌ İstek hatası: {str(e)}")
    else:
        st.error("❌ Token alınamadı!")

# --- YEREL AI FONKSİYONU ---
def yerel_ai_analizi(Z_data, lat, lon, cap):
    Z_clean = Z_data[~np.isnan(Z_data)]
    if len(Z_clean) == 0:
        return "⚠️ HATA: Geçerli veri yok!"
    
    mean_val = np.mean(Z_clean)
    max_val = np.max(Z_clean)
    min_val = np.min(Z_clean)
    std_val = np.std(Z_clean)
    
    def safe_log10(x):
        return 10 * math.log10(x) if x > 0 else -100
    
    mean_db = safe_log10(mean_val)
    
    if mean_db > -5:
        yuzey = "🚨 ÇOK GÜÇLÜ YANSITICI (Şehir/Metal yapılar)"
    elif mean_db > -10:
        yuzey = "🏢 GÜÇLÜ YANSITICI (Orman/Tarım arazisi)"
    elif mean_db > -15:
        yuzey = "🌾 ORTA YANSITICI (Açık arazi/Çıplak toprak)"
    elif mean_db > -20:
        yuzey = "🏜️ ZAYIF YANSITICI (Düz yüzey)"
    else:
        yuzey = "🌊 ÇOK ZAYIF YANSITICI (Su yüzeyi)"
    
    rapor = f"""
    ## 🤖 AI ANALİZ RAPORU
    
    **📍 Konum Bilgileri:**
    - Enlem: {lat:.6f}
    - Boylam: {lon:.6f}
    - Çap: {cap} metre
    - Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}
    
    **📊 İstatistikler:**
    - Ortalama VV Değeri: {mean_val:.4f}
    - Yaklaşık dB: {mean_db:.1f} dB
    - Maksimum: {max_val:.4f}
    - Minimum: {min_val:.4f}
    - Standart Sapma: {std_val:.4f}
    
    **🔍 Analiz Sonuçları:**
    1. **Yüzey Tipi:** {yuzey}
    2. **Topoğrafya:** {'⛰️ Engebeli arazi' if std_val > 0.3 else '🏞️ Hafif engebeli' if std_val > 0.15 else '🏙️ Düz arazi'}
    3. **Değişim:** {'🎯 Yüksek değişim' if (max_val-min_val) > 0.5 else '📊 Orta değişim' if (max_val-min_val) > 0.2 else '🔄 Düşük değişim'}
    
    **💡 Öneriler:**
    - Veri kalitesi: {'✅ İYİ' if std_val < 0.5 else '⚠️ ORTA' if std_val < 1 else '❌ DÜŞÜK'}
    - Örnekleme: {'✅ YETERLİ' if cap < 100 else '⚠️ SINIRLI'}
    """
    
    return rapor

# AI YORUM BUTONU
if ai_yorum_butonu:
    if st.session_state.Z_data is None:
        st.warning("⚠️ Önce 'ANALİZ' butonuyla veri çekmelisiniz!")
    else:
        with st.spinner('🤖 AI analiz yapıyor...'):
            ai_sonuc = yerel_ai_analizi(
                st.session_state.Z_data, 
                st.session_state.lat, 
                st.session_state.lon, 
                cap
            )
            
            st.markdown("---")
            st.markdown("### 🤖 AI Analiz Sonucu")
            st.markdown(f'<div class="ai-card">{ai_sonuc}</div>', unsafe_allow_html=True)
            
            st.markdown("---")
            col_s1, col_s2 = st.columns([3, 1])
            with col_s1:
                rapor_adi = st.text_input(
                    "Rapor Adı", 
                    value=f"AI_Analiz_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}_{datetime.now().strftime('%H%M')}"
                )
            with col_s2:
                if st.button("💾 Kaydet", use_container_width=True):
                    rapor = {
                        "rapor_adi": rapor_adi,
                        "tarih": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "koordinat": {
                            "enlem": st.session_state.lat,
                            "boylam": st.session_state.lon
                        },
                        "cap_metre": cap,
                        "istatistikler": {
                            "ortalama": float(np.mean(st.session_state.Z_data)),
                            "maksimum": float(np.max(st.session_state.Z_data)),
                            "minimum": float(np.min(st.session_state.Z_data)),
                            "standart_sapma": float(np.std(st.session_state.Z_data))
                        },
                        "ai_yorum": ai_sonuc
                    }
                    
                    if ai_rapor_kaydet(rapor):
                        st.success(f"✅ '{rapor_adi}' kaydedildi!")
                    else:
                        st.error("❌ Kayıt başarısız!")

# Geçmiş raporlar
st.markdown("---")
with st.expander("📁 Geçmiş AI Raporları", expanded=False):
    raporlar = ai_raporlari_yukle()
    if raporlar:
        for i, rapor in enumerate(reversed(raporlar[-5:])):
            with st.container():
                col_r1, col_r2 = st.columns([3, 1])
                with col_r1:
                    st.write(f"**{rapor.get('rapor_adi', 'Rapor')}**")
                    st.caption(f"📍 {rapor['koordinat']['enlem']:.4f}, {rapor['koordinat']['boylam']:.4f} | 📅 {rapor['tarih']}")
                with col_r2:
                    if st.button("👁️ Gör", key=f"gor_{i}"):
                        st.info(rapor['ai_yorum'])
                st.divider()
    else:
        st.info("Henüz kayıtlı AI raporu yok")

# Alt bilgi
st.markdown("---")
st.caption("🛰️ Turkeller Surfer Pro v3.0 | 📱 Mobil Uyumlu | 🤖 Yerel AI")
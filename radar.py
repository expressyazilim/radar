import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import requests
import io
import json
import os
from PIL import Image

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Turkeller Surfer Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (MOBİL GÖRÜNÜM İÇİN) ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 50px; font-size: 16px; border-radius: 10px; }
    [data-testid="stSidebar"] { min-width: 300px; }
    </style>
    """, unsafe_allow_html=True)

# --- VERİ TABANI YÖNETİMİ (SİYAH EKRAN ÇÖZÜMÜ) ---
DB_FILE = "kayitli_yerler.json"

def yerleri_yukle():
    # Dosya yoksa veya boşsa uygulama çökmesin diye kontrol ekledik
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump([], f)
        return []
    try:
        with open(DB_FILE, "r") as f:
            content = f.read()
            return json.loads(content) if content else []
    except:
        return []

def yer_kaydet(isim, lat, lon):
    yerler = yerleri_yukle()
    yerler.append({"isim": isim, "lat": lat, "lon": lon})
    with open(DB_FILE, "w") as f:
        json.dump(yerler, f)

def yer_sil(index):
    yerler = yerleri_yukle()
    if 0 <= index < len(yerler):
        del yerler[index]
        with open(DB_FILE, "w") as f:
            json.dump(yerler, f)
        return True
    return False

# --- GÜVENLİK ---
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

def check_password():
    if not st.session_state["password_correct"]:
        st.title("🔐 Turkeller Surfer Pro Giriş") 
        pwd = st.text_input("Erişim Şifresi", type="password")
        if st.button("Giriş Yap"):
            if pwd == "altin2026":
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("❌ Hatalı şifre!")
        return False
    return True

# --- ANA UYGULAMA DÖNGÜSÜ ---
if check_password():
    # KOORDİNAT HAFIZASI
    if 'lat' not in st.session_state: st.session_state.lat = 40.104844
    if 'lon' not in st.session_state: st.session_state.lon = 27.769064

    # --- SOL PANEL (MENÜ) ---
    st.sidebar.title("🎮 Kontrol Paneli")
    
    st.sidebar.subheader("🚀 Arama ve Analiz")
    lat_input = st.sidebar.number_input("Enlem", value=st.session_state.lat, format="%.6f", step=0.000001)
    lon_input = st.sidebar.number_input("Boylam", value=st.session_state.lon, format="%.6f", step=0.000001)
    
    st.session_state.lat = lat_input
    st.session_state.lon = lon_input
    cap = st.sidebar.slider("Tarama Çapı (m)", 20, 300, 50)
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        analiz_butonu = st.button("🔍 ANALİZ", use_container_width=True)
    with col_b:
        maps_url = f"https://www.google.com/maps?q={st.session_state.lat},{st.session_state.lon}"
        st.link_button("🌍 HARİTA", maps_url, use_container_width=True)
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📌 Konumu Kaydet")
    yeni_isim = st.sidebar.text_input("Konum Adı", placeholder="Örn: Bölge-1")
    if st.sidebar.button("💾 HAFIZAYA AL"):
        if yeni_isim:
            yer_kaydet(yeni_isim, st.session_state.lat, st.session_state.lon)
            st.rerun()

    st.sidebar.divider()

    st.sidebar.subheader("📂 Kayıtlı Yerler")
    yerler = yerleri_yukle()
    if yerler:
        for i, y in enumerate(yerler):
            c_btn, c_del = st.sidebar.columns([0.8, 0.2])
            with c_btn:
                if st.button(f"📍 {y['isim']}", key=f"get_{i}", use_container_width=True):
                    st.session_state.lat = y['lat']
                    st.session_state.lon = y['lon']
                    st.rerun()
            with c_del:
                if st.button("🗑️", key=f"del_{i}"):
                    if yer_sil(i): st.rerun()
    else:
        st.sidebar.info("Henüz kayıt yok.")

    # --- ANA EKRAN ---
    st.title("🛰️ Turkeller Surfer Pro")

    if analiz_butonu:
        c_lat, c_lon = st.session_state.lat, st.session_state.lon
        
        CLIENT_ID = 'sh-8334dee7-cd0a-412e-8278-ceda2e981f0d'
        CLIENT_SECRET = 'QhUU1AbK8oBk8zBFvjQ0DIL3wjPEdVKN'

        def get_token():
            try:
                auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
                data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
                resp = requests.post(auth_url, data=data, timeout=10)
                return resp.json().get('access_token')
            except: return None

        token = get_token()
        if token:
            with st.spinner('Uydu verileri işleniyor...'):
                lat_f = cap / 111320.0
                lon_f = cap / (40075000.0 * math.cos(math.radians(c_lat)) / 360.0)
                bbox = [c_lon - lon_f, c_lat - lat_f, c_lon + lon_f, c_lat + lat_f]
                
                url = "https://sh.dataspace.copernicus.eu/api/v1/process"
                headers = {"Authorization": f"Bearer {token}"}
                
                evalscript = """
                function setup() {
                    return {
                        input: ["VV"],
                        output: { id: "default", bands: 1, sampleType: "FLOAT32" }
                    };
                }
                function evaluatePixel(sample) {
                    return [sample.VV];
                }
                """
                
                payload = {
                    "input": { 
                        "bounds": {
                            "bbox": bbox, 
                            "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}
                        },
                        "data": [{"type": "sentinel-1-grd"}] 
                    },
                    "output": {
                        "width": 120, "height": 120, 
                        "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]
                    },
                    "evalscript": evalscript
                }
                
                res = requests.post(url, headers=headers, json=payload, timeout=20)
                
                if res.status_code == 200:
                    try:
                        Z = np.nan_to_num(np.array(Image.open(io.BytesIO(res.content))), nan=0.0)
                        x_ax = np.linspace(bbox[0], bbox[2], Z.shape[1])
                        y_ax = np.linspace(bbox[1], bbox[3], Z.shape[0])
                        X, Y = np.meshgrid(x_ax, y_ax)

                        fig = go.Figure(data=[go.Surface(
                            z=Z, x=X, y=Y, colorscale='Viridis',
                            colorbar=dict(orientation='h', y=-0.2, thickness=15, title="Kütle Yoğunluğu")
                        )])

                        fig.update_layout(
                            scene=dict(
                                aspectratio=dict(x=1, y=1, z=0.5),
                                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                            ),
                            margin=dict(l=0, r=0, b=0, t=30),
                            height=600,
                            dragmode='turntable'
                        )

                        st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
                        st.success(f"Analiz Tamamlandı: {c_lat}, {c_lon}")
                        
                    except Exception as e:
                        st.error(f"Grafik oluşturulamadı: {e}")
                else:
                    st.error("Uydu sunucusu yanıt vermedi. Lütfen biraz sonra tekrar deneyin.")
        else:
            st.error("API Bağlantı Hatası: Lütfen internet bağlantınızı ve API anahtarlarınızı kontrol edin.")

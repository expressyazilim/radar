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
    initial_sidebar_state="collapsed" # Mobilde ekranı genişletir
)

# --- CSS (MOBİL BUTONLAR VE EKRAN YÖNETİMİ) ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; height: 55px; font-size: 18px !important; border-radius: 12px; }
    [data-testid="stSidebar"] { min-width: 300px; }
    /* Mobilde input alanlarını büyüt */
    input { font-size: 16px !important; }
    </style>
    """, unsafe_allow_html=True)

# --- GİRİŞ KONTROLÜ (MOBİLDE TAKILMAYAN YAPI) ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    st.title("🔐 Turkeller Surfer Pro")
    st.subheader("Mobil Giriş Paneli")
    pwd = st.text_input("Erişim Şifresi", type="password")
    if st.button("Sisteme Giriş Yap"):
        if pwd == "altin2026":
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Hatalı Şifre!")

if not st.session_state["authenticated"]:
    login()
else:
    # --- UYGULAMA İÇERİĞİ BURADAN BAŞLAR ---
    
    DB_FILE = "kayitli_yerler.json"

    def yerleri_yukle():
        if not os.path.exists(DB_FILE):
            with open(DB_FILE, "w") as f: json.dump([], f)
            return []
        try:
            with open(DB_FILE, "r") as f: return json.load(f)
        except: return []

    def yer_kaydet(isim, lat, lon):
        yerler = yerleri_yukle()
        yerler.append({"isim": isim, "lat": lat, "lon": lon})
        with open(DB_FILE, "w") as f: json.dump(yerler, f)

    def yer_sil(index):
        yerler = yerleri_yukle()
        if 0 <= index < len(yerler):
            del yerler[index]
            with open(DB_FILE, "w") as f: json.dump(yerler, f)
            return True
        return False

    # KOORDİNAT HAFIZASI
    if 'lat' not in st.session_state: st.session_state.lat = 40.104844
    if 'lon' not in st.session_state: st.session_state.lon = 27.769064

    # --- YAN MENÜ ---
    st.sidebar.title("🎮 Turkeller Menü")
    lat_input = st.sidebar.number_input("Enlem", value=st.session_state.lat, format="%.6f")
    lon_input = st.sidebar.number_input("Boylam", value=st.session_state.lon, format="%.6f")
    
    st.session_state.lat = lat_input
    st.session_state.lon = lon_input
    cap = st.sidebar.slider("Tarama Çapı (m)", 20, 300, 50)
    
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        analiz_butonu = st.button("🔍 ANALİZ", use_container_width=True)
    with col_b:
        maps_url = f"https://www.google.com/maps/search/?api=1&query={st.session_state.lat},{st.session_state.lon}"
        st.link_button("🌍 HARİTA", maps_url, use_container_width=True)
    
    st.sidebar.divider()
    
    # KAYITLI YERLER LİSTESİ
    yerler = yerleri_yukle()
    for i, y in enumerate(yerler):
        c_btn, c_del = st.sidebar.columns([0.8, 0.2])
        with c_btn:
            if st.button(f"📍 {y['isim']}", key=f"y_{i}"):
                st.session_state.lat, st.session_state.lon = y['lat'], y['lon']
                st.rerun()
        with c_del:
            if st.button("🗑️", key=f"d_{i}"):
                if yer_sil(i): st.rerun()

    # --- ANA ANALİZ EKRANI ---
    st.header("🛰️ Turkeller Surfer Pro")
    
    if analiz_butonu:
        # API ve Grafik kodları (Aynen devam ediyor...)
        CLIENT_ID = 'sh-8334dee7-cd0a-412e-8278-ceda2e981f0d'
        CLIENT_SECRET = 'QhUU1AbK8oBk8zBFvjQ0DIL3wjPEdVKN'

        def get_token():
            try:
                auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
                data = {"grant_type": "client_credentials", "client_id": CLIENT_ID, "client_secret": CLIENT_SECRET}
                return requests.post(auth_url, data=data).json().get('access_token')
            except: return None

        token = get_token()
        if token:
            with st.spinner('Veri çekiliyor...'):
                # BBOX Hesaplama
                lat_f = cap / 111320.0
                lon_f = cap / (40075000.0 * math.cos(math.radians(st.session_state.lat)) / 360.0)
                bbox = [st.session_state.lon - lon_f, st.session_state.lat - lat_f, st.session_state.lon + lon_f, st.session_state.lat + lat_f]
                
                # API Sorgusu
                evalscript = """function setup() { return { input: ["VV"], output: { id: "default", bands: 1, sampleType: "FLOAT32" } }; } function evaluatePixel(sample) { return [sample.VV]; }"""
                payload = {
                    "input": { "bounds": { "bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"} }, "data": [{"type": "sentinel-1-grd"}] },
                    "output": { "width": 120, "height": 120, "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}] },
                    "evalscript": evalscript
                }
                
                res = requests.post("https://sh.dataspace.copernicus.eu/api/v1/process", headers={"Authorization": f"Bearer {token}"}, json=payload)
                
                if res.status_code == 200:
                    Z = np.nan_to_num(np.array(Image.open(io.BytesIO(res.content))), nan=0.0)
                    X, Y = np.meshgrid(np.linspace(bbox[0], bbox[2], 120), np.linspace(bbox[1], bbox[3], 120))

                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', colorbar=dict(orientation='h', y=-0.2))])
                    fig.update_layout(scene=dict(aspectratio=dict(x=1, y=1, z=0.5)), margin=dict(l=0, r=0, b=0, t=0), height=600)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error("Veri alınamadı.")

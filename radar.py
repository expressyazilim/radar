import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import requests
import io
import json
import os
import tifffile as tiff
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
    .main {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%) !important;
        color: white !important;
        padding: 1rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #1a252f 0%, #2c3e50 100%) !important;
        color: white !important;
    }
    .main *, [data-testid="stSidebar"] * { color: white !important; }

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
    .stButton > button:hover {
        background: linear-gradient(135deg, #7f8c8d 0%, #5d6d7e 100%) !important;
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    .stat-card {
        background: linear-gradient(135deg, #34495e 0%, #2c3e50 100%) !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #5d6d7e !important;
        text-align: center;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .ai-card {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%) !important;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #5d6d7e !important;
        margin: 10px 0;
        white-space: pre-wrap;
    }
    hr { border-color: #5d6d7e !important; margin: 20px 0 !important; }
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
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #2c3e50 0%, #1a252f 100%) !important;
        border: 1px solid #5d6d7e !important;
        border-radius: 10px !important;
        padding: 15px !important;
    }
    .js-plotly-plot {
        background: #1a252f !important;
        border-radius: 10px;
        padding: 10px;
    }
    label { color: #bdc3c7 !important; font-weight: 500 !important; }
    .stCaption { color: #95a5a6 !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# LOGIN
# ---------------------------
APP_PASSWORD = "altin2026"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔐 Turkeller Surfer Pro")
        st.markdown("---")
        pwd = st.text_input("**Erişim Şifresi**", type="password", key="login_pwd")
        if st.button("🚀 Giriş Yap", use_container_width=True):
            if pwd == APP_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Hatalı şifre!")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# ---------------------------
# STORAGE
# ---------------------------
DB_FILE = "kayitli_yerler.json"
AI_REPORTS_FILE = "ai_analiz_raporlari.jsonl"

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

def yer_kaydet(isim, lat, lon):
    yerler = yerleri_yukle()
    yerler.append({"isim": isim, "lat": round(float(lat), 6), "lon": round(float(lon), 6)})
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

def ai_raporlari_yukle():
    if not os.path.exists(AI_REPORTS_FILE):
        return []
    try:
        with open(AI_REPORTS_FILE, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except:
        return []

def ai_rapor_kaydet(rapor):
    try:
        with open(AI_REPORTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rapor, ensure_ascii=False) + "\n")
        return True
    except:
        return False

# ---------------------------
# SESSION STATE
# ---------------------------
if "lat" not in st.session_state:
    st.session_state.lat = 40.104844
if "lon" not in st.session_state:
    st.session_state.lon = 27.769064
if "Z_data" not in st.session_state:
    st.session_state.Z_data = None
if "X_data" not in st.session_state:
    st.session_state.X_data = None
if "Y_data" not in st.session_state:
    st.session_state.Y_data = None

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("## 🎮 Kontrol Paneli")

    st.session_state.lat = st.number_input("**Enlem**", value=float(st.session_state.lat), format="%.6f")
    st.session_state.lon = st.number_input("**Boylam**", value=float(st.session_state.lon), format="%.6f")
    cap = st.slider("**Tarama Çapı (m)**", 20, 300, 50)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        analiz_butonu = st.button("🔍 ANALİZ", type="primary", use_container_width=True)
    with c2:
        ai_yorum_butonu = st.button("🤖 AI YORUM", disabled=st.session_state.Z_data is None, use_container_width=True)

    st.markdown("---")
    maps_url = f"https://www.google.com/maps/search/?api=1&query={st.session_state.lat},{st.session_state.lon}"
    st.markdown(f"""
    <a href="{maps_url}" target="_blank">
        <button style="width:100%;height:50px;background:linear-gradient(135deg,#5d6d7e 0%,#34495e 100%);color:white;border:none;border-radius:8px;font-size:16px;font-weight:600;cursor:pointer;margin:10px 0;">
        🌍 HARİTADA GÖSTER
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
                    st.session_state.lat = y["lat"]
                    st.session_state.lon = y["lon"]
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"sil_{i}"):
                    if yer_sil(i):
                        st.rerun()

    with st.expander("➕ Yeni Yer Kaydet", expanded=False):
        kayit_isim = st.text_input("Yer İsmi", placeholder="Örn: İşyeri")
        if st.button("💾 Kaydet", use_container_width=True):
            if kayit_isim.strip():
                yer_kaydet(kayit_isim, st.session_state.lat, st.session_state.lon)
                st.success(f"✅ '{kayit_isim}' kaydedildi!")
                st.rerun()

    st.markdown("---")
    st.caption("🛰️ Turkeller Surfer Pro v3.1")

# ---------------------------
# MAIN UI
# ---------------------------
st.markdown("# 🛰️ Turkeller Surfer Pro")
st.markdown("### 📍 Mevcut Konum")

col_loc1, col_loc2, col_loc3 = st.columns(3)
with col_loc1:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">ENLEM</div>
        <div style="font-size:24px;font-weight:bold;">{st.session_state.lat:.6f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_loc2:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">BOYLAM</div>
        <div style="font-size:24px;font-weight:bold;">{st.session_state.lon:.6f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_loc3:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">TARAMA ÇAPI</div>
        <div style="font-size:24px;font-weight:bold;">{cap} m</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------
# TOKEN (robust + debug)
# ---------------------------
def get_token_debug():
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # 1) client_credentials dene
    try:
        data = {
            "grant_type": "client_credentials",
            "client_id": st.secrets["CDSE_CLIENT_ID"],
            "client_secret": st.secrets["CDSE_CLIENT_SECRET"],
        }
        r = requests.post(auth_url, data=data, timeout=30)
        if r.status_code == 200:
            return r.json().get("access_token")
        st.warning(f"⚠️ client_credentials başarısız: HTTP {r.status_code} | {r.text[:200]}")
    except Exception as e:
        st.warning(f"⚠️ client_credentials exception: {e}")

    # 2) password grant dene (varsayılan değil, secrets varsa)
    try:
        if "CDSE_USERNAME" in st.secrets and "CDSE_PASSWORD" in st.secrets:
            data = {
                "grant_type": "password",
                "client_id": st.secrets["CDSE_CLIENT_ID"],
                "username": st.secrets["CDSE_USERNAME"],
                "password": st.secrets["CDSE_PASSWORD"],
            }
            r = requests.post(auth_url, data=data, timeout=30)
            if r.status_code == 200:
                return r.json().get("access_token")
            st.error(f"❌ password grant başarısız: HTTP {r.status_code} | {r.text[:200]}")
        else:
            st.error("❌ Token alınamadı: client_credentials başarısız ve CDSE_USERNAME/CDSE_PASSWORD secrets ekli değil.")
            return None
    except Exception as e:
        st.error(f"❌ password grant exception: {e}")
        return None

# ---------------------------
# ANALYZE
# ---------------------------
if analiz_butonu:
    # Secrets kontrol
    if "CDSE_CLIENT_ID" not in st.secrets or "CDSE_CLIENT_SECRET" not in st.secrets:
        st.error("❌ Secrets eksik! Settings → Secrets içine CDSE_CLIENT_ID ve CDSE_CLIENT_SECRET ekle.")
        st.stop()

    token = get_token_debug()
    if not token:
        st.stop()

    with st.spinner("🛰️ Veri çekiliyor..."):
        try:
            lat_f = cap / 111320.0
            lon_f = cap / (40075000.0 * math.cos(math.radians(st.session_state.lat)) / 360.0)
            bbox = [
                st.session_state.lon - lon_f,
                st.session_state.lat - lat_f,
                st.session_state.lon + lon_f,
                st.session_state.lat + lat_f
            ]

            evalscript = """
            function setup() {
              return {
                input: ["VV"],
                output: { id: "default", bands: 1, sampleType: "FLOAT32" }
              };
            }
            function evaluatePixel(sample) { return [sample.VV]; }
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

            if res.status_code != 200:
                st.error(f"❌ Veri alınamadı! HTTP {res.status_code} | {res.text[:200]}")
                st.stop()

            Z = tiff.imread(io.BytesIO(res.content)).astype(np.float32)

            X, Y = np.meshgrid(
                np.linspace(bbox[0], bbox[2], 120),
                np.linspace(bbox[1], bbox[3], 120)
            )

            st.session_state.Z_data = Z
            st.session_state.X_data = X
            st.session_state.Y_data = Y

            fig = go.Figure(data=[go.Surface(
                z=Z, x=X, y=Y, colorscale="Viridis",
                hovertemplate=(
                    "<b>Boylam</b>: %{x:.6f}<br>"
                    "<b>Enlem</b>: %{y:.6f}<br>"
                    "<b>VV</b>: %{z:.4f}<br><extra></extra>"
                )
            )])

            fig.update_layout(
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=0.5),
                    xaxis_title="Boylam",
                    yaxis_title="Enlem",
                    zaxis_title="VV",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                ),
                margin=dict(l=0, r=0, b=0, t=40),
                height=520,
                title=f"Sentinel-1 SAR (VV) - {cap}m Çap"
            )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("📊 Detaylı İstatistikler", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ortalama", f"{np.mean(Z):.4f}")
                col2.metric("Maksimum", f"{np.max(Z):.4f}")
                col3.metric("Minimum", f"{np.min(Z):.4f}")
                col4.metric("Std", f"{np.std(Z):.4f}")

            st.success("✅ Analiz tamamlandı!")

        except Exception as e:
            st.error(f"❌ Analiz exception: {e}")

# ---------------------------
# LOCAL AI
# ---------------------------
def yerel_ai_analizi(Z_data, lat, lon, cap):
    Z_clean = Z_data[~np.isnan(Z_data)]
    if len(Z_clean) == 0:
        return "⚠️ HATA: Geçerli veri yok!"

    mean_val = float(np.mean(Z_clean))
    max_val = float(np.max(Z_clean))
    min_val = float(np.min(Z_clean))
    std_val  = float(np.std(Z_clean))

    eps = 1e-10
    mean_db = 10 * math.log10(max(mean_val, eps))

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

    rapor = f"""🤖 AI ANALİZ RAPORU

📍 Konum:
- Enlem: {lat:.6f}
- Boylam: {lon:.6f}
- Çap: {cap} m
- Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}

📊 İstatistik:
- Ortalama VV: {mean_val:.4f}
- dB (yaklaşık): {mean_db:.1f} dB
- Maks: {max_val:.4f}
- Min: {min_val:.4f}
- Std: {std_val:.4f}

🔍 Yorum:
- Yüzey tipi: {yuzey}
- Topoğrafya: {'⛰️ Engebeli' if std_val > 0.3 else '🏞️ Hafif engebeli' if std_val > 0.15 else '🏙️ Düz'}
- Değişim: {'🎯 Yüksek' if (max_val-min_val) > 0.5 else '📊 Orta' if (max_val-min_val) > 0.2 else '🔄 Düşük'}
"""
    return rapor

if "ai_yorum_butonu" in locals() and ai_yorum_butonu:
    if st.session_state.Z_data is None:
        st.warning("⚠️ Önce 'ANALİZ' ile veri çek!")
    else:
        with st.spinner("🤖 AI analiz yapıyor..."):
            ai_sonuc = yerel_ai_analizi(st.session_state.Z_data, st.session_state.lat, st.session_state.lon, cap)

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
                    "koordinat": {"enlem": st.session_state.lat, "boylam": st.session_state.lon},
                    "cap_metre": cap,
                    "istatistikler": {
                        "ortalama": float(np.mean(st.session_state.Z_data)),
                        "maksimum": float(np.max(st.session_state.Z_data)),
                        "minimum": float(np.min(st.session_state.Z_data)),
                        "standart_sapma": float(np.std(st.session_state.Z_data)),
                    },
                    "ai_yorum": ai_sonuc
                }
                if ai_rapor_kaydet(rapor):
                    st.success(f"✅ '{rapor_adi}' kaydedildi!")
                else:
                    st.error("❌ Kayıt başarısız!")

st.markdown("---")
with st.expander("📁 Geçmiş AI Raporları", expanded=False):
    raporlar = ai_raporlari_yukle()
    if raporlar:
        for i, rapor in enumerate(reversed(raporlar[-5:])):
            col_r1, col_r2 = st.columns([3, 1])
            with col_r1:
                st.write(f"**{rapor.get('rapor_adi','Rapor')}**")
                st.caption(f"📍 {rapor['koordinat']['enlem']:.4f}, {rapor['koordinat']['boylam']:.4f} | 📅 {rapor['tarih']}")
            with col_r2:
                if st.button("👁️ Gör", key=f"gor_{i}"):
                    st.info(rapor["ai_yorum"])
            st.divider()
    else:
        st.info("Henüz kayıtlı AI raporu yok")

st.caption("🛰️ Turkeller Surfer Pro v3.1 | Streamlit Cloud | Debug Token Aktif")

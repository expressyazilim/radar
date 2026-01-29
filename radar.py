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
import re
from collections import deque

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Turkeller Surfer Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS (KOYU GRİ/ANTRASİT) ---
st.markdown("""
<style>
.main { background: linear-gradient(135deg,#2c3e50 0%,#34495e 100%) !important; color:white !important; padding:1rem; }
[data-testid="stSidebar"] { background: linear-gradient(135deg,#1a252f 0%,#2c3e50 100%) !important; color:white !important; }
.main *, [data-testid="stSidebar"] * { color:white !important; }

.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background:#0d141c !important; color:white !important; border:1px solid #34495e !important;
    border-radius:8px !important; font-size:16px !important; height:45px !important; padding:5px 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg,#5d6d7e 0%,#34495e 100%) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    height:50px !important; font-size:16px !important; font-weight:600 !important;
    margin:5px 0 !important; width:100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#7f8c8d 0%,#5d6d7e 100%) !important;
    transform: translateY(-2px); transition: all 0.3s ease;
}
.stat-card{
    background: linear-gradient(135deg,#34495e 0%,#2c3e50 100%) !important;
    padding:20px; border-radius:12px; border:1px solid #5d6d7e !important; text-align:center; margin:10px 0;
    box-shadow:0 4px 6px rgba(0,0,0,0.3);
}
.ai-card{
    background: linear-gradient(135deg,#2c3e50 0%,#1a252f 100%) !important;
    padding:20px; border-radius:12px; border:1px solid #5d6d7e !important; margin:10px 0; white-space:pre-wrap;
}
hr{ border-color:#5d6d7e !important; margin:20px 0 !important; }
.streamlit-expanderHeader{
    background:#2c3e50 !important; color:white !important; border:1px solid #5d6d7e !important; border-radius:8px !important;
}
.streamlit-expanderContent{
    background:#1a252f !important; border:1px solid #5d6d7e !important; border-radius:0 0 8px 8px !important;
}
[data-testid="metric-container"]{
    background: linear-gradient(135deg,#2c3e50 0%,#1a252f 100%) !important;
    border:1px solid #5d6d7e !important; border-radius:10px !important; padding:15px !important;
}
.js-plotly-plot{ background:#1a252f !important; border-radius:10px; padding:10px; }
label{ color:#bdc3c7 !important; font-weight:500 !important; }
.stCaption{ color:#95a5a6 !important; }
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
    yerler.append({"isim": isim, "lat": float(lat), "lon": float(lon)})
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
if "lat_str" not in st.session_state:
    st.session_state.lat_str = "40.104844000000"
if "lon_str" not in st.session_state:
    st.session_state.lon_str = "27.769064000000"

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
# HELPERS
# ---------------------------
def parse_coord(s: str):
    s = (s or "").strip()
    s = s.replace(",", ".")
    if not re.fullmatch(r"-?\d+(\.\d+)?", s):
        return None
    try:
        return float(s)
    except:
        return None

def connected_components(mask: np.ndarray):
    """
    8-komşuluk connected components.
    Dönen liste: her bileşen için dict:
      pixels: list[(r,c)]
      area: int
      bbox: (rmin,rmax,cmin,cmax)
    """
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []

    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),       ( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]

    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                q = deque()
                q.append((r, c))
                visited[r, c] = True
                pixels = []
                rmin=rmax=r
                cmin=cmax=c

                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))
                    if rr < rmin: rmin = rr
                    if rr > rmax: rmax = rr
                    if cc < cmin: cmin = cc
                    if cc > cmax: cmax = cc

                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                q.append((nr, nc))

                comps.append({
                    "pixels": pixels,
                    "area": len(pixels),
                    "bbox": (rmin, rmax, cmin, cmax)
                })
    return comps

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("## 🎮 Kontrol Paneli")

    lat_in = st.text_input("**Enlem (Lat)**", value=st.session_state.lat_str)
    lon_in = st.text_input("**Boylam (Lon)**", value=st.session_state.lon_str)

    lat_val = parse_coord(lat_in)
    lon_val = parse_coord(lon_in)

    if lat_val is None or lon_val is None:
        st.warning("⚠️ Enlem/Boylam formatı geçersiz. Örn: 40.123456789 veya 40,123456789")
    else:
        st.session_state.lat = lat_val
        st.session_state.lon = lon_val
        st.session_state.lat_str = lat_in
        st.session_state.lon_str = lon_in

    cap = st.slider("**Tarama Çapı (m)**", 20, 300, 50)
    anomali_esik = st.slider("**Anomali Eşiği (z-score)**", 1.5, 5.0, 2.8, 0.1)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        analiz_butonu = st.button("🔍 ANALİZ", type="primary", use_container_width=True, disabled=(lat_val is None or lon_val is None))
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
                    st.session_state.lat = float(y["lat"])
                    st.session_state.lon = float(y["lon"])
                    st.session_state.lat_str = f"{st.session_state.lat:.12f}"
                    st.session_state.lon_str = f"{st.session_state.lon:.12f}"
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"sil_{i}"):
                    if yer_sil(i):
                        st.rerun()

    with st.expander("➕ Yeni Yer Kaydet", expanded=False):
        kayit_isim = st.text_input("Yer İsmi", placeholder="Örn: İşyeri")
        if st.button("💾 Kaydet", use_container_width=True):
            if kayit_isim.strip() and lat_val is not None and lon_val is not None:
                yer_kaydet(kayit_isim.strip(), st.session_state.lat, st.session_state.lon)
                st.success(f"✅ '{kayit_isim}' kaydedildi!")
                st.rerun()

    st.markdown("---")
    st.caption("🛰️ Turkeller Surfer Pro v3.3")

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
        <div style="font-size:22px;font-weight:bold;">{st.session_state.lat:.12f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_loc2:
    st.markdown(f"""
    <div class="stat-card">
        <div style="font-size:14px;opacity:0.9;margin-bottom:10px;">BOYLAM</div>
        <div style="font-size:22px;font-weight:bold;">{st.session_state.lon:.12f}</div>
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
# TOKEN
# ---------------------------
def get_token_debug():
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # client_credentials
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

    # password grant fallback
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
            return None
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
              return { input: ["VV"], output: { id: "default", bands: 1, sampleType: "FLOAT32" } };
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
                st.error(f"❌ Veri alınamadı! HTTP {res.status_code} | {res.text[:300]}")
                st.stop()

            Z = tiff.imread(io.BytesIO(res.content)).astype(np.float32)

            X, Y = np.meshgrid(
                np.linspace(bbox[0], bbox[2], 120),
                np.linspace(bbox[1], bbox[3], 120)
            )

            # --- VV -> dB ---
            eps = 1e-10
            Z_db = 10.0 * np.log10(np.maximum(Z, eps))

            # --- Robust clip ---
            valid = Z_db[~np.isnan(Z_db)]
            p1, p99 = np.percentile(valid, [1, 99])
            Z_db_clip = np.clip(Z_db, p1, p99)

            # --- z-score ---
            mu = float(np.mean(Z_db_clip))
            sd = float(np.std(Z_db_clip)) if float(np.std(Z_db_clip)) > 1e-6 else 1.0
            Z_z = (Z_db_clip - mu) / sd

            # store
            st.session_state.Z_data = Z_db_clip
            st.session_state.X_data = X
            st.session_state.Y_data = Y

            # --- anomaly mask + blobs ---
            anom_mask = np.abs(Z_z) >= float(anomali_esik)
            comps = connected_components(anom_mask)

            # score each component: peak_abs_z * sqrt(area)
            ranked = []
            for comp in comps:
                pix = comp["pixels"]
                rr = np.array([p[0] for p in pix], dtype=int)
                cc = np.array([p[1] for p in pix], dtype=int)
                peak = float(np.max(np.abs(Z_z[rr, cc]))) if len(pix) else 0.0
                area = comp["area"]
                score = peak * math.sqrt(max(area, 1))
                rmin, rmax, cmin, cmax = comp["bbox"]
                center_lat = float(np.mean(Y[rr, cc]))
                center_lon = float(np.mean(X[rr, cc]))
                ranked.append({
                    "score": score,
                    "peak_z": peak,
                    "area": area,
                    "bbox_rc": (rmin, rmax, cmin, cmax),
                    "center_lat": center_lat,
                    "center_lon": center_lon,
                })

            ranked.sort(key=lambda d: d["score"], reverse=True)
            top3 = ranked[:3]

            # --- 2D + 3D side-by-side ---
            colA, colB = st.columns([1, 1])

            with colA:
                st.subheader("🗺️ 2D Heatmap (VV dB)")

                heat_fig = go.Figure()
                heat_fig.add_trace(go.Heatmap(
                    z=Z_db_clip,
                    x=X[0, :],
                    y=Y[:, 0],
                    colorbar=dict(title="VV (dB)")
                ))

                # kontur (anomali maskesi)
                heat_fig.add_trace(go.Contour(
                    z=anom_mask.astype(int),
                    x=X[0, :],
                    y=Y[:, 0],
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1),
                    line=dict(width=2),
                    hoverinfo="skip",
                    name="Anomali"
                ))

                # top3 bounding box + label
                for i, t in enumerate(top3, start=1):
                    rmin, rmax, cmin, cmax = t["bbox_rc"]
                    x0 = float(X[0, cmin]); x1 = float(X[0, cmax])
                    y0 = float(Y[rmin, 0]); y1 = float(Y[rmax, 0])

                    heat_fig.add_shape(
                        type="rect",
                        x0=min(x0, x1), x1=max(x0, x1),
                        y0=min(y0, y1), y1=max(y0, y1),
                        line=dict(width=3),
                    )
                    heat_fig.add_trace(go.Scatter(
                        x=[t["center_lon"]],
                        y=[t["center_lat"]],
                        mode="markers+text",
                        text=[f"#{i}"],
                        textposition="top center",
                        marker=dict(size=10),
                        name=f"Top {i}"
                    ))

                heat_fig.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=35, b=0),
                    xaxis_title="Boylam",
                    yaxis_title="Enlem",
                    title="2D Isı Haritası + Anomali Konturu + Top3 Kutular"
                )
                st.plotly_chart(heat_fig, use_container_width=True)

                st.markdown("### 🎯 En Güçlü 3 Anomali")
                if not top3:
                    st.info("Bu eşikte anomali bulunamadı. Eşiği düşürmeyi deneyebilirsin.")
                else:
                    for i, t in enumerate(top3, start=1):
                        st.success(
                            f"#{i} | score={t['score']:.2f} | peak z={t['peak_z']:.2f} | alan={t['area']} px"
                        )
                        # kopyalanabilir blok
                        st.code(f"{t['center_lat']:.8f}, {t['center_lon']:.8f}", language="text")

            with colB:
                st.subheader("🧊 3D Surface (VV dB)")

                surf_fig = go.Figure(data=[go.Surface(
                    z=Z_db_clip,
                    x=X,
                    y=Y,
                    colorscale="Viridis",
                    hovertemplate=(
                        "<b>Boylam</b>: %{x:.6f}<br>"
                        "<b>Enlem</b>: %{y:.6f}<br>"
                        "<b>VV (dB)</b>: %{z:.2f}<br><extra></extra>"
                    )
                )])

                surf_fig.update_layout(
                    scene=dict(
                        aspectratio=dict(x=1, y=1, z=0.5),
                        xaxis_title="Boylam",
                        yaxis_title="Enlem",
                        zaxis_title="VV (dB)",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
                    ),
                    margin=dict(l=0, r=0, b=0, t=35),
                    height=520,
                    title="3D dB Yüzeyi"
                )

                st.plotly_chart(surf_fig, use_container_width=True)

            # --- Stats ---
            st.markdown("---")
            with st.expander("📊 Detaylı İstatistikler (dB)", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Ortalama (dB)", f"{np.mean(Z_db_clip):.2f}")
                col2.metric("Maks (dB)", f"{np.max(Z_db_clip):.2f}")
                col3.metric("Min (dB)", f"{np.min(Z_db_clip):.2f}")
                col4.metric("Std (dB)", f"{np.std(Z_db_clip):.2f}")

            st.success("✅ Analiz tamamlandı!")

        except Exception as e:
            st.error(f"❌ Analiz exception: {e}")

# ---------------------------
# LOCAL AI (dB üzerinden)
# ---------------------------
def yerel_ai_analizi(Z_db_data, lat, lon, cap):
    Z_clean = Z_db_data[~np.isnan(Z_db_data)]
    if len(Z_clean) == 0:
        return "⚠️ HATA: Geçerli veri yok!"

    mean_db = float(np.mean(Z_clean))
    max_db = float(np.max(Z_clean))
    min_db = float(np.min(Z_clean))
    std_db = float(np.std(Z_clean))

    if mean_db > -5:
        yuzey = "🚨 Çok güçlü yansıtıcı (şehir/metal yapı olasılığı)"
    elif mean_db > -10:
        yuzey = "🏢 Güçlü yansıtıcı (orman/tarım karma)"
    elif mean_db > -15:
        yuzey = "🌾 Orta yansıtıcı (açık arazi/çıplak toprak)"
    elif mean_db > -20:
        yuzey = "🏜️ Zayıf yansıtıcı (daha düz yüzey)"
    else:
        yuzey = "🌊 Çok zayıf yansıtıcı (su/çok düşük geri saçılım)"

    rapor = f"""🤖 AI ANALİZ RAPORU (VV dB)

📍 Konum:
- Enlem: {lat:.12f}
- Boylam: {lon:.12f}
- Çap: {cap} m
- Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}

📊 İstatistik (dB):
- Ortalama: {mean_db:.2f} dB
- Maks: {max_db:.2f} dB
- Min: {min_db:.2f} dB
- Std: {std_db:.2f} dB

🔍 Yorum:
- Yüzey tipi: {yuzey}
- Topoğrafya: {'⛰️ Engebeli' if std_db > 3 else '🏞️ Hafif engebeli' if std_db > 1.5 else '🏙️ Düz'}
- Değişim: {'🎯 Yüksek' if (max_db-min_db) > 6 else '📊 Orta' if (max_db-min_db) > 3 else '🔄 Düşük'}
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
                    "istatistikler_db": {
                        "ortalama_db": float(np.mean(st.session_state.Z_data)),
                        "maksimum_db": float(np.max(st.session_state.Z_data)),
                        "minimum_db": float(np.min(st.session_state.Z_data)),
                        "standart_sapma_db": float(np.std(st.session_state.Z_data)),
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

st.caption("🛰️ Turkeller Surfer Pro v3.3 | 2D+3D | Top3 Anomali + BBox")

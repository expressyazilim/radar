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
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG (MOBILE FRIENDLY)
# =========================
st.set_page_config(
    page_title="Turkeller Surfer Pro",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# CSS (MOBILE RESPONSIVE)
# =========================
st.markdown(
    """
<style>
/* Background + global */
.main { background: linear-gradient(135deg,#2c3e50 0%,#34495e 100%) !important; color:white !important; padding:0.6rem; }
.main * { color:white !important; }
hr{ border-color:#5d6d7e !important; margin:14px 0 !important; }

.stTextInput > div > div > input {
    background:#0d141c !important; color:white !important; border:1px solid #34495e !important;
    border-radius:10px !important; font-size:16px !important; height:44px !important; padding:6px 10px !important;
}

.stButton > button {
    background: linear-gradient(135deg,#5d6d7e 0%,#34495e 100%) !important;
    color:white !important; border:none !important; border-radius:10px !important;
    height:46px !important; font-size:15px !important; font-weight:650 !important;
    margin:4px 0 !important; width:100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#7f8c8d 0%,#5d6d7e 100%) !important;
    transform: translateY(-1px); transition: all 0.2s ease;
}

.small-card{
    background: rgba(0,0,0,0.18);
    border:1px solid rgba(255,255,255,0.12);
    border-radius:12px;
    padding:10px 12px;
    margin:8px 0;
}

.ai-card{
    background: linear-gradient(135deg,#2c3e50 0%,#1a252f 100%) !important;
    padding:14px; border-radius:12px; border:1px solid #5d6d7e !important; margin:10px 0; white-space:pre-wrap;
}

/* Plotly container spacing */
.js-plotly-plot{ background:#1a252f !important; border-radius:12px; padding:6px; }

/* Reduce Streamlit default top padding */
.block-container { padding-top: 0.6rem; padding-bottom: 1.2rem; }

/* Mobile tweaks */
@media (max-width: 640px){
  .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
  .stButton > button { height: 44px !important; font-size:14px !important; }
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# LOGIN (PASSWORD UNCHANGED)
# =========================
APP_PASSWORD = "altin2026"

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "remember_session" not in st.session_state:
    st.session_state["remember_session"] = True  # session-based remember
if "login_pwd_cache" not in st.session_state:
    st.session_state["login_pwd_cache"] = ""

def login():
    st.markdown("## 🔐 Turkeller Surfer Pro")
    st.caption("Mobil uyumlu sürüm | Oturum bazlı şifre hatırlama aktif.")
    st.markdown("---")

    default_pwd = st.session_state["login_pwd_cache"] if st.session_state["remember_session"] else ""
    pwd = st.text_input("**Erişim Şifresi**", type="password", value=default_pwd, key="login_pwd")

    colA, colB = st.columns([1, 1])
    with colA:
        remember = st.checkbox("Bu oturumda hatırla", value=True, key="remember_chk")
    with colB:
        st.write("")

    if st.button("🚀 Giriş Yap", use_container_width=True):
        if pwd == APP_PASSWORD:
            st.session_state["authenticated"] = True
            if remember:
                st.session_state["login_pwd_cache"] = pwd
                st.session_state["remember_session"] = True
            else:
                st.session_state["login_pwd_cache"] = ""
                st.session_state["remember_session"] = False
            st.rerun()
        else:
            st.error("❌ Hatalı şifre!")

if not st.session_state["authenticated"]:
    login()
    st.stop()

# =========================
# STORAGE
# =========================
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

# =========================
# SESSION STATE
# =========================
if "coord_str" not in st.session_state:
    st.session_state.coord_str = "40.1048440 27.7690640"
if "lat" not in st.session_state:
    st.session_state.lat = 40.1048440
if "lon" not in st.session_state:
    st.session_state.lon = 27.7690640

if "Z_data" not in st.session_state:
    st.session_state.Z_data = None
if "X_data" not in st.session_state:
    st.session_state.X_data = None
if "Y_data" not in st.session_state:
    st.session_state.Y_data = None

if "focus_lat" not in st.session_state: st.session_state.focus_lat = None
if "focus_lon" not in st.session_state: st.session_state.focus_lon = None
if "focus_label" not in st.session_state: st.session_state.focus_label = None

if "history" not in st.session_state:
    st.session_state.history = []  # list of dict snapshots

# =========================
# HELPERS
# =========================
def parse_coord_pair(s: str):
    """
    Accept: "40.10 27.76" or "40,10 27,76" or "40.10,27.76"
    Returns (lat, lon) or (None, None)
    """
    if not s:
        return None, None
    s = s.strip()
    s = s.replace(",", " ")
    parts = [p for p in s.split() if p.strip()]
    if len(parts) < 2:
        return None, None
    def _to_float(x):
        x = x.strip().replace(",", ".")
        if not re.fullmatch(r"-?\d+(\.\d+)?", x):
            return None
        try:
            return float(x)
        except:
            return None
    lat = _to_float(parts[0])
    lon = _to_float(parts[1])
    if lat is None or lon is None:
        return None, None
    return lat, lon

def connected_components(mask: np.ndarray):
    """8-neighborhood CC"""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps = []
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),       ( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not visited[r, c]:
                q = deque([(r, c)])
                visited[r, c] = True
                pixels = []
                rmin=rmax=r
                cmin=cmax=c
                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))
                    rmin = min(rmin, rr); rmax = max(rmax, rr)
                    cmin = min(cmin, cc); cmax = max(cmax, cc)
                    for dr, dc in neighbors:
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                            visited[nr, nc] = True
                            q.append((nr, nc))
                comps.append({"pixels": pixels, "area": len(pixels), "bbox": (rmin, rmax, cmin, cmax)})
    return comps

def robust_z(x: np.ndarray):
    """median + MAD"""
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return x * np.nan
    med = np.median(valid)
    mad = np.median(np.abs(valid - med))
    denom = (1.4826 * mad) if mad > 1e-9 else (np.std(valid) if np.std(valid) > 1e-9 else 1.0)
    return (x - med) / denom

def classic_z(x: np.ndarray):
    valid = x[~np.isnan(x)]
    if valid.size == 0:
        return x * np.nan
    mu = float(np.mean(valid))
    sd = float(np.std(valid)) if float(np.std(valid)) > 1e-9 else 1.0
    return (x - mu) / sd

def box_blur(img: np.ndarray, k: int = 3):
    """simple mean filter, k odd"""
    if k <= 1:
        return img
    pad = k // 2
    a = np.pad(img, ((pad,pad),(pad,pad)), mode="edge")
    out = np.zeros_like(img, dtype=np.float32)
    # integral image for speed
    s = np.cumsum(np.cumsum(a, axis=0), axis=1)
    h, w = img.shape
    for r in range(h):
        r0 = r
        r1 = r + k
        for c in range(w):
            c0 = c
            c1 = c + k
            total = s[r1, c1] - s[r0, c1] - s[r1, c0] + s[r0, c0]
            out[r, c] = total / (k * k)
    return out

def bbox_from_latlon(lat, lon, cap_m):
    lat_f = cap_m / 111320.0
    lon_f = cap_m / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    return [lon - lon_f, lat - lat_f, lon + lon_f, lat + lat_f]

# =========================
# CACHED TOKEN + FETCH
# =========================
@st.cache_data(ttl=45*60, show_spinner=False)
def cached_token(client_id: str, client_secret: str, username: str | None = None, password: str | None = None):
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    # client_credentials
    try:
        data = {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret}
        r = requests.post(auth_url, data=data, timeout=30)
        if r.status_code == 200:
            return r.json().get("access_token")
    except:
        pass

    # password fallback
    if username and password:
        try:
            data = {"grant_type": "password", "client_id": client_id, "username": username, "password": password}
            r = requests.post(auth_url, data=data, timeout=30)
            if r.status_code == 200:
                return r.json().get("access_token")
        except:
            pass
    return None

@st.cache_data(ttl=30*60, show_spinner=False)
def fetch_s1_tiff(token: str, bbox: list[float], width: int, height: int) -> bytes:
    evalscript = """
    function setup() {
      return { input: ["VV"], output: { id: "default", bands: 1, sampleType: "FLOAT32" } };
    }
    function evaluatePixel(sample) { return [sample.VV]; }
    """
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-1-grd"}],
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }
    res = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=60,
    )
    if res.status_code != 200:
        raise RuntimeError(f"HTTP {res.status_code} | {res.text[:300]}")
    return res.content

# =========================
# GEOLOCATION (MOBILE)
# =========================
# We use query params updated by JS to bring lat/lon into Streamlit
qp = st.query_params

def apply_qp_location():
    try:
        if "glat" in qp and "glon" in qp:
            lat = float(str(qp["glat"]))
            lon = float(str(qp["glon"]))
            st.session_state.lat = lat
            st.session_state.lon = lon
            st.session_state.coord_str = f"{lat:.7f} {lon:.7f}"
    except:
        pass

apply_qp_location()

def geolocation_js(sample_seconds=3.5, min_acc=80):
    # Collect multiple samples for stabilization then update URL query params
    # Filtering by accuracy.
    return f"""
    <script>
    (function() {{
      const waitSec = {float(sample_seconds)};
      const minAcc = {float(min_acc)};
      let samples = [];
      let started = Date.now();

      function done() {{
        if (!samples.length) {{
          alert("Konum alınamadı. Telefon konum izni açık mı?");
          return;
        }}
        // median for stability
        samples.sort((a,b)=>a.lat-b.lat);
        const mLat = samples[Math.floor(samples.length/2)].lat;
        samples.sort((a,b)=>a.lon-b.lon);
        const mLon = samples[Math.floor(samples.length/2)].lon;

        // also compute mean for smoother result
        const meanLat = samples.reduce((s,x)=>s+x.lat,0)/samples.length;
        const meanLon = samples.reduce((s,x)=>s+x.lon,0)/samples.length;

        // final: blend median+mean
        const fLat = (mLat*0.6 + meanLat*0.4);
        const fLon = (mLon*0.6 + meanLon*0.4);

        const url = new URL(window.location.href);
        url.searchParams.set("glat", fLat.toFixed(7));
        url.searchParams.set("glon", fLon.toFixed(7));
        url.searchParams.set("gt", String(Date.now()));
        window.location.href = url.toString();
      }}

      function onPos(pos) {{
        const lat = pos.coords.latitude;
        const lon = pos.coords.longitude;
        const acc = pos.coords.accuracy || 9999;
        if (acc <= minAcc) {{
          samples.push({{lat, lon, acc}});
        }}
        const elapsed = (Date.now() - started) / 1000.0;
        if (elapsed >= waitSec) {{
          navigator.geolocation.clearWatch(wid);
          done();
        }}
      }}

      function onErr(err) {{
        alert("Konum hatası: " + err.message);
      }}

      if (!navigator.geolocation) {{
        alert("Tarayıcı konum desteği yok.");
        return;
      }}

      const wid = navigator.geolocation.watchPosition(onPos, onErr, {{
        enableHighAccuracy: true,
        maximumAge: 0,
        timeout: 15000
      }});
    }})();
    </script>
    <div style="padding:10px;border-radius:10px;background:rgba(0,0,0,0.2);border:1px solid rgba(255,255,255,0.12);">
      📍 Konum alınıyor... (stabilize ediliyor)
    </div>
    """

# =========================
# MAIN HEADER
# =========================
st.markdown("# 🛰️ Turkeller Surfer Pro")
st.caption("Sentinel-1 VV | Mobil uyumlu 2D+3D | TopN Anomali")

# =========================
# CONTROLS (MAIN, MOBILE)
# =========================
with st.container():
    st.markdown('<div class="small-card">', unsafe_allow_html=True)

    coord_in = st.text_input(
        "📌 Koordinat (tek kutu) — örn: `40.1048440 27.7690640`",
        value=st.session_state.coord_str,
        key="coord_input",
    )

    lat_val, lon_val = parse_coord_pair(coord_in)
    st.session_state.coord_str = coord_in

    col1, col2 = st.columns([1, 1])
    with col1:
        cap = st.slider("Tarama Çapı (m)", 20, 300, 50)
    with col2:
        res_opt = st.selectbox("Çözünürlük", [120, 200, 300], index=0)

    col3, col4 = st.columns([1, 1])
    with col3:
        topn = st.slider("TopN", 1, 15, 3)
    with col4:
        anomali_esik = st.slider("Eşik (z)", 1.5, 6.0, 2.8, 0.1)

    col5, col6 = st.columns([1, 1])
    with col5:
        z_mode = st.selectbox("Z türü", ["Robust (Median+MAD)", "Klasik (Mean+Std)"], index=0)
    with col6:
        clip_lo, clip_hi = st.slider("Clip %", 0, 10, (1, 99))

    col7, col8 = st.columns([1, 1])
    with col7:
        smooth_on = st.checkbox("Smoothing (BoxBlur)", value=True)
    with col8:
        smooth_k = st.selectbox("Kernel", [1, 3, 5], index=1, disabled=(not smooth_on))

    col9, col10 = st.columns([1, 1])
    with col9:
        anomaly_view = st.selectbox("Anomali görünümü", ["BBox + Kontur", "Sadece BBox", "Sadece Kontur"], index=0)
    with col10:
        posneg = st.checkbox("Pozitif/Negatif ayır", value=True)

    # location controls
    st.markdown("---")
    cA, cB = st.columns([1, 1])
    with cA:
        use_geo = st.button("📍 Canlı Konumu Çek (stabil)", use_container_width=True)
    with cB:
        clear_focus = st.button("🧹 Odak Temizle", use_container_width=True)

    if clear_focus:
        st.session_state.focus_lat = None
        st.session_state.focus_lon = None
        st.session_state.focus_label = None

    if use_geo:
        st.components.v1.html(geolocation_js(sample_seconds=3.5, min_acc=80), height=70)

    # Apply manual coords if valid
    if lat_val is not None and lon_val is not None:
        st.session_state.lat = float(lat_val)
        st.session_state.lon = float(lon_val)
    else:
        st.warning("⚠️ Koordinat formatı geçersiz. Örn: `40.1048440 27.7690640`")

    st.markdown(
        f"**Mevcut:** `{st.session_state.lat:.7f} {st.session_state.lon:.7f}`",
    )

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# HISTORY (QUICK LOAD)
# =========================
with st.expander("🕘 Son Analizler (History)", expanded=False):
    if not st.session_state.history:
        st.info("Henüz history yok.")
    else:
        labels = [
            f"{i+1}) {h['t']} | {h['lat']:.5f},{h['lon']:.5f} | cap {h['cap']} | res {h['res']} | z {h['thr']}"
            for i, h in enumerate(st.session_state.history[::-1][:10])
        ]
        pick = st.selectbox("Seç", ["—"] + labels, index=0)
        if pick != "—":
            idx = labels.index(pick)
            h = st.session_state.history[::-1][idx]
            st.session_state.lat = h["lat"]
            st.session_state.lon = h["lon"]
            st.session_state.coord_str = f"{h['lat']:.7f} {h['lon']:.7f}"
            st.success("✅ History yüklendi. Analiz butonuna basabilirsin.")

# =========================
# ANALYZE BUTTONS
# =========================
colA, colB = st.columns([1, 1])
with colA:
    analiz_butonu = st.button("🔍 ANALİZ", type="primary", use_container_width=True, disabled=(lat_val is None and ("glat" not in qp)))
with colB:
    ai_yorum_butonu = st.button("🤖 AI YORUM", use_container_width=True, disabled=(st.session_state.Z_data is None))

# =========================
# TOKEN HELPER (DEBUG LIKE)
# =========================
def get_token_fast():
    if "CDSE_CLIENT_ID" not in st.secrets or "CDSE_CLIENT_SECRET" not in st.secrets:
        st.error("❌ Secrets eksik! Settings → Secrets içine CDSE_CLIENT_ID ve CDSE_CLIENT_SECRET ekle.")
        return None
    return cached_token(
        st.secrets["CDSE_CLIENT_ID"],
        st.secrets["CDSE_CLIENT_SECRET"],
        st.secrets.get("CDSE_USERNAME"),
        st.secrets.get("CDSE_PASSWORD"),
    )

# =========================
# ANALYZE
# =========================
def estimate_relative_depth(area_px: int, peak_abs_z: float):
    """
    Sentinel-1 VV'den 'gerçek derinlik (m)' çıkarılamaz.
    Bu yüzden: daha geniş + daha düşük peak = daha "derin" gibi bir GÖRECELİ skor.
    """
    peak = max(peak_abs_z, 1e-6)
    rel = (math.sqrt(max(area_px, 1)) / peak)
    # normalize-ish
    return float(rel)

def save_png_heatmap(Z_db, X, Y, top_list, bbox, out_path="export_map.png"):
    plt.figure(figsize=(6, 6), dpi=160)
    plt.imshow(Z_db, origin="lower")
    plt.title("VV dB Heatmap")
    plt.axis("off")
    # mark centers (approx in pixel coords)
    for i, t in enumerate(top_list, start=1):
        # convert lon/lat -> pixel roughly
        lon = t["center_lon"]; lat = t["center_lat"]
        w = Z_db.shape[1]; h = Z_db.shape[0]
        x = int((lon - bbox[0]) / (bbox[2]-bbox[0]) * (w-1))
        y = int((lat - bbox[1]) / (bbox[3]-bbox[1]) * (h-1))
        plt.scatter([x], [y], s=30)
        plt.text(x, y, f"{i}", fontsize=10)
    plt.tight_layout(pad=0)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path

if analiz_butonu:
    token = get_token_fast()
    if not token:
        st.stop()

    with st.spinner("🛰️ Veri çekiliyor ve analiz ediliyor..."):
        try:
            bbox = bbox_from_latlon(st.session_state.lat, st.session_state.lon, cap)

            # fetch
            tiff_bytes = fetch_s1_tiff(token, bbox, res_opt, res_opt)
            Z = tiff.imread(io.BytesIO(tiff_bytes)).astype(np.float32)

            X, Y = np.meshgrid(
                np.linspace(bbox[0], bbox[2], res_opt),
                np.linspace(bbox[1], bbox[3], res_opt),
            )

            # VV -> dB
            eps = 1e-10
            Z_db = 10.0 * np.log10(np.maximum(Z, eps))

            # clip percentiles
            valid = Z_db[~np.isnan(Z_db)]
            p_lo, p_hi = np.percentile(valid, [clip_lo, clip_hi])
            Z_db_clip = np.clip(Z_db, p_lo, p_hi)

            # smoothing
            if smooth_on and smooth_k > 1:
                Z_db_clip = box_blur(Z_db_clip.astype(np.float32), k=int(smooth_k))

            # z-score
            if z_mode.startswith("Robust"):
                Z_z = robust_z(Z_db_clip)
            else:
                Z_z = classic_z(Z_db_clip)

            # masks
            thr = float(anomali_esik)
            if posneg:
                pos_mask = (Z_z >= thr)
                neg_mask = (Z_z <= -thr)
            else:
                pos_mask = (np.abs(Z_z) >= thr)
                neg_mask = np.zeros_like(pos_mask, dtype=bool)

            comps_pos = connected_components(pos_mask) if np.any(pos_mask) else []
            comps_neg = connected_components(neg_mask) if np.any(neg_mask) else []

            def score_components(comps, sign_label):
                ranked = []
                for comp in comps:
                    pix = comp["pixels"]
                    rr = np.array([p[0] for p in pix], dtype=int)
                    cc = np.array([p[1] for p in pix], dtype=int)

                    # signed peak
                    signed_peak = float(np.max(Z_z[rr, cc])) if sign_label == "POS" else float(np.min(Z_z[rr, cc]))
                    peak_abs = abs(signed_peak)
                    area = int(comp["area"])
                    rmin, rmax, cmin, cmax = comp["bbox"]
                    bbox_area = int((rmax-rmin+1) * (cmax-cmin+1))
                    fill = (area / bbox_area) if bbox_area > 0 else 0.0

                    # stable score
                    score = peak_abs * math.log1p(area) * (0.6 + 0.8*fill)

                    center_lat = float(np.mean(Y[rr, cc]))
                    center_lon = float(np.mean(X[rr, cc]))

                    rel_z = estimate_relative_depth(area, peak_abs)

                    ranked.append({
                        "type": sign_label,
                        "score": float(score),
                        "peak_z": float(signed_peak),
                        "peak_abs": float(peak_abs),
                        "area": area,
                        "fill": float(fill),
                        "bbox_rc": (rmin, rmax, cmin, cmax),
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "rel_depth": float(rel_z),
                    })
                ranked.sort(key=lambda d: d["score"], reverse=True)
                return ranked

            ranked = []
            ranked += score_components(comps_pos, "POS")
            ranked += score_components(comps_neg, "NEG")
            ranked.sort(key=lambda d: d["score"], reverse=True)
            topN = ranked[: int(topn)]

            # store for AI & view
            st.session_state.Z_data = Z_db_clip
            st.session_state.X_data = X
            st.session_state.Y_data = Y

            # history snapshot
            st.session_state.history.append({
                "t": datetime.now().strftime("%d.%m %H:%M"),
                "lat": float(st.session_state.lat),
                "lon": float(st.session_state.lon),
                "cap": int(cap),
                "res": int(res_opt),
                "thr": float(anomali_esik),
            })
            st.session_state.history = st.session_state.history[-30:]

            # pixel uncertainty ~ bbox span / res
            cell_m_x = (bbox[2]-bbox[0]) * (40075000.0 * math.cos(math.radians(st.session_state.lat)) / 360.0) / res_opt
            cell_m_y = (bbox[3]-bbox[1]) * 111320.0 / res_opt
            approx_unc_m = 0.5 * math.sqrt(cell_m_x**2 + cell_m_y**2)

            # =========================
            # PLOTS (MOBILE: STACK)
            # =========================
            st.markdown("---")
            st.subheader("🗺️ 2D Heatmap")

            heat_fig = go.Figure()
            heat_fig.add_trace(go.Heatmap(
                z=Z_db_clip,
                x=X[0, :],
                y=Y[:, 0],
                colorbar=dict(title="VV (dB)"),
                hovertemplate="Lon=%{x:.6f}<br>Lat=%{y:.6f}<br>VV(dB)=%{z:.2f}<extra></extra>"
            ))

            show_bbox = anomaly_view in ["BBox + Kontur", "Sadece BBox"]
            show_cont = anomaly_view in ["BBox + Kontur", "Sadece Kontur"]

            if show_cont:
                # combined mask for contour
                mask_any = (np.abs(Z_z) >= thr) if not posneg else ((Z_z >= thr) | (Z_z <= -thr))
                heat_fig.add_trace(go.Contour(
                    z=mask_any.astype(int),
                    x=X[0, :],
                    y=Y[:, 0],
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1),
                    line=dict(width=2),
                    hoverinfo="skip",
                    name="Anomali Kontur"
                ))

            # TopN shapes + markers
            for i, t in enumerate(topN, start=1):
                rmin, rmax, cmin, cmax = t["bbox_rc"]
                x0 = float(X[0, cmin]); x1 = float(X[0, cmax])
                y0 = float(Y[rmin, 0]); y1 = float(Y[rmax, 0])

                if show_bbox:
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
                    name=f"Top {i}",
                    hovertemplate=(
                        f"<b>#{i}</b> ({t['type']})<br>"
                        "Lon=%{x:.6f}<br>Lat=%{y:.6f}<br>"
                        f"score={t['score']:.2f}<br>"
                        f"peak z={t['peak_z']:.2f}<br>"
                        f"alan={t['area']} px<br>"
                        f"Z(göreceli)={t['rel_depth']:.2f}<extra></extra>"
                    )
                ))

            # Focus marker + zoom
            if st.session_state.focus_lat is not None and st.session_state.focus_lon is not None:
                heat_fig.add_trace(go.Scatter(
                    x=[st.session_state.focus_lon],
                    y=[st.session_state.focus_lat],
                    mode="markers+text",
                    text=[f"ODAK {st.session_state.focus_label or ''}"],
                    textposition="bottom center",
                    marker=dict(size=16, symbol="x"),
                    name="Odak"
                ))
                heat_fig.update_xaxes(range=[st.session_state.focus_lon - (bbox[2]-bbox[0])*0.25,
                                             st.session_state.focus_lon + (bbox[2]-bbox[0])*0.25])
                heat_fig.update_yaxes(range=[st.session_state.focus_lat - (bbox[3]-bbox[1])*0.25,
                                             st.session_state.focus_lat + (bbox[3]-bbox[1])*0.25])

            heat_fig.update_layout(
                height=520 if res_opt <= 200 else 560,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Boylam",
                yaxis_title="Enlem",
                title=f"2D | ±{approx_unc_m:.1f}m piksel belirsizliği (yaklaşık)"
            )
            st.plotly_chart(heat_fig, use_container_width=True)

            st.subheader("🧊 3D Surface")
            surf_fig = go.Figure(data=[go.Surface(
                z=Z_db_clip,
                x=X,
                y=Y,
                hovertemplate="<b>Lon</b> %{x:.6f}<br><b>Lat</b> %{y:.6f}<br><b>VV(dB)</b> %{z:.2f}<extra></extra>"
            )])
            surf_fig.update_layout(
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=0.5),
                    xaxis_title="Boylam",
                    yaxis_title="Enlem",
                    zaxis_title="VV (dB)",
                    camera=dict(eye=dict(x=1.4, y=1.4, z=0.9)),
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                height=520,
                title="3D dB Yüzeyi"
            )
            st.plotly_chart(surf_fig, use_container_width=True)

            # =========================
            # TOPN LIST + ACTIONS
            # =========================
            st.markdown("---")
            st.subheader(f"🎯 Top {topn} Hedef")

            if not topN:
                st.info("Bu eşikte anomali bulunamadı. Eşiği düşürmeyi deneyebilirsin.")
            else:
                for i, t in enumerate(topN, start=1):
                    tag = "🟢 POS" if t["type"] == "POS" else "🔴 NEG"
                    st.markdown(
                        f"**#{i} {tag}** | score=`{t['score']:.2f}` | peak z=`{t['peak_z']:.2f}` | alan=`{t['area']}` px | "
                        f"Z(göreceli)=`{t['rel_depth']:.2f}`"
                    )
                    st.code(f"{t['center_lat']:.8f} {t['center_lon']:.8f}", language="text")

                    c1, c2, c3 = st.columns([1, 1, 1])
                    with c1:
                        if st.button("📍 Anomaliye Git", key=f"goto_{i}", use_container_width=True):
                            st.session_state.focus_lat = t["center_lat"]
                            st.session_state.focus_lon = t["center_lon"]
                            st.session_state.focus_label = f"#{i}"
                            st.rerun()
                    with c2:
                        maps_url = f"https://www.google.com/maps/search/?api=1&query={t['center_lat']},{t['center_lon']}"
                        st.link_button("🌍 Haritada Aç", maps_url, use_container_width=True)
                    with c3:
                        if st.button("🧷 Odakla + Kapat", key=f"focus_close_{i}", use_container_width=True):
                            st.session_state.focus_lat = t["center_lat"]
                            st.session_state.focus_lon = t["center_lon"]
                            st.session_state.focus_label = f"#{i}"
                            st.rerun()

                    st.divider()

            # =========================
            # EXPORTS
            # =========================
            st.markdown("---")
            st.subheader("📤 Export")

            # CSV
            if ranked:
                import pandas as pd
                df = pd.DataFrame(ranked)
                st.download_button(
                    "⬇️ Anomali Listesi CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"anomali_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # NPY raster
            buf = io.BytesIO()
            np.save(buf, Z_db_clip)
            st.download_button(
                "⬇️ dB Raster (NPY)",
                data=buf.getvalue(),
                file_name=f"vv_db_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}.npy",
                mime="application/octet-stream",
                use_container_width=True
            )

            # PNG heatmap (matplotlib)
            png_path = save_png_heatmap(Z_db_clip, X, Y, topN, bbox, out_path="export_map.png")
            with open(png_path, "rb") as f:
                st.download_button(
                    "⬇️ Harita PNG",
                    data=f.read(),
                    file_name=f"map_{st.session_state.lat:.4f}_{st.session_state.lon:.4f}.png",
                    mime="image/png",
                    use_container_width=True
                )

            # =========================
            # STATS
            # =========================
            with st.expander("📊 İstatistik (dB)", expanded=False):
                st.write(f"Ortalama: **{np.mean(Z_db_clip):.2f} dB**")
                st.write(f"Maks: **{np.max(Z_db_clip):.2f} dB**")
                st.write(f"Min: **{np.min(Z_db_clip):.2f} dB**")
                st.write(f"Std: **{np.std(Z_db_clip):.2f} dB**")

            st.success("✅ Analiz tamamlandı!")

        except Exception as e:
            st.error(f"❌ Analiz exception: {e}")

# =========================
# LOCAL AI (UNCHANGED LOGIC, SMALL IMPROVE: mention relative depth)
# =========================
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
- Enlem: {lat:.7f}
- Boylam: {lon:.7f}
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

ℹ️ Not:
- Sentinel-1 VV verisinden gerçek “derinlik (m)” çıkarılamaz.
- Listede gösterilen Z değeri “göreceli derinlik skoru”dur (hedefler arası kıyas içindir).
"""
    return rapor

if ai_yorum_butonu:
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

with st.expander("📁 Geçmiş AI Raporları", expanded=False):
    raporlar = ai_raporlari_yukle()
    if raporlar:
        for i, rapor in enumerate(reversed(raporlar[-5:])):
            st.write(f"**{rapor.get('rapor_adi','Rapor')}**")
            st.caption(f"📍 {rapor['koordinat']['enlem']:.4f}, {rapor['koordinat']['boylam']:.4f} | 📅 {rapor['tarih']}")
            if st.button("👁️ Gör", key=f"gor_{i}", use_container_width=True):
                st.info(rapor["ai_yorum"])
            st.divider()
    else:
        st.info("Henüz kayıtlı AI raporu yok")

st.caption("🛰️ Turkeller Surfer Pro | Mobil uyumlu | TopN + Odak + Export + Cache")

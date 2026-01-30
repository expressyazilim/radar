import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math
import requests
import io
import tifffile as tiff
import re
from collections import deque

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Turkeller Surfer Pro", layout="centered", initial_sidebar_state="collapsed")

# =========================
# CSS
# =========================
st.markdown("""
<style>
.main { background: linear-gradient(135deg,#2c3e50 0%,#34495e 100%) !important; color:white !important; }
.main * { color:white !important; }
.block-container { padding-top:1.2rem !important; padding-bottom:1rem !important; }
.stButton > button { background: linear-gradient(135deg,#5d6d7e 0%,#34495e 100%) !important; border-radius:10px !important; height:46px !important; font-weight:650 !important; width:100% !important; }
.stTextInput input { background:#0d141c !important; color:white !important; border-radius:10px !important; }
.small-card { background:rgba(0,0,0,0.2); border-radius:12px; padding:12px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.10); }
.js-plotly-plot{ background:#1a252f !important; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================
APP_PASSWORD = "altin2026"
if "auth" not in st.session_state:
    st.session_state.auth = False

def login():
    st.markdown("## 🛰️ Turkeller Surfer Pro")
    st.caption("Saha Modu | Otomatik giriş: bu cihazda hatırla")
    st.markdown("---")
    pwd = st.text_input("Şifre", type="password")
    remember = st.checkbox("Bu cihazda hatırla", value=True)

    if st.button("Giriş Yap", use_container_width=True):
        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            if remember:
                st.components.v1.html("""<script>localStorage.setItem("turkeller_auth","1");</script>""", height=0)
            st.rerun()
        else:
            st.error("Hatalı şifre")

    st.components.v1.html("""
    <script>
    try{
      if (localStorage.getItem("turkeller_auth")==="1"){
        const u=new URL(window.location.href);
        if(u.searchParams.get("autologin")!=="1"){
          u.searchParams.set("autologin","1");
          window.location.href=u.toString();
        }
      }
    }catch(e){}
    </script>
    """, height=0)

if not st.session_state.auth:
    if str(st.query_params.get("autologin","")) == "1":
        st.session_state.auth = True
        st.rerun()
    login()
    st.stop()

with st.expander("🔓 Oturum", expanded=False):
    if st.button("Çıkış Yap", use_container_width=True):
        st.components.v1.html("""<script>localStorage.removeItem("turkeller_auth");</script>""", height=0)
        st.session_state.auth = False
        try:
            if "autologin" in st.query_params:
                del st.query_params["autologin"]
        except:
            pass
        st.rerun()

# =========================
# HELPERS
# =========================
def parse_coord_pair(s: str):
    try:
        s = (s or "").strip().replace(",", " ")
        parts = [p for p in s.split() if p.strip()]
        if len(parts) < 2:
            return None, None
        return float(parts[0]), float(parts[1])
    except:
        return None, None

def bbox_from_latlon(lat, lon, cap):
    lat_f = cap / 111320.0
    lon_f = cap / (40075000.0 * math.cos(math.radians(lat)) / 360.0)
    return [lon - lon_f, lat - lat_f, lon + lon_f, lat + lat_f]

def robust_z(x: np.ndarray):
    v = x[~np.isnan(x)]
    if v.size == 0:
        return x * np.nan
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    denom = (1.4826 * mad) if mad > 1e-9 else (np.std(v) if np.std(v) > 1e-9 else 1.0)
    return (x - med) / denom

def box_blur(img: np.ndarray, k: int = 3):
    if k <= 1:
        return img
    pad = k // 2
    a = np.pad(img, ((pad, pad), (pad, pad)), mode="edge").astype(np.float32)
    out = np.empty_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = float(np.mean(a[i:i+k, j:j+k]))
    return out

def connected_components(mask: np.ndarray):
    h, w = mask.shape
    vis = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if mask[r, c] and not vis[r, c]:
                q = deque([(r, c)])
                vis[r, c] = True
                pix = []
                while q:
                    rr, cc = q.popleft()
                    pix.append((rr, cc))
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not vis[nr, nc]:
                                vis[nr, nc] = True
                                q.append((nr, nc))
                comps.append(pix)
    return comps

def weighted_center(rr, cc, z, X, Y):
    w = np.abs(z[rr, cc]).astype(np.float64)
    s = float(np.sum(w))
    if s < 1e-12:
        return float(Y[rr[0], cc[0]]), float(X[rr[0], cc[0]])
    lat = float(np.sum(w * Y[rr, cc]) / s)
    lon = float(np.sum(w * X[rr, cc]) / s)
    return lat, lon

# =========================
# GEOLOCATION (PARENT REDIRECT)
# =========================
def geo_js():
    return """
<script>
(function(){
  let samples=[];
  let start=Date.now();
  function done(){
    if(samples.length===0){alert("Konum alınamadı (izin?)");return;}
    samples.sort((a,b)=>a.lat-b.lat);
    const mLat=samples[Math.floor(samples.length/2)].lat;
    samples.sort((a,b)=>a.lon-b.lon);
    const mLon=samples[Math.floor(samples.length/2)].lon;

    const meanLat=samples.reduce((s,x)=>s+x.lat,0)/samples.length;
    const meanLon=samples.reduce((s,x)=>s+x.lon,0)/samples.length;

    const lat=(mLat*0.6 + meanLat*0.4);
    const lon=(mLon*0.6 + meanLon*0.4);

    const u=new URL(window.parent.location.href);
    u.searchParams.set("glat",lat.toFixed(7));
    u.searchParams.set("glon",lon.toFixed(7));
    u.searchParams.set("gt",String(Date.now()));
    window.parent.location.href=u.toString();
  }
  function ok(p){
    const acc=p.coords.accuracy||9999;
    if(acc<=80) samples.push({lat:p.coords.latitude,lon:p.coords.longitude,acc});
    if(Date.now()-start>3500){ navigator.geolocation.clearWatch(w); done(); }
  }
  function err(e){ alert("Konum hatası: "+e.message); }
  if(!navigator.geolocation){ alert("Konum desteği yok"); return; }
  const w=navigator.geolocation.watchPosition(ok,err,{enableHighAccuracy:true,maximumAge:0,timeout:15000});
})();
</script>
<div style="padding:10px;border-radius:10px;background:rgba(0,0,0,0.25);border:1px solid rgba(255,255,255,0.12);">
📍 Konum alınıyor… (stabilize)
</div>
"""

if "coord" not in st.session_state:
    st.session_state.coord = "40.0000000 27.0000000"

try:
    if "glat" in st.query_params and "glon" in st.query_params:
        st.session_state.coord = f"{float(str(st.query_params['glat'])):.7f} {float(str(st.query_params['glon'])):.7f}"
except:
    pass

# =========================
# TOKEN + FETCH (ROBUST)
# =========================
@st.cache_data(ttl=45*60, show_spinner=False)
def get_token():
    auth_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    if "CDSE_CLIENT_ID" not in st.secrets or "CDSE_CLIENT_SECRET" not in st.secrets:
        raise RuntimeError("Secrets eksik: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET")
    data = {
        "grant_type": "client_credentials",
        "client_id": st.secrets["CDSE_CLIENT_ID"],
        "client_secret": st.secrets["CDSE_CLIENT_SECRET"],
    }
    r = requests.post(auth_url, data=data, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Token alınamadı: HTTP {r.status_code} | {r.text[:250]}")
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("Token yanıtında access_token yok")
    return tok

def fetch_tiff_bytes(token: str, bbox, width=200, height=200):
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
    r = requests.post(
        "https://sh.dataspace.copernicus.eu/api/v1/process",
        headers={"Authorization": f"Bearer {token}"},
        json=payload,
        timeout=60,
    )
    ct = (r.headers.get("Content-Type") or "").lower()
    if r.status_code != 200:
        raise RuntimeError(f"Veri alınamadı: HTTP {r.status_code} | CT={ct} | {r.text[:300]}")
    if ("tiff" not in ct) and (not r.content.startswith(b"II")) and (not r.content.startswith(b"MM")):
        snippet = r.content[:300]
        try:
            snippet = snippet.decode("utf-8", errors="ignore")
        except:
            snippet = str(snippet)
        raise RuntimeError(f"TIFF değil: CT={ct} | içerik örneği: {snippet[:250]}")
    return r.content

# =========================
# SAHA ANALYSIS (2D+3D)
# =========================
def run_saha_analysis(lat_center, lon_center, cap_m, thr, topn):
    # Preset kilitli
    RES = 200
    CLIP_LO, CLIP_HI = 1, 99
    SMOOTH_K = 3
    AUTO_REFINE = True

    token = get_token()

    def single_pass(latc, lonc, capx):
        bbox = bbox_from_latlon(latc, lonc, capx)
        tiff_bytes = fetch_tiff_bytes(token, bbox, width=RES, height=RES)
        Z = tiff.imread(io.BytesIO(tiff_bytes)).astype(np.float32)

        eps = 1e-10
        Zdb = 10.0 * np.log10(np.maximum(Z, eps))

        v = Zdb[~np.isnan(Zdb)]
        p1, p99 = np.percentile(v, [CLIP_LO, CLIP_HI])
        Zdb = np.clip(Zdb, p1, p99)

        Zdb = box_blur(Zdb.astype(np.float32), SMOOTH_K)
        Zz = robust_z(Zdb)

        H, W = Zdb.shape
        X, Y = np.meshgrid(
            np.linspace(bbox[0], bbox[2], W),
            np.linspace(bbox[1], bbox[3], H),
        )

        pos_mask = (Zz >= thr)
        neg_mask = (Zz <= -thr)

        comps_pos = connected_components(pos_mask) if np.any(pos_mask) else []
        comps_neg = connected_components(neg_mask) if np.any(neg_mask) else []

        ranked = []

        def add_comps(comps, typ):
            for pix in comps:
                rr = np.array([p[0] for p in pix], dtype=int)
                cc = np.array([p[1] for p in pix], dtype=int)
                vals = Zz[rr, cc]
                k = int(np.argmax(vals)) if typ == "POS" else int(np.argmin(vals))
                peak_z = float(vals[k])
                peak_abs = abs(peak_z)
                area = int(len(pix))
                score = float(peak_abs * math.log1p(area))

                pr = int(rr[k]); pc = int(cc[k])
                peak_lat = float(Y[pr, pc]); peak_lon = float(X[pr, pc])

                tgt_lat, tgt_lon = weighted_center(rr, cc, Zz, X, Y)

                # 3D için hedefin z değeri: o pikselin Zdb değeri
                tgt_r = int(np.clip(int(np.round((tgt_lat - bbox[1]) / (bbox[3] - bbox[1]) * (H - 1))), 0, H-1))
                tgt_c = int(np.clip(int(np.round((tgt_lon - bbox[0]) / (bbox[2] - bbox[0]) * (W - 1))), 0, W-1))
                tgt_db = float(Zdb[tgt_r, tgt_c])

                ranked.append({
                    "type": typ,
                    "score": score,
                    "peak_z": peak_z,
                    "area": area,
                    "peak_lat": peak_lat,
                    "peak_lon": peak_lon,
                    "target_lat": float(tgt_lat),
                    "target_lon": float(tgt_lon),
                    "target_db": tgt_db,
                })

        add_comps(comps_pos, "POS")
        add_comps(comps_neg, "NEG")
        ranked.sort(key=lambda d: d["score"], reverse=True)
        return bbox, Zdb, X, Y, Zz, ranked

    bbox1, Zdb1, X1, Y1, Zz1, ranked1 = single_pass(lat_center, lon_center, cap_m)
    top1 = ranked1[0] if ranked1 else None

    if AUTO_REFINE and top1 and cap_m > 25:
        cap2 = max(20, min(30, int(cap_m * 0.5)))
        bbox2, Zdb2, X2, Y2, Zz2, ranked2 = single_pass(top1["target_lat"], top1["target_lon"], cap2)
        return True, cap2, bbox2, Zdb2, X2, Y2, Zz2, ranked2[:topn]
    else:
        return False, None, bbox1, Zdb1, X1, Y1, Zz1, ranked1[:topn]

# =========================
# UI – SAHA MODU
# =========================
st.markdown("# 🛰️ Turkeller Surfer Pro")
st.caption("Saha Modu | 2D + 3D | Robust + Smooth(k=3) + Clip 1–99 + Res 200 + OtoRefine")

with st.container():
    st.markdown('<div class="small-card">', unsafe_allow_html=True)

    coord = st.text_input("Koordinat (tek kutu)", st.session_state.coord)
    lat, lon = parse_coord_pair(coord)
    if lat is not None and lon is not None:
        st.session_state.coord = coord

    cap = st.slider("Tarama Çapı (m)", 20, 300, 50)
    esik = st.slider("Anomali Eşiği (z)", 2.0, 2.4, 2.1, 0.1)
    topn = st.slider("TopN", 1, 10, 5)

    colg1, colg2 = st.columns([1, 1])
    with colg1:
        if st.button("📍 Canlı Konum (stabil)", use_container_width=True):
            st.components.v1.html(geo_js(), height=90)
    with colg2:
        st.write("")
        st.write(f"**Mevcut:** `{st.session_state.coord}`")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# ANALYZE BUTTON
# =========================
if st.button("🔍 ANALİZ", use_container_width=True):
    if lat is None or lon is None:
        st.error("Koordinat geçersiz. Örn: 40.1048440 27.7690640")
    else:
        with st.spinner("🛰️ Veri çekiliyor..."):
            try:
                refined, cap2, bbox, Zdb, X, Y, Zz, top = run_saha_analysis(lat, lon, cap, float(esik), int(topn))

                if refined:
                    st.success(f"✅ Oto Refine: Top1 merkezine {cap2}m ile tekrar tarandı.")

                # ----------------- 2D
                st.subheader("🗺️ 2D Heatmap (VV dB)")
                fig2d = go.Figure()
                fig2d.add_trace(go.Heatmap(z=Zdb, x=X[0, :], y=Y[:, 0], colorbar=dict(title="VV (dB)")))
                for i, t in enumerate(top, start=1):
                    fig2d.add_trace(go.Scatter(
                        x=[t["target_lon"]],
                        y=[t["target_lat"]],
                        mode="markers+text",
                        text=[f"#{i}"],
                        textposition="top center",
                        marker=dict(size=10),
                        name=f"#{i}"
                    ))
                fig2d.update_layout(height=520, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig2d, use_container_width=True)

                # ----------------- 3D
                st.subheader("🧊 3D Surface (VV dB) + Hedefler")
                surf = go.Surface(
                    z=Zdb,
                    x=X,
                    y=Y,
                    hovertemplate="<b>Lon</b>:%{x:.6f}<br><b>Lat</b>:%{y:.6f}<br><b>VV(dB)</b>:%{z:.2f}<extra></extra>"
                )
                fig3d = go.Figure(data=[surf])

                # hedefleri 3D scatter olarak koy
                if top:
                    xs = [t["target_lon"] for t in top]
                    ys = [t["target_lat"] for t in top]
                    zs = [t.get("target_db", float("nan")) for t in top]
                    labels = [f"#{i} {t['type']} z={t['peak_z']:.2f}" for i, t in enumerate(top, start=1)]
                    fig3d.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode="markers+text",
                        text=[f"#{i}" for i in range(1, len(top)+1)],
                        textposition="top center",
                        marker=dict(size=4),
                        hovertext=labels,
                        hoverinfo="text"
                    ))

                fig3d.update_layout(
                    height=520,
                    margin=dict(l=0, r=0, t=30, b=0),
                    scene=dict(
                        xaxis_title="Boylam",
                        yaxis_title="Enlem",
                        zaxis_title="VV (dB)",
                        aspectratio=dict(x=1, y=1, z=0.55),
                        camera=dict(eye=dict(x=1.4, y=1.4, z=0.8))
                    )
                )
                st.plotly_chart(fig3d, use_container_width=True)

                # ----------------- List
                st.markdown("---")
                st.subheader(f"🎯 Top {len(top)} (Saha Modu)")
                if not top:
                    st.info("Bu eşikte hedef çıkmadı. Eşiği 2.0’a çekmeyi dene.")
                else:
                    for i, t in enumerate(top, start=1):
                        tag = "🟢 POS" if t["type"] == "POS" else "🔴 NEG"
                        st.markdown(f"**#{i} {tag}** | score=`{t['score']:.2f}` | peak z=`{t['peak_z']:.2f}` | alan=`{t['area']}` px | dB@target=`{t.get('target_db',0):.2f}`")
                        st.code(f"{t['target_lat']:.8f} {t['target_lon']:.8f}", language="text")
                        maps_url = f"https://www.google.com/maps/search/?api=1&query={t['target_lat']},{t['target_lon']}"
                        st.link_button("🌍 Haritada Aç", maps_url, use_container_width=True)
                        with st.expander("Debug (Peak koordinat)", expanded=False):
                            st.code(f"{t['peak_lat']:.8f} {t['peak_lon']:.8f}", language="text")
                        st.divider()

            except Exception as e:
                st.error(f"❌ Analiz hata: {e}")

st.caption("Turkeller Surfer Pro | Saha Modu | 2D+3D aktif")

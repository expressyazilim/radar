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

# =========================
# PAGE CONFIG – SAHA MODU
# =========================
st.set_page_config(
    page_title="Turkeller Surfer Pro",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# =========================
# CSS – MOBILE SAFE
# =========================
st.markdown("""
<style>
.main { background: linear-gradient(135deg,#2c3e50 0%,#34495e 100%) !important; color:white !important; }
.main * { color:white !important; }
.block-container { padding-top:1.2rem !important; padding-bottom:1rem !important; }
.stButton > button {
    background: linear-gradient(135deg,#5d6d7e 0%,#34495e 100%) !important;
    border-radius:10px !important;
    height:46px !important;
    font-weight:600 !important;
}
.stTextInput input {
    background:#0d141c !important;
    color:white !important;
    border-radius:10px !important;
}
.small-card {
    background:rgba(0,0,0,0.2);
    border-radius:12px;
    padding:12px;
    margin-bottom:10px;
}
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
    st.markdown("---")
    pwd = st.text_input("Şifre", type="password")
    remember = st.checkbox("Bu cihazda hatırla", value=True)

    if st.button("Giriş Yap", use_container_width=True):
        if pwd == APP_PASSWORD:
            st.session_state.auth = True
            if remember:
                st.components.v1.html("""
                <script>
                localStorage.setItem("turkeller_auth","1");
                </script>
                """, height=0)
            st.rerun()
        else:
            st.error("Hatalı şifre")

    st.components.v1.html("""
    <script>
    if (localStorage.getItem("turkeller_auth")==="1"){
        window.location.search="?autologin=1";
    }
    </script>
    """, height=0)

if not st.session_state.auth:
    if st.query_params.get("autologin")=="1":
        st.session_state.auth=True
        st.rerun()
    login()
    st.stop()

# =========================
# HELPERS
# =========================
def parse_coord_pair(s):
    try:
        a=s.replace(","," ").split()
        return float(a[0]), float(a[1])
    except:
        return None,None

def bbox_from_latlon(lat,lon,cap):
    lat_f=cap/111320.0
    lon_f=cap/(40075000.0*math.cos(math.radians(lat))/360.0)
    return [lon-lon_f, lat-lat_f, lon+lon_f, lat+lat_f]

def robust_z(x):
    v=x[~np.isnan(x)]
    med=np.median(v)
    mad=np.median(np.abs(v-med))
    d=1.4826*mad if mad>1e-9 else np.std(v)
    return (x-med)/(d if d>1e-9 else 1)

def box_blur(img,k=3):
    if k<=1: return img
    pad=k//2
    a=np.pad(img,((pad,pad),(pad,pad)),'edge')
    out=np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i,j]=np.mean(a[i:i+k,j:j+k])
    return out

def connected_components(mask):
    h,w=mask.shape
    vis=np.zeros_like(mask,bool)
    comps=[]
    for r in range(h):
        for c in range(w):
            if mask[r,c] and not vis[r,c]:
                q=[(r,c)]
                vis[r,c]=True
                pix=[]
                while q:
                    rr,cc=q.pop()
                    pix.append((rr,cc))
                    for dr in (-1,0,1):
                        for dc in (-1,0,1):
                            nr,nc=rr+dr,cc+dc
                            if 0<=nr<h and 0<=nc<w and mask[nr,nc] and not vis[nr,nc]:
                                vis[nr,nc]=True
                                q.append((nr,nc))
                comps.append(pix)
    return comps

def weighted_peak(rr,cc,z,X,Y):
    w=np.abs(z[rr,cc])
    s=np.sum(w)
    if s<1e-9:
        return float(Y[rr[0],cc[0]]),float(X[rr[0],cc[0]])
    return float(np.sum(w*Y[rr,cc])/s), float(np.sum(w*X[rr,cc])/s)

# =========================
# GEOLOCATION (FIXED)
# =========================
def geo_js():
    return """
<script>
(function(){
let samples=[];
let start=Date.now();
function done(){
 if(samples.length===0){alert("Konum alınamadı");return;}
 samples.sort((a,b)=>a.lat-b.lat);
 let lat=samples[Math.floor(samples.length/2)].lat;
 samples.sort((a,b)=>a.lon-b.lon);
 let lon=samples[Math.floor(samples.length/2)].lon;
 let u=new URL(window.parent.location.href);
 u.searchParams.set("glat",lat.toFixed(7));
 u.searchParams.set("glon",lon.toFixed(7));
 window.parent.location.href=u.toString();
}
function ok(p){
 if(p.coords.accuracy<80)
   samples.push({lat:p.coords.latitude,lon:p.coords.longitude});
 if(Date.now()-start>3500){navigator.geolocation.clearWatch(w);done();}
}
function err(e){alert("Konum hatası");}
let w=navigator.geolocation.watchPosition(ok,err,{enableHighAccuracy:true});
})();
</script>
<div>📍 Konum alınıyor…</div>
"""

if "glat" in st.query_params:
    st.session_state.coord=f"{st.query_params['glat']} {st.query_params['glon']}"

# =========================
# UI – SAHA MODU
# =========================
st.markdown("# 🛰️ Turkeller Surfer Pro")
st.caption("SAHA MODU – Kalibre edilmiş")

if "coord" not in st.session_state:
    st.session_state.coord="40.0000000 27.0000000"

with st.container():
    st.markdown('<div class="small-card">',unsafe_allow_html=True)
    coord=st.text_input("Koordinat",st.session_state.coord)
    lat,lon=parse_coord_pair(coord)
    if lat: st.session_state.coord=coord

    cap=st.slider("Tarama Çapı (m)",20,300,50)
    esik=st.slider("Anomali Eşiği (z)",2.0,2.4,2.1,0.1)
    topn=st.slider("TopN",1,10,5)

    if st.button("📍 Canlı Konum",use_container_width=True):
        st.components.v1.html(geo_js(),height=80)

    st.markdown('</div>',unsafe_allow_html=True)

# =========================
# ANALYZE
# =========================
if st.button("🔍 ANALİZ",use_container_width=True):
    if lat is None:
        st.error("Koordinat geçersiz")
    else:
        with st.spinner("Analiz..."):
            token_url="https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            data={"grant_type":"client_credentials",
                  "client_id":st.secrets["CDSE_CLIENT_ID"],
                  "client_secret":st.secrets["CDSE_CLIENT_SECRET"]}
            t=requests.post(token_url,data=data).json()["access_token"]

            bbox=bbox_from_latlon(lat,lon,cap)
            payload={
              "input":{"bounds":{"bbox":bbox,"properties":{"crs":"http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
                       "data":[{"type":"sentinel-1-grd"}]},
              "output":{"width":200,"height":200,"responses":[{"identifier":"d","format":{"type":"image/tiff"}}]},
              "evalscript":"function setup(){return{input:['VV'],output:{bands:1}};}function evaluatePixel(s){return[s.VV];}"
            }
            r=requests.post("https://sh.dataspace.copernicus.eu/api/v1/process",
                headers={"Authorization":f"Bearer {t}"},json=payload)
            Z=tiff.imread(io.BytesIO(r.content)).astype(np.float32)
            Zdb=10*np.log10(np.maximum(Z,1e-9))
            Zdb=box_blur(Zdb,3)
            Zz=robust_z(Zdb)

            mask=Zz>=esik
            comps=connected_components(mask)

            X,Y=np.meshgrid(np.linspace(bbox[0],bbox[2],Z.shape[1]),
                            np.linspace(bbox[1],bbox[3],Z.shape[0]))

            ranked=[]
            for pix in comps:
                rr=np.array([p[0] for p in pix])
                cc=np.array([p[1] for p in pix])
                k=np.argmax(Zz[rr,cc])
                lat_t,lon_t=weighted_peak(rr,cc,Zz,X,Y)
                score=float(np.max(Zz[rr,cc])*math.log(len(pix)+1))
                ranked.append({"lat":lat_t,"lon":lon_t,"score":score})

            ranked=sorted(ranked,key=lambda x:x["score"],reverse=True)[:topn]

            fig=go.Figure(go.Heatmap(z=Zdb,x=X[0],y=Y[:,0]))
            for i,tg in enumerate(ranked,1):
                fig.add_trace(go.Scatter(x=[tg["lon"]],y=[tg["lat"]],
                    mode="markers+text",text=[f"#{i}"]))
            st.plotly_chart(fig,use_container_width=True)

            for i,tg in enumerate(ranked,1):
                st.success(f"#{i}  {tg['lat']:.7f} {tg['lon']:.7f}  score={tg['score']:.2f}")
                st.link_button("Haritada Aç",
                    f"https://www.google.com/maps/search/?api=1&query={tg['lat']},{tg['lon']}")

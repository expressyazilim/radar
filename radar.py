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
    page_title="Turkeller Surfer Pro", # Tarayıcı sekme adı değiştirildi
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

# --- GÜVENLİK ---
def check_password():
    if "password_correct" not in st.session_state:
        # Giriş ekranı başlığı değiştirildi
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

if check_password():
    # --- VERİ TABANI (JSON) ---
    DB_FILE = "kayitli_yerler.json"

    def yerleri_yukle():
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "r") as f:
                    return json.load(f)
            except: return []
        return []

    def yer_kaydet(isim, lat, lon):
        yerler = yerleri_yukle()
        yerler.append({"isim": isim, "lat": lat, "lon": lon})
        with open(DB_FILE, "w") as f:
            json.dump(yerler, f)

    def yer_sil(index):
        yerler = yerleri_

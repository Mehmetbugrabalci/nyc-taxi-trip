# app.py
import streamlit as st

st.set_page_config(
    page_title="NYC Taxi Süre Tahmin",
    layout="wide",
    initial_sidebar_state="expanded"
)


import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# eklediklerimiz:
import matplotlib.pyplot as plt
import seaborn as sns

from streamlit_folium import st_folium
import folium
import pydeck as pdk

import gdown
import joblib
import streamlit as st
import os

MODEL_DRIVE_ID = "1KlmM2dKvn93BLSolqap3ScQBGg3Q7GBh"
MODEL_FILE = "model_lgb_log.pkl"

@st.cache_resource
def load_model_from_drive(drive_id, dest_path):
    if not os.path.exists(dest_path):
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, dest_path, quiet=False)
    return joblib.load(dest_path)

model = load_model_from_drive(MODEL_DRIVE_ID, MODEL_FILE)

# proje kök dizini
ROOT = os.path.abspath(os.path.dirname(__file__))

# sidebar menüsü
menu = st.sidebar.selectbox(
    "📂 Menü",
    ["Ana Sayfa", "Görselleştirme Sayfası", "Sunum Sayfası", "Predict"]
)

# ortak importlar
model = joblib.load(os.path.join(ROOT, "model_lgb_log.pkl"))
from src.preprocessing import clean
from src.features import (
    add_distance,
    add_bearing,
    add_distance_to_center,
    add_is_weekend,
    add_time_features,
    add_location_clusters
)

# --- 1) Ana Sayfa ---
if menu == "Ana Sayfa":
    st.title("🗽 NYC Taxi Yolculuk Süresi Tahmin")
    st.write(
        """
        Bu uygulama, NYC taksi yolculuk verisi üzerinde eğittiğimiz LightGBM modelini
        kullanarak bir yolculuğun tahmini süresini (dakika cinsinden) hesaplar.
        Harita üzerinden alış ve bırakış noktalarını seçip tarih-saat ve yolcu sayısını girdikten sonra,
        “Tahmin Et” butonuna basarak sonucu görebilirsiniz.
        """
    )

# --- 2) Görselleştirme Sayfası ---
elif menu == "Görselleştirme Sayfası":
    st.title("📊 Görselleştirme & EDA")

    # parquet'ten yükle ve temizle
    df = pd.read_parquet(
        os.path.join(ROOT, "data", "processed", "train_sample.parquet")
    )
    df = clean(df)
    df = add_distance(df)  # distance_km eksikse ekle

    # 1️⃣ Trip Duration dağılımı
    st.subheader("1️⃣ Trip Duration Dağılımı (log ölçekte)")
    fig1, ax1 = plt.subplots()
    ax1.hist(np.log1p(df["trip_duration"]), bins=50)
    ax1.set_xlabel("log(trip_duration + 1)")
    ax1.set_ylabel("Frekans")
    st.pyplot(fig1)
    plt.close(fig1)

    # 2️⃣ Pickup Lokasyonları
    st.subheader("2️⃣ Pickup Lokasyonları (örnek 5k nokta)")
    sample = df.sample(n=min(5000, len(df)), random_state=42)
    m = folium.Map(location=[40.75, -73.97], zoom_start=11)
    for _, row in sample.iterrows():
        folium.CircleMarker(
            [row.pickup_latitude, row.pickup_longitude],
            radius=1, color="blue", fill=True, fill_opacity=0.3
        ).add_to(m)
    st_folium(m, width=700, height=400)

    # 3️⃣ Dropoff Lokasyonları
    st.subheader("3️⃣ Dropoff Lokasyonları (örnek 5k nokta)")
    sample2 = df.sample(n=min(5000, len(df)), random_state=24)
    m2 = folium.Map(location=[40.75, -73.97], zoom_start=11)
    for _, row in sample2.iterrows():
        folium.CircleMarker(
            [row.dropoff_latitude, row.dropoff_longitude],
            radius=1, color="red", fill=True, fill_opacity=0.3
        ).add_to(m2)
    st_folium(m2, width=700, height=400)


    # 4️⃣ Mesafe Dağılımı (0–50 km)
    st.subheader("4️⃣ Mesafe Dağılımı (km)")
    fig4, ax4 = plt.subplots()
    ax4.hist(df["distance_km"], bins=50, range=(0, 50))
    ax4.set_xlim(0, 50)
    ax4.set_xlabel("Mesafe (km)")
    ax4.set_ylabel("Frekans")
    st.pyplot(fig4)

    # 5️⃣ Trip Duration vs. Distance (0–50 km)
    st.subheader("5️⃣ Trip Duration vs. Distance")

    # distance ≤ 50 km'yi al, sample ile biraz küçült (örneğin 5k nokta)
    df_scatter = df[df["distance_km"] <= 50].sample(n=min(5000, len(df)), random_state=42)

    # Süreyi dakikaya çevir
    df_scatter["duration_min"] = df_scatter["trip_duration"] / 60

    fig5, ax5 = plt.subplots()
    ax5.scatter(
        df_scatter["distance_km"],
        df_scatter["duration_min"],
        s=5,  # nokta boyutu
        alpha=0.3,  # saydamlık
    )
    ax5.set_xlim(0, 50)
    ax5.set_xlabel("Distance (km)")
    ax5.set_ylabel("Duration (dk)")
    ax5.set_title("Trip Duration vs. Distance (0–50 km)")
    st.pyplot(fig5)

    # 6️⃣ Saat Dilimine Göre Ortalama Süre
    st.subheader("6️⃣ Saat Dilimine Göre Ortalama Süre")
    df["hour"] = df["pickup_datetime"].dt.hour
    avg_by_hour = df.groupby("hour")["trip_duration"].mean() / 60
    fig5, ax5 = plt.subplots()
    ax5.plot(avg_by_hour.index, avg_by_hour.values, marker="o")
    ax5.set_xlabel("Saat")
    ax5.set_ylabel("Ortalama Süre (dk)")
    st.pyplot(fig5)
    plt.close(fig5)

    # 7️⃣ Haftanın Gününe Göre Yolculuk Sayısı
    st.subheader("7️⃣ Haftanın Gününe Göre Yolculuk Sayısı")
    df["weekday"] = df["pickup_datetime"].dt.day_name(locale="en_US")
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts = df["weekday"].value_counts().reindex(order)
    fig6, ax6 = plt.subplots()
    ax6.bar(counts.index, counts.values)
    ax6.set_xlabel("Gün")
    ax6.set_ylabel("Yolculuk Adedi")
    plt.xticks(rotation=45)
    st.pyplot(fig6)
    plt.close(fig6)

# --- 3) Sunum Sayfası ---
elif menu == "Sunum Sayfası":
    st.title("📑 Proje Sunum")

    # --- 3.0: Sunum için veri yükleme ve özellik mühendisliği ---
    df = pd.read_parquet(os.path.join(ROOT, "data", "processed", "train_sample.parquet"))
    df = clean(df)
    df = add_distance(df)
    df = add_bearing(df)
    df = add_distance_to_center(df)
    df = add_is_weekend(df)
    df = add_time_features(df)
    df = add_location_clusters(df)

    # 3.1 Proje Özeti
    st.header("🔍 Proje Özeti")
    st.markdown(
        """
        **Amaç:** NYC taksi yolculuk verisi kullanarak, yolculuk sürelerini dakikа cinsinden tahmin eden bir web uygulaması.  
        **Model:** LightGBM Regressor – en düşük RMSLE = 0.16  
        **Veri Seti:** ~1.5M kayıt, 2019 NYC taksi yolculukları.  
        **Özellikler:** Coğrafi, zaman ve bölgesel kümeleme temelli mühendislik değişkenleri.  
        """
    )

    # 3.2 Veri Setindeki Değişkenler
    st.header("🗂️ Veri Setindeki Değişkenler")
    st.markdown("Orijinal ve yeni üretilen tüm değişkenler:")
    st.table({
        "Özellik": [
            "pickup_datetime","pickup_latitude","pickup_longitude",
            "dropoff_latitude","dropoff_longitude","passenger_count",
            "distance_km","bearing","dist_to_center_km",
            "is_weekend","hour_sin","hour_cos","dow_sin","dow_cos",
            "pickup_cluster","dropoff_cluster"
        ],
        "Açıklama": [
            "Yolculuğun başlangıç zamanı",
            "Başlangıç enlemi","Başlangıç boylamı",
            "Bitiş enlemi","Bitiş boylamı",
            "Yolcu sayısı",
            "Kuş uçuşu mesafesi (km)",
            "Yön (derece)",
            "Merkeze uzaklık (km)",
            "Hafta sonu mu (1/0)",
            "Saatin sin dönüşümü","Saatin cos dönüşümü",
            "Günün sin dönüşümü","Günün cos dönüşümü",
            "Başlangıç noktası küme etiketi",
            "Bitiş noktası küme etiketi"
        ]
    })

    # 3.3 Korelasyon Matrisi
    st.header("📈 Korelasyon Matrisi")
    feat_cols = [
        "trip_duration","distance_km","bearing","dist_to_center_km",
        "is_weekend","hour_sin","hour_cos","dow_sin","dow_cos",
        "pickup_cluster","dropoff_cluster","passenger_count"
    ]
    corr = df[feat_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Öznitelikler ve Hedef Arasındaki Korelasyon")
    st.pyplot(fig)
    st.write(
        "- **distance_km** ile süre arasında güçlü pozitif ilişki.\n"
        "- Zaman ve kümeleme özellikleri de anlamlı korelasyonlar gösteriyor."
    )

    # 3.4 Özellik Mühendisliği
    st.header("🛠️ Özellik Mühendisliği")
    st.markdown(
        """
        - **distance_km**: Mesafe-süre ilişkisinin temeli.  
        - **bearing**: Farklı yol koşullarını yakalar.  
        - **dist_to_center_km**: Merkez trafiğini özetler.  
        - **is_weekend**: Hafta içi/sonu trafik farkı.  
        - **hour_sin**, **hour_cos**, **dow_sin**, **dow_cos**: Zamanın döngüselliği.  
        - **pickup_cluster**, **dropoff_cluster**: Bölgesel yoğunlukları özetler.
        """
    )

    # 3.5 Model Seçimi ve Değerlendirme
    st.header("🤖 Model Seçimi ve Değerlendirme")
    st.markdown("Denenen modeller ve **RMSLE** skorları:")
    model_scores = {
        "Linear Regression": 0.45,
        "Decision Tree":      0.32,
        "Random Forest":      0.24,
        "Gradient Boosting":  0.19,
        "LightGBM":           0.16
    }
    score_df = pd.DataFrame.from_dict(model_scores, orient="index", columns=["RMSLE"])
    fig, ax = plt.subplots(figsize=(6,4))
    score_df.sort_values("RMSLE", ascending=False).plot.barh(ax=ax, legend=False)
    ax.set_xlabel("RMSLE")
    ax.set_ylabel("Model")
    ax.set_title("Modellerin Karşılaştırmalı Performansı")
    st.pyplot(fig)
    st.write(
        "- LightGBM en düşük RMSLE ile en iyi performansı verdi.\n"
        "- Ağaç tabanlı yöntemler lineerden belirgin şekilde önde."
    )

    # 3.6 Feature Importance
    st.header("🌟 Feature Importance (LightGBM)")
    fi = model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feat_cols[1:],  # drop trip_duration
        "importance": fi
    }).sort_values("importance", ascending=False)
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(data=fi_df, x="importance", y="feature", ax=ax)
    ax.set_title("LightGBM Öznitelik Önemleri")
    st.pyplot(fig)
    st.write(
        "- **distance_km** ve **dist_to_center_km** en yüksek katkı.\n"
        "- Zaman ve kümeleme özellikleri de modelin doğruluğunu artırdı."
    )


# --- 4) Predict Sayfası ---
else:
    st.title("🗺️ Nokta Seçimi & Tahmin Et")
    col1, col2 = st.columns([1, 2], gap="large")

    # 4.1 kullanıcı girdileri
    with col1:
        st.subheader("Girdiler")
        pickup_dt       = st.date_input("Alış Tarihi")
        pickup_tm       = st.time_input("Alış Saati")
        passenger_count = st.number_input("Yolcu Sayısı", min_value=1, max_value=6, value=1)
        predict_btn     = st.button("Tahmin Et")

    # 4.2 haritalar
    if "pu" not in st.session_state:  st.session_state.pu = None
    if "do" not in st.session_state:  st.session_state.do = None

    with col2:
        left, right = st.columns(2, gap="small")

        with left:
            st.markdown("**1️⃣ Biniş Noktası**")
            m1 = folium.Map(location=[40.75, -73.97], zoom_start=12)
            folium.TileLayer("cartodbpositron").add_to(m1)
            m1.add_child(folium.LatLngPopup())
            pu_data = st_folium(m1, width=300, height=300, key="pu_map")
            if pu_data and pu_data.get("last_clicked"):
                st.session_state.pu = (
                    pu_data["last_clicked"]["lat"],
                    pu_data["last_clicked"]["lng"]
                )
            if st.session_state.pu:
                st.success(f"Biniş: {st.session_state.pu}")

        with right:
            st.markdown("**2️⃣ İniş Noktası**")
            m2 = folium.Map(location=[40.75, -73.97], zoom_start=12)
            folium.TileLayer("cartodbpositron").add_to(m2)
            m2.add_child(folium.LatLngPopup())
            do_data = st_folium(m2, width=300, height=300, key="do_map")
            if do_data and do_data.get("last_clicked"):
                st.session_state.do = (
                    do_data["last_clicked"]["lat"],
                    do_data["last_clicked"]["lng"]
                )
            if st.session_state.do:
                st.success(f"İniş: {st.session_state.do}")

    # 4.3 tahmin
    st.markdown("---")
    if predict_btn:
        if not st.session_state.pu or not st.session_state.do:
            st.error("Lütfen hem biniş hem de iniş noktalarını seçin.")
        else:
            # hazırlık
            dt = pd.to_datetime(f"{pickup_dt} {pickup_tm}")
            df_in = pd.DataFrame([{
                "pickup_datetime":   dt,
                "pickup_latitude":   st.session_state.pu[0],
                "pickup_longitude":  st.session_state.pu[1],
                "dropoff_latitude":  st.session_state.do[0],
                "dropoff_longitude": st.session_state.do[1],
                "passenger_count":   passenger_count
            }])
            # özellikler
            df_in = add_distance(df_in)
            df_in = add_bearing(df_in)
            df_in = add_distance_to_center(df_in)
            df_in = add_is_weekend(df_in)
            df_in = add_time_features(df_in)
            df_in = add_location_clusters(df_in)

            feats = [
                "distance_km","bearing","dist_to_center_km",
                "is_weekend","hour_sin","hour_cos","dow_sin","dow_cos",
                "pickup_cluster","dropoff_cluster","passenger_count"
            ]
            X = df_in[feats].values

            # tahmin
            with st.spinner("Model çalışıyor…"):
                logp    = model.predict(X)[0]
                seconds = np.expm1(logp)
                minutes = seconds / 60

            st.success("✅ Tahmin Hazır!")
            c1, c2 = st.columns(2, gap="medium")
            c1.metric("⏱ Süre (dk)", f"{minutes:.1f}")
            c2.metric("🛣 Mesafe (km)", f"{df_in.distance_km.values[0]:.2f}")

            # yol çizimi (kuş uçuşu)
            df_pu = pd.DataFrame([{"lat": st.session_state.pu[0], "lon": st.session_state.pu[1]}])
            df_do = pd.DataFrame([{"lat": st.session_state.do[0], "lon": st.session_state.do[1]}])
            path = pd.DataFrame([{
                "path": [
                    [st.session_state.pu[1], st.session_state.pu[0]],
                    [st.session_state.do[1], st.session_state.do[0]]
                ]
            }])

            layers = [
                pdk.Layer("ScatterplotLayer", df_pu, get_position=["lon","lat"], get_radius=100, get_fill_color=[0,128,255]),
                pdk.Layer("ScatterplotLayer", df_do, get_position=["lon","lat"], get_radius=100, get_fill_color=[255,50,50]),
                pdk.Layer("PathLayer", path, get_path="path", get_width=4, get_color=[0,255,0])
            ]
            view = pdk.ViewState(
                latitude=(st.session_state.pu[0]+st.session_state.do[0])/2,
                longitude=(st.session_state.pu[1]+st.session_state.do[1])/2,
                zoom=12
            )
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=view, map_style="mapbox://styles/mapbox/dark-v10"))

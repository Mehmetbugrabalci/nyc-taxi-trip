import os
import sys
import pandas as pd
import joblib
from sklearn.cluster import MiniBatchKMeans

# Proje kökünü path’e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.preprocessing import load_and_sample, clean

if __name__ == "__main__":
    # 1) Veri yükle ve temizle (1M örnek)
    csv_path = os.path.join(ROOT, "data", "raw", "NYC.csv")
    df = load_and_sample(csv_path, n=1_000_000)
    df = clean(df)

    # 2) Koordinatları hazırla
    pickup_coords = df[["pickup_latitude", "pickup_longitude"]].values
    dropoff_coords = df[["dropoff_latitude", "dropoff_longitude"]].values

    # 3) MiniBatchKMeans ile kümele
    km_pickup = MiniBatchKMeans(n_clusters=10, batch_size=100_000, random_state=42)
    km_dropoff = MiniBatchKMeans(n_clusters=10, batch_size=100_000, random_state=42)
    km_pickup.fit(pickup_coords)
    km_dropoff.fit(dropoff_coords)

    # 4) Model dosyalarını kaydet
    model_dir = os.path.join(ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(km_pickup, os.path.join(model_dir, "kmeans_pickup.pkl"))
    joblib.dump(km_dropoff, os.path.join(model_dir, "kmeans_dropoff.pkl"))
    print("KMeans modelleri kaydedildi → models/kmeans_pickup.pkl & kmeans_dropoff.pkl")

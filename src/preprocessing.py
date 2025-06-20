import pandas as pd
import numpy as np

# src/preprocessing.py
def load_and_sample(filepath, n=None, random_state=42):
    """
    CSV veya Parquet formatındaki veriyi yükler,
    eğer n belirtilmişse rastgele en fazla n örneğini alır.
    """
    # 1) Dosyayı oku
    if filepath.lower().endswith((".parquet", ".parq")):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath)

    # 2) Eğer n belirtildiyse ve df uzunluğu > 0 ise örnekle
    if n is not None and n > 0:
        # Popülasyondan büyük örnek istemezsek, sample_size = min(n, len(df))
        sample_size = min(n, len(df))
        df = df.sample(n=sample_size, random_state=random_state)

    return df

def clean(df: pd.DataFrame):
    """Null, 0 koordinat, çok kısa/uzun süreleri filtrele."""
    df = df.dropna()
    # Koordinat anomalileri
    coord_cols = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']
    for c in coord_cols:
        df = df[df[c] != 0]
    # Süre: 1 sn’den kısa ve üst yüzde 99’dan büyükleri çıkar
    df = df[(df['trip_duration'] >= 1)]
    upper = df['trip_duration'].quantile(0.99)
    df = df[df['trip_duration'] <= upper]
    return df.reset_index(drop=True)

def main():
    # 1) Örnek altküme oluştur
    sample = load_and_sample("data/raw/NYC.csv", n=100_000)

    # 2) Temizle
    cleaned = clean(sample)

    # 3) Kaydet
    cleaned.to_parquet("data/processed/train_sample.parquet", index=False)
    print(f"Kaydedildi: {cleaned.shape[0]} satır × {cleaned.shape[1]} sütun")

if __name__ == "__main__":
    main()

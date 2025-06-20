import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMRegressor

# Proje kökünü path’e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.preprocessing import load_and_sample, clean
from src.features import add_all_features

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))

if __name__ == "__main__":
    # 1) Veri yükle ve ön işleme (1M örnek)
    csv_path = os.path.join(ROOT, "data", "raw", "NYC.csv")
    df = load_and_sample(csv_path, n=1_000_000)
    df = clean(df)
    df = add_all_features(df)

    # 2) Özellik ve hedef
    feature_cols = [
        "distance_km","bearing","dist_to_center_km",
        "is_weekend","hour_sin","hour_cos","dow_sin","dow_cos",
        "pickup_cluster","dropoff_cluster","passenger_count"
    ]
    X = df[feature_cols].values
    y = np.log1p(df["trip_duration"].values)

    # 3) Modeli eğit
    model = LGBMRegressor(
        n_estimators=456,
        learning_rate=0.015985218032219677,
        num_leaves=150,
        max_depth=9,
        min_child_samples=37,
        subsample=0.8656,
        colsample_bytree=0.9526,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X, y)

    # 4) Performans kontrolü
    preds = model.predict(X)
    score = rmsle(np.expm1(y), np.expm1(preds))
    print(f"Train RMSLE (log-model): {score:.4f}")

    # 5) Modeli kaydet
    out_path = os.path.join(ROOT, "model_lgb_log.pkl")
    joblib.dump(model, out_path)
    print(f"Log-model kaydedildi → {out_path}")

# src/modeling.py

import os
import sys
import numpy as np
import pandas as pd
import logging
import warnings
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
from statistics import mean

# Uyarıları filtrele (sklearn validation ve lightgbm)
warnings.filterwarnings("ignore", message="X does not have valid feature names*")
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# Proje kökünü path'e ekle
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from src.preprocessing import load_and_sample, clean
from src.features import add_all_features


def rmsle(y_true, y_pred):
    """RMSLE hesaplar."""
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))


def get_data(sample_size=100000, cluster_k=10):
    """
    Veriyi yükle, temizle ve tüm özellikleri ekle.
    """
    csv_path = os.path.join(ROOT, "data", "raw", "NYC.csv")
    df = load_and_sample(csv_path, n=sample_size)
    df = clean(df)
    df = add_all_features(df, cluster_k=cluster_k)
    return df


def prepare_xy(df):
    """
    Model girişi X ve hedef y'yi hazırla.
    """
    feature_cols = [
        "distance_km", "bearing", "speed_kmh", "dist_to_center_km", "is_weekend",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "pickup_cluster", "dropoff_cluster", "passenger_count"
    ]
    X = df[feature_cols].values
    y = df["trip_duration"].values
    return X, y


def run_cv(model, X, y, n_splits=5):
    """
    K-Fold CV ile RMSLE skorlarını döner.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        scores.append(rmsle(y_te, preds))
    return scores


def run_models():
    # 1) Veri Hazırla
    df = get_data()
    X, y = prepare_xy(df)

    # 2) Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3) Baseline: median tahmin
    median = np.median(y_train)
    baseline_pred = np.full_like(y_test, median)
    print("Baseline RMSLE:", rmsle(y_test, baseline_pred))

    # 4) Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    print("Linear Regression RMSLE:", rmsle(y_test, lr.predict(X_test)))

    # 5) Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print("Random Forest RMSLE:", rmsle(y_test, rf.predict(X_test)))

    # 6) XGBoost
    xgb = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    print("XGBoost RMSLE:", rmsle(y_test, xgb.predict(X_test)))

    # 7) LightGBM (Tuned)
    lgb = LGBMRegressor(
        n_estimators=456,
        learning_rate=0.015985218032219677,
        num_leaves=150,
        max_depth=9,
        min_child_samples=37,
        subsample=0.8656155651827101,
        colsample_bytree=0.9525702386505803,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    lgb.fit(X_train, y_train)
    print("LightGBM (Tuned) RMSLE:", rmsle(y_test, lgb.predict(X_test)))

    # 8) CatBoost
    cb = CatBoostRegressor(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        random_state=42,
        verbose=0
    )
    cb.fit(X_train, y_train)
    print("CatBoost RMSLE:", rmsle(y_test, cb.predict(X_test)))

    # 9) En iyi modeli kaydet
    results = {
        "random_forest": (rf, rmsle(y_test, rf.predict(X_test))),
        "xgboost":       (xgb, rmsle(y_test, xgb.predict(X_test))),
        "lightgbm":      (lgb, rmsle(y_test, lgb.predict(X_test))),
        "catboost":      (cb, rmsle(y_test, cb.predict(X_test))),
    }
    best_name, (best_model, best_score) = min(results.items(), key=lambda x: x[1][1])
    best_path = os.path.join(ROOT, f"model_best_{best_name}.pkl")
    joblib.dump(best_model, best_path)
    print(f"Best Model: {best_name} (RMSLE={best_score:.4f}), saved to {best_path}")

    # 10) 5-Fold CV Sonuçları (Tuned LightGBM)
    print("\n--- 5-Fold CV Results (Tuned LightGBM) ---")
    cv_scores = run_cv(lgb, X, y, n_splits=5)
    print("Fold RMSLEs:", [round(s, 4) for s in cv_scores])
    print("Mean RMSLE:", round(mean(cv_scores), 4))


if __name__ == "__main__":
    run_models()

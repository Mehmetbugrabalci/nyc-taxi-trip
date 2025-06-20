import os
import numpy as np
import pandas as pd
import joblib
from math import radians, sin, cos, sqrt, asin

# --- Temel özellik fonksiyonları ---
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def add_distance(df):
    df = df.copy()
    df["distance_km"] = haversine_distance(
        df.pickup_latitude, df.pickup_longitude,
        df.dropoff_latitude, df.dropoff_longitude
    )
    return df

def add_bearing(df):
    df = df.copy()
    lat1, lon1 = map(radians, df.pickup_latitude), map(radians, df.pickup_longitude)
    lat2, lon2 = map(radians, df.dropoff_latitude), map(radians, df.dropoff_longitude)
    bearings = []
    for φ1, λ1, φ2, λ2 in zip(lat1, lon1, lat2, lon2):
        y = sin(λ2-λ1)*cos(φ2)
        x = cos(φ1)*sin(φ2) - sin(φ1)*cos(φ2)*cos(λ2-λ1)
        θ = np.degrees(np.arctan2(y, x))
        bearings.append((θ+360) % 360)
    df["bearing"] = bearings
    return df

def add_distance_to_center(df, center_lat=40.7589, center_lon=-73.9851):
    df = df.copy()
    df["dist_to_center_km"] = haversine_distance(
        df.pickup_latitude, df.pickup_longitude,
        center_lat, center_lon
    )
    return df

def add_is_weekend(df):
    df = df.copy()
    df["is_weekend"] = df.pickup_datetime.dt.dayofweek.isin([5,6]).astype(int)
    return df

def add_time_features(df):
    df = df.copy()
    df["pickup_hour"] = df.pickup_datetime.dt.hour
    df["hour_sin"] = np.sin(2*np.pi*df.pickup_hour/24)
    df["hour_cos"] = np.cos(2*np.pi*df.pickup_hour/24)
    df["pickup_dow"] = df.pickup_datetime.dt.dayofweek
    df["dow_sin"] = np.sin(2*np.pi*df.pickup_dow/7)
    df["dow_cos"] = np.cos(2*np.pi*df.pickup_dow/7)
    return df

def add_location_clusters(df):
    df = df.copy()
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    km_p = joblib.load(os.path.join(ROOT, "models", "kmeans_pickup.pkl"))
    km_d = joblib.load(os.path.join(ROOT, "models", "kmeans_dropoff.pkl"))
    df["pickup_cluster"] = km_p.predict(df[["pickup_latitude","pickup_longitude"]].values)
    df["dropoff_cluster"] = km_d.predict(df[["dropoff_latitude","dropoff_longitude"]].values)
    return df

def add_all_features(df):
    df = add_distance(df)
    df = add_bearing(df)
    df = add_distance_to_center(df)
    df = add_is_weekend(df)
    df = add_time_features(df)
    df = add_location_clusters(df)
    return df

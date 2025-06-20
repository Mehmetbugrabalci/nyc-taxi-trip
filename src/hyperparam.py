import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
import optuna
from lightgbm import LGBMRegressor

# proje kökünü path’e ekleyelim
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(dir_path)
from src.preprocessing import load_and_sample, clean
from src.features import add_all_features


def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, np.maximum(0, y_pred)))


def get_xy(sample_size=100000, cluster_k=10):
    df = load_and_sample(
        os.path.join(dir_path, 'data', 'raw', 'NYC.csv'),
        n=sample_size
    )
    df = clean(df)
    df = add_all_features(df, cluster_k=cluster_k)
    feature_cols = [
        'distance_km', 'bearing', 'speed_kmh',
        'dist_to_center_km', 'is_weekend',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        'pickup_cluster', 'dropoff_cluster', 'passenger_count'
    ]
    X = df[feature_cols].values
    y = df['trip_duration'].values
    return train_test_split(X, y, test_size=0.2, random_state=42)


def objective(trial):
    X_train, X_test, y_train, y_test = get_xy()
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'n_jobs': -1
    }
    model = LGBMRegressor(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return rmsle(y_test, preds)


def main():
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print('En iyi parametreler:', study.best_params)
    print('En iyi RMSLE   :', study.best_value)

if __name__ == '__main__':
    main()

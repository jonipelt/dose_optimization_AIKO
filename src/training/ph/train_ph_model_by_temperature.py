# train_ph_temp.py

import os
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configuration (EDIT THESE PATHS FOR YOUR OWN DATA)

# Input data path (preprocessed training data)
DATA_PATH = "data/cleaned_training_data.csv"  # Change this to your actual file

# Output directory for trained models
MODEL_OUTPUT_DIR = "models/temp_based"  # Change if needed

# Output path for saving model evaluation metrics
METRICS_OUTPUT_PATH = "models/metrics/metrics_ph_temp.csv"  # Change if needed

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_OUTPUT_PATH), exist_ok=True)

# Helper Functions

def calculate_basin_count(flow_rate: float) -> int:
    return 2 if flow_rate < 1300 else 3

def classify_temperature(temp: float) -> str:
    if temp < 10:
        return "cold"
    elif 10 <= temp < 18:
        return "moderate"
    elif temp >= 18:
        return "warm"
    return "other"

def segment_analysis(df, y_true, y_pred, segment_column):
    print(f"\nSegment-wise error analysis: {segment_column}")
    df_copy = df.copy()
    df_copy["y_true"] = y_true
    df_copy["y_pred"] = y_pred

    for group in df_copy[segment_column].unique():
        subset = df_copy[df_copy[segment_column] == group]
        if len(subset) < 20:
            continue
        rmse = mean_squared_error(subset["y_true"], subset["y_pred"]) ** 0.5
        mae = mean_absolute_error(subset["y_true"], subset["y_pred"])
        r2 = r2_score(subset["y_true"], subset["y_pred"])
        print(f"{group:>8} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

# Data Loading & Preprocessing

df = pd.read_csv(DATA_PATH, sep=";", decimal=",", on_bad_lines='skip')

required_cols = ["raaka_sameus", "tuleva_virtaus", "tuleva_lampotila", "alku_pH", "kemikaaliannos", "flotaatio_sameus"]
df = df.dropna(subset=required_cols)

df["basin_count"] = df["tuleva_virtaus"].apply(calculate_basin_count)
df["temp_class"] = df["tuleva_lampotila"].apply(classify_temperature)

# Keep only rows where flotation turbidity is within target range
df = df[(df["flotaatio_sameus"] >= 0.5) & (df["flotaatio_sameus"] <= 1.0)].copy()

# Remove spurious dose outliers
dose_array = df["kemikaaliannos"].values
mask = np.ones(len(df), dtype=bool)

for i in range(1, len(df) - 1):
    prev, curr, next_ = dose_array[i - 1], dose_array[i], dose_array[i + 1]
    if curr < 0.5 * prev and curr < 0.5 * next_:
        mask[i] = False

df = df[mask].reset_index(drop=True)
print(f"Removed {np.sum(~mask)} rows with anomalously low chemical dose.")

# Model Training

metrics = []

def train_model_for_temp_class(temp_class, df_sub):
    print(f"\nTraining initial pH model for temperature class: {temp_class} ({len(df_sub)} rows)")

    features = ["raaka_sameus", "tuleva_virtaus", "tuleva_lampotila", "basin_count"]
    target = "alku_pH"

    X = df_sub[features]
    y = df_sub[target]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 50),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "random_state": 42,
            "verbosity": -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds) ** 0.5

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("\nValidation Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

    segment_analysis(df_sub.iloc[X_val.index], y_val, y_pred, "temp_class")

    model_filename = f"ph_model_temp_{temp_class}.pkl"
    joblib.dump(model, os.path.join(MODEL_OUTPUT_DIR, model_filename))
    print(f"Saved model to: {model_filename}")

    metrics.append({
        "model": model_filename,
        "class": temp_class,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "rows": len(df_sub)
    })

# Train models for all temperature classes

for temp_class in df["temp_class"].unique():
    subset = df[df["temp_class"] == temp_class]
    if len(subset) < 100:
        print(f"Skipping {temp_class}: insufficient data")
        continue
    train_model_for_temp_class(temp_class, subset)

# Save evaluation metrics
pd.DataFrame(metrics).sort_values("class").to_csv(METRICS_OUTPUT_PATH, sep=";", decimal=",", index=False)
print(f"\nSaved evaluation metrics to: {METRICS_OUTPUT_PATH}")

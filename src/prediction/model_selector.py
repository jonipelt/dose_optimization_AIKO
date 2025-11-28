import os
import joblib
import pandas as pd
import numpy as np

# Configuration

MODEL_DIR_FLOW = "models/flow_based"
MODEL_DIR_TEMP = "models/temperature_based"
MODEL_FINAL_PH = "models/flot_ph/final_pH_model.pkl"

# Internal Model Cache
_cached_models = {}

# Classification Functions

def classify_flow(flow: float) -> str:
    if 700 <= flow < 950:
        return "low"
    elif 950 <= flow < 1100:
        return "medium"
    elif 1100 <= flow <= 2200:
        return "high"
    return "other"

def classify_temp(temp: float) -> str:
    if pd.isna(temp):
        return None
    if temp < 10:
        return "cold"
    elif 10 <= temp < 18:
        return "moderate"
    else:
        return "warm"
    
def calculate_basin_count(flow: float) -> int:
    return 2 if flow < 1300 else 3

# Internal Utilities

def _load_model(path: str):
    if path in _cached_models:
        return _cached_models[path]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    _cached_models[path] = model
    return model

def _prepare_input(row: pd.Series, features: list[str]) -> pd.DataFrame:
    data = {f: row.get(f, np.nan) for f in features}
    if "basin_count" in features:
        data["basin_count"] = calculate_basin_count(row.get("tuleva_virtaus", np.nan))
    return pd.DataFrame([data])

def _predict(row: pd.Series, model_path: str) -> float:
    model = _load_model(model_path)
    features = getattr(model, "feature_name_", [])
    X = _prepare_input(row, features)
    return float(np.asarray(model.predict(X)).ravel()[0])

# Model Path Helpers
def _path_flow(model_type: str, flow_class: str) -> str:
    return os.path.join(MODEL_DIR_FLOW, f"{model_type}_model_{flow_class}.pkl")

def _path_temp(model_type: str, temp_class: str) -> str:
    return os.path.join(MODEL_DIR_TEMP, f"{model_type}_model_{temp_class}.pkl")

# Prediction Interfaces

def predict_dose_by_flow(row: pd.Series) -> float:
    flow_class = classify_flow(row.get("tuleva_virtaus"))
    return _predict(row, _path_flow("dose", flow_class))

def predict_dose_by_temp(row: pd.Series) -> float:
    temp_class = classify_temp(row.get("tuleva_lampotila"))
    return _predict(row, _path_temp("dose", temp_class))

def predict_dose_avg(row: pd.Series) -> float:
    return np.nanmean([predict_dose_by_flow(row), predict_dose_by_temp(row)])

def predict_ph_by_flow(row: pd.Series) -> float:
    flow_class = classify_flow(row.get("tuleva_virtaus"))
    return _predict(row, _path_flow("ph", flow_class))

def predict_ph_by_temp(row: pd.Series) -> float:
    temp_class = classify_temp(row.get("tuleva_lampotila"))
    return _predict(row, _path_temp("ph", temp_class))

def predict_ph_avg(row: pd.Series) -> float:
    return np.nanmean([predict_ph_by_flow(row), predict_ph_by_temp(row)])

def predict_quality_by_flow(row: pd.Series) -> float:
    flow_class = classify_flow(row.get("tuleva_virtaus"))
    return _predict(row, _path_flow("quality", flow_class))

def predict_quality_by_temp(row: pd.Series) -> float:
    temp_class = classify_temp(row.get("tuleva_lampotila"))
    return _predict(row, _path_temp("quality", temp_class))

def predict_quality_avg(row: pd.Series) -> float:
    return np.nanmean([predict_quality_by_flow(row), predict_quality_by_temp(row)])

def predict_final_pH(row: pd.Series) -> float:
    return _predict(row, MODEL_FINAL_PH)
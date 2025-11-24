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

# Output directory for trained model
MODEL_OUTPUT_DIR = "models/flot_ph/"  # Change if needed

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Helper functions

def calculate_basin_count(flow_rate: float) -> int:
    return 2 if flow_rate < 1300 else 3

def get_flow_class(flow):
    if 700 <= flow < 950:
        return "low"
    elif 950 <= flow < 1400:
        return "medium"
    elif 1400 <= flow <= 2200:
        return "high"
    else:
        return "other"

def get_temp_class(temp):
    if temp < 10:
        return "cold"
    elif 10 <= temp < 18:
        return "moderate"
    else:
        return "warm"

def segment_analysis(df, y_true, y_pred, segment_column):
    print(f"\nSegment-wise error analysis: {segment_column}")
    df_copy = df.copy()
    df_copy["y_true"] = y_true
    df_copy["y_pred"] = y_pred

    for cls in df_copy[segment_column].unique():
        subset = df_copy[df_copy[segment_column] == cls]
        if len(subset) < 20:
            continue
        rmse = mean_squared_error(subset["y_true"], subset["y_pred"]) ** 0.5
        mae = mean_absolute_error(subset["y_true"], subset["y_pred"])
        r2 = r2_score(subset["y_true"], subset["y_pred"])
        print(f"{cls:>8} | RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f}")

# Data loading and filtering
df = pd.read_csv(DATA_PATH, sep=";", decimal=",", on_bad_lines='skip')

required_cols = ["raaka_sameus", "tuleva_virtaus", "tuleva_lampotila", "alku_pH", "kemikaaliannos", "flotaatio_pH"]
df = df.dropna(subset=required_cols)

# Filter rows where flotation turbidity is within desired range (optional quality filter)
df = df[(df["flotaatio_sameus"] >= 0.5) & (df["flotaatio_sameus"] <= 1.0)].copy()

# Add class columns for later segmentation
df["flow_class"] = df["tuleva_virtaus"].apply(get_flow_class)
df["temp_class"] = df["tuleva_lampotila"].apply(get_temp_class)

# Features and target
features = ["raaka_sameus", "tuleva_virtaus", "tuleva_lampotila", "alku_pH", "kemikaaliannos"]
target = "flotaatio_pH"

X = df[features]
y = df[target]

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optuna hyperparameter search
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

print("\nGlobal final pH model performance")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  R²:   {r2:.4f}")

segment_analysis(df.iloc[y_val.index], y_val, y_pred, "flow_class")
segment_analysis(df.iloc[y_val.index], y_val, y_pred, "temp_class")

# Save model
model_filename = os.path.join(MODEL_OUTPUT_DIR, "final_pH_model.pkl")
joblib.dump(model, model_filename)
print(f"\nModel saved to: {model_filename}")

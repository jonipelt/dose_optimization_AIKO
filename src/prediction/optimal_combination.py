import os
import pandas as pd
import numpy as np
from model_selector import (
    predict_pH_avg,
    predict_dose_avg,
    predict_sameus_avg,
    predict_final_pH
)

# CONFIGURATION
RAW_TURBIDITY = 2.5               # Change this to simulate different raw water conditions
FLOW_RANGE = np.arange(700, 2101, 100)
OUTPUT_CSV_PATH = "figures/optimal_table_2.5_FTU.csv"  # <- Change for your use case
os.makedirs(os.path.dirname(OUTPUT_CSV_PATH), exist_ok=True)

# Limits and targets
SAMEUS_TARGET = 0.9
MIN_PH = 4.9
MAX_PH = 5.8
MIN_DOSE = 0.0
MAX_DOSE = 120.0
MIN_CONCENTRATION = 48.0  # g/m³

# Temperature class definitions
TEMP_CLASSES = {
    "cold": list(range(0, 10)),
    "moderate": list(range(10, 18)),
    "warm": list(range(18, 30))
}

# FUNCTIONS

def calculate_concentration(dose_l: float, flow_m3h: float) -> float:
    """Calculate chemical concentration in g/m³."""
    return (dose_l * 1.54 * 1000) / flow_m3h

def optimize_combination_table():
    results = []

    print("Generating optimal pH + dose combinations...")

    for flow in FLOW_RANGE:
        for temp_class, temp_values in TEMP_CLASSES.items():
            pH_values, dose_values, sameus_values = [], [], []
            final_pH_values, conc_values = [], []

            for temp in temp_values:
                base_input = {
                    "raaka_sameus": RAW_TURBIDITY,
                    "tuleva_virtaus": flow,
                    "tuleva_lampotila": temp
                }

                # Predict initial pH and clip within limits
                init_pH = predict_pH_avg(pd.Series(base_input))
                init_pH = np.clip(init_pH, MIN_PH, MAX_PH)

                # Predict dose and ensure within bounds
                dose_input = {**base_input, "alku_pH": init_pH}
                dose = predict_dose_avg(pd.Series(dose_input))
                dose = np.clip(dose, MIN_DOSE, MAX_DOSE)

                # Check concentration
                conc = calculate_concentration(dose, flow)
                if conc < MIN_CONCENTRATION:
                    conc = MIN_CONCENTRATION
                    dose = conc * flow / (1.54 * 1000)

                # Predict quality outcomes
                quality_input = pd.Series({**dose_input, "kemikaaliannos": dose})
                pred_sameus = predict_sameus_avg(quality_input)
                pred_final_pH = predict_final_pH(quality_input)

                # Store values
                pH_values.append(init_pH)
                dose_values.append(dose)
                sameus_values.append(pred_sameus)
                final_pH_values.append(pred_final_pH)
                conc_values.append(conc)

            results.append({
                "flow": flow,
                "temp_class": temp_class,
                "raw_turbidity": RAW_TURBIDITY,
                "opt_pH": round(np.nanmean(pH_values), 2),
                "opt_dose": round(np.nanmean(dose_values), 2),
                "pred_turbidity": round(np.nanmean(sameus_values), 3),
                "pred_final_pH": round(np.nanmean(final_pH_values), 2),
                "concentration": round(np.nanmean(conc_values), 1)
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_result = optimize_combination_table()
    df_result.to_csv(OUTPUT_CSV_PATH, sep=";", decimal=",", index=False)
    print(f"Saved table to: {OUTPUT_CSV_PATH}")

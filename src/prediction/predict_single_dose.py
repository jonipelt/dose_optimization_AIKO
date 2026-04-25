import pandas as pd
import numpy as np
from model_selector import (
    predict_ph_avg,
    predict_dose_avg,
    predict_quality_avg,
    predict_final_pH
)

# Configuration (EDIT IF NEEDED)
SAMEUS_TARGET = 0.9
SAMEUS_LOWER = 0.5
SAMEUS_UPPER = 1.0
MIN_DOSE = 0.0
MAX_DOSE = 120.0
MIN_PH = 4.9
MAX_PH = 5.8
MIN_CONCENTRATION = 48.0  # g/m³

def evaluate_quality(sameus):
    if SAMEUS_LOWER <= sameus <= SAMEUS_UPPER:
        return "Turbidity within target"
    elif sameus < SAMEUS_LOWER:
        return "Turbidity too low"
    elif sameus > SAMEUS_UPPER:
        return "Turbidity too high"
    return "Invalid turbidity"

def calculate_concentration(dose_liters, flow_rate):
    return (dose_liters * 1.54 * 1000) / flow_rate

def predict_single_case(raaka_sameus, tuleva_virtaus, tuleva_lampotila):
    try:
        # Initialize input
        base = pd.Series({
            "raaka_sameus": raaka_sameus,
            "tuleva_virtaus": tuleva_virtaus,
            "tuleva_lampotila": tuleva_lampotila,
        })

        # Predict initial pH
        predicted_pH = predict_ph_avg(base)
        predicted_pH = np.clip(predicted_pH, MIN_PH, MAX_PH)

        # Predict chemical dose
        dose_input = base.copy()
        dose_input["alku_pH"] = predicted_pH
        predicted_dose = predict_dose_avg(dose_input)
        predicted_dose = np.clip(predicted_dose, MIN_DOSE, MAX_DOSE)

        # Check concentration limit
        concentration = calculate_concentration(predicted_dose, tuleva_virtaus)
        if concentration < MIN_CONCENTRATION:
            concentration = MIN_CONCENTRATION
            predicted_dose = concentration * tuleva_virtaus / (1.54 * 1000)

        # Predict turbidity
        quality_input = dose_input.copy()
        quality_input["kemikaaliannos"] = predicted_dose
        predicted_sameus = predict_quality_avg(quality_input)

        fallback_used = False
        rounded_sameus = round(predicted_sameus, 1)

        # Fallback optimization if target not met
        if not SAMEUS_LOWER <= rounded_sameus <= SAMEUS_UPPER or np.isnan(predicted_sameus):
            fallback_used = True
            best_candidate = None
            best_error = np.inf

            for dose in np.arange(MIN_DOSE, MAX_DOSE + 1.0, 1.0):
                conc = calculate_concentration(dose, tuleva_virtaus)
                if conc < MIN_CONCENTRATION:
                    continue

                for pH in np.arange(MIN_PH, MAX_PH + 0.1, 0.1):
                    row = base.copy()
                    row["alku_pH"] = round(pH, 2)
                    row["kemikaaliannos"] = round(dose, 2)
                    turbidity = predict_quality_avg(row)
                    if np.isnan(turbidity):
                        continue
                    turbidity_rounded = round(turbidity, 1)

                    if SAMEUS_LOWER <= turbidity_rounded <= SAMEUS_UPPER:
                        error = abs(turbidity_rounded - SAMEUS_TARGET)
                        if error < best_error:
                            best_error = error
                            best_candidate = {
                                "alku_pH": round(pH, 2),
                                "kemikaaliannos": round(dose, 2),
                                "ennustettu_sameus": turbidity,
                                "pitoisuus": conc
                            }

            if best_candidate:
                predicted_pH = best_candidate["alku_pH"]
                predicted_dose = best_candidate["kemikaaliannos"]
                predicted_sameus = best_candidate["ennustettu_sameus"]
                concentration = best_candidate["pitoisuus"]
                rounded_sameus = round(predicted_sameus, 1)
            else:
                raise ValueError("Fallback optimization failed.")

        # Final pH prediction
        final_pH_input = pd.Series({
            "raaka_sameus": raaka_sameus,
            "tuleva_virtaus": tuleva_virtaus,
            "tuleva_lampotila": tuleva_lampotila,
            "alku_pH": predicted_pH,
            "kemikaaliannos": predicted_dose
        })
        predicted_final_pH = predict_final_pH(final_pH_input)

        # Output
        print("\nOptimized Prediction")
        print(f"Initial pH:         {predicted_pH:.2f}")
        print(f"Chemical dose:      {predicted_dose:.2f} L")
        print(f"Turbidity:          {predicted_sameus:.2f} NTU")
        print(f"Final pH:           {predicted_final_pH:.2f}")
        print(f"Concentration:      {concentration:.2f} g/m³")
        print(f"Quality evaluation: {evaluate_quality(predicted_sameus)}")
        if fallback_used:
            print("Note: Fallback optimization was applied.")

    except Exception as e:
        print(f"Error: {e}")

# CLI mode for manual input
def cli_mode():
    try:
        raw_turbidity = float(input("Raw water turbidity (NTU): "))
        incoming_flow = float(input("Incoming flow (m³/h): "))
        temperature = float(input("Incoming temperature (°C): "))
        predict_single_case(raw_turbidity, incoming_flow, temperature)
    except Exception as e:
        print(f"Invalid input: {e}")

if __name__ == "__main__":
    cli_mode()

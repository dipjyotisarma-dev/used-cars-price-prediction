"""
src/pipeline.py
Production inference pipeline and business logic for the Used Car Fair Value Engine.
Decoupled from any UI framework.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

# Define default model path relative to this file's directory
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "rf_tuned_pipeline.joblib"
)

# Business Logic: Segmented Error Brackets (derived from Phase 10 Error Analysis)
BRACKET_CONFIG = [
    {
        "name": "Budget (< ₹5L)",
        "max_price": 500_000,
        "margin": 0.37,
        "warning": (
            "⚠️ Budget Tier Alert: Vehicles under ₹5 Lakh exhibit high price variance (±37% MAPE). "
            "Actual mechanical condition, local market demand, and service history significantly influence final price."
        ),
    },
    {
        "name": "Mid-Range (₹5L - ₹15L)",
        "max_price": 1_500_000,
        "margin": 0.23,
        "warning": None,
    },
    {
        "name": "Premium (₹15L - ₹30L)",
        "max_price": 3_000_000,
        "margin": 0.25,
        "warning": None,
    },
    {
        "name": "Luxury (> ₹30L)",
        "max_price": float("inf"),
        "margin": 0.34,
        "warning": (
            "⚠️ Luxury Tier Alert: High-end luxury vehicles display elevated dispersion (±34% MAPE) "
            "due to steep, non-linear depreciation curves and high variation in premium package configurations."
        ),
    },
]


def load_trained_pipeline(model_path: Optional[Path] = None) -> Any:
    """
    Safely loads the serialized scikit-learn pipeline artifact.
    """
    target_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at: {target_path}. "
            "Ensure Phase 6 tuning and serialization completed successfully."
        )

    try:
        pipeline = joblib.load(target_path)
        return pipeline
    except Exception as exc:
        raise RuntimeError(
            f"Failed to deserialize model artifact at {target_path}: {exc}"
        ) from exc


def get_bracket_and_margin(price_inr: float) -> Tuple[str, float, Optional[str]]:
    """
    Determines the price tier bracket, error margin (MAPE), and warning message
    based on the predicted INR value.
    """
    for config in BRACKET_CONFIG:
        if price_inr <= config["max_price"]:
            return config["name"], config["margin"], config["warning"]

    # Fallback to luxury if somehow above all thresholds
    luxury = BRACKET_CONFIG[-1]
    return luxury["name"], luxury["margin"], luxury["warning"]


def predict_fair_price(
    input_data: pd.DataFrame, pipeline: Any
) -> Dict[str, Any]:
    """
    Executes model inference and applies post-processing business logic:
    1. Validates input schema.
    2. Runs pipeline prediction in log-space.
    3. Inverts log transformation using np.expm1().
    4. Calculates dynamic confidence intervals based on price tier MAPE.
    5. Formulates tier-specific business warnings.

    Returns:
        Dict containing predicted price, lower/upper bounds, bracket, margin, and warning.
    """
    expected_cols = ["Brand", "Age", "kmDriven", "Transmission", "Owner", "FuelType"]
    missing_cols = [col for col in expected_cols if col not in input_data.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required features: {missing_cols}")

    # Ensure correct column ordering
    ordered_input = input_data[expected_cols].copy()

    # 1. Inference in log-space
    pred_log = pipeline.predict(ordered_input)[0]

    # 2. Invert log transformation
    pred_inr = float(np.expm1(pred_log))

    # Guardrail: Price cannot be negative
    pred_inr = max(0.0, pred_inr)

    # 3. Dynamic uncertainty intervals
    bracket_name, margin, warning = get_bracket_and_margin(pred_inr)
    lower_bound = max(0.0, pred_inr * (1.0 - margin))
    upper_bound = pred_inr * (1.0 + margin)

    return {
        "predicted_price": pred_inr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "bracket": bracket_name,
        "margin_percentage": margin * 100,
        "warning": warning,
    }


if __name__ == "__main__":
    print("--- Testing Production Pipeline Standalone ---")
    try:
        model = load_trained_pipeline()
        print("✓ Pipeline successfully loaded.")

        # Test case 1: Typical budget / mid-range car
        sample_car = pd.DataFrame(
            [
                {
                    "Brand": "Maruti Suzuki",
                    "Age": 5,
                    "kmDriven": 45000.0,
                    "Transmission": "Manual",
                    "Owner": "first",
                    "FuelType": "petrol",
                }
            ]
        )

        result = predict_fair_price(sample_car, model)
        print("\nTest Prediction Result:")
        print(f"  Predicted Price: ₹{result['predicted_price']:,.2f}")
        print(f"  Confidence Band: ₹{result['lower_bound']:,.2f} - ₹{result['upper_bound']:,.2f}")
        print(f"  Tier Bracket:    {result['bracket']} (±{result['margin_percentage']:.1f}%)")
        print(f"  Warning:         {result['warning']}")

        # Test case 2: Luxury car test (to verify luxury warning)
        sample_luxury = pd.DataFrame(
            [
                {
                    "Brand": "Porsche",
                    "Age": 2,
                    "kmDriven": 12000.0,
                    "Transmission": "Automatic",
                    "Owner": "first",
                    "FuelType": "petrol",
                }
            ]
        )
        lux_result = predict_fair_price(sample_luxury, model)
        print("\nLuxury Prediction Result:")
        print(f"  Predicted Price: ₹{lux_result['predicted_price']:,.2f}")
        print(f"  Confidence Band: ₹{lux_result['lower_bound']:,.2f} - ₹{lux_result['upper_bound']:,.2f}")
        print(f"  Tier Bracket:    {lux_result['bracket']} (±{lux_result['margin_percentage']:.1f}%)")
        print(f"  Warning:         {lux_result['warning']}")

        print("\n✓ Standalone verification complete.")
    except Exception as err:
        print(f"❌ Error during standalone test: {err}")
"""
tests/test_pipeline.py
Unit tests for the Used Car Fair Value Engine inference pipeline (src/pipeline.py).
"""

from pathlib import Path
import pytest
import pandas as pd
from src.pipeline import (
    load_trained_pipeline,
    get_bracket_and_margin,
    predict_fair_price,
)


@pytest.fixture(scope="module")
def pipeline_model():
    """Fixture that loads the production pipeline once for the entire module."""
    return load_trained_pipeline()


@pytest.fixture
def sample_car_input():
    """Fixture providing a standard, valid car input DataFrame."""
    return pd.DataFrame(
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


# ── 1. Model Loading Tests ──────────────────────────────────────────


def test_load_pipeline_success(pipeline_model):
    """Ensure the pipeline artifact loads successfully and has a predict method."""
    assert pipeline_model is not None
    assert hasattr(pipeline_model, "predict"), "Loaded model must have a 'predict' method."


def test_load_pipeline_missing_file():
    """Ensure load_trained_pipeline raises FileNotFoundError if path does not exist."""
    invalid_path = Path("models/non_existent_model.joblib")
    with pytest.raises(FileNotFoundError):
        load_trained_pipeline(invalid_path)


# ── 2. Business Logic / Bracket Tests ───────────────────────────────


@pytest.mark.parametrize(
    "price, expected_bracket, expected_margin",
    [
        (300_000, "Budget (< ₹5L)", 0.37),
        (500_000, "Budget (< ₹5L)", 0.37),
        (1_000_000, "Mid-Range (₹5L - ₹15L)", 0.23),
        (1_500_000, "Mid-Range (₹5L - ₹15L)", 0.23),
        (2_000_000, "Premium (₹15L - ₹30L)", 0.25),
        (3_000_000, "Premium (₹15L - ₹30L)", 0.25),
        (4_500_000, "Luxury (> ₹30L)", 0.34),
    ],
)
def test_bracket_and_margin_boundaries(price, expected_bracket, expected_margin):
    """Verify that prices are mapped to correct error brackets and MAPE margins."""
    bracket, margin, _ = get_bracket_and_margin(price)
    assert bracket == expected_bracket
    assert margin == expected_margin


# ── 3. Inference Contract & Guardrails ──────────────────────────────


def test_predict_fair_price_happy_path(pipeline_model, sample_car_input):
    """Verify that predict_fair_price returns the expected dictionary structure and invariants."""
    result = predict_fair_price(sample_car_input, pipeline_model)

    # Contract checks: all required keys must exist
    expected_keys = {
        "predicted_price",
        "lower_bound",
        "upper_bound",
        "bracket",
        "margin_percentage",
        "warning",
    }
    assert expected_keys.issubset(result.keys()), f"Missing keys in result: {expected_keys - set(result.keys())}"

    # Invariant checks: physical sanity
    price = result["predicted_price"]
    lower = result["lower_bound"]
    upper = result["upper_bound"]

    assert isinstance(price, float)
    assert price > 0, "Predicted price must be strictly positive."
    assert lower >= 0, "Lower bound cannot be negative."
    assert lower <= price <= upper, "Confidence bounds invariant violated: lower <= price <= upper."


def test_predict_fair_price_missing_columns(pipeline_model, sample_car_input):
    """Verify that omitting a required feature raises a ValueError."""
    bad_input = sample_car_input.drop(columns=["kmDriven"])
    with pytest.raises(ValueError, match="missing required features"):
        predict_fair_price(bad_input, pipeline_model)


def test_predict_fair_price_scrambled_column_order(pipeline_model, sample_car_input):
    """Verify that passing columns in scrambled order produces the exact same prediction."""
    # Reverse column order
    scrambled_input = sample_car_input[["FuelType", "Owner", "Transmission", "kmDriven", "Age", "Brand"]]
    
    result_normal = predict_fair_price(sample_car_input, pipeline_model)
    result_scrambled = predict_fair_price(scrambled_input, pipeline_model)

    assert pytest.approx(result_normal["predicted_price"], rel=1e-5) == result_scrambled["predicted_price"]
"""
app/data_loader.py
Data access layer: caches the trained ML pipeline and provides
query helpers against the historical pre-owned cars dataset.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import streamlit as st

from src.pipeline import load_trained_pipeline

ROOT_DIR = Path(__file__).resolve().parent.parent
CLEANED_DATA_PATH = ROOT_DIR / "data" / "interim" / "used_cars_cleaned.csv"


# ── Cached Model Loader ──────────────────────────────────────────────
@st.cache_resource
def get_model() -> Any:
    """Load and cache the trained 53MB Random Forest pipeline."""
    return load_trained_pipeline()


# ── Cached Dataset Loader ────────────────────────────────────────────
@st.cache_data
def get_market_dataset() -> pd.DataFrame:
    """
    Loads and caches the cleaned dataset (13,652 listings)
    for real-time market comparisons and recruiter analytics.
    """
    if not CLEANED_DATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLEANED_DATA_PATH)


# ── Category Extractor ───────────────────────────────────────────────
def get_categories(pipeline: Any) -> Dict[str, List[str]]:
    """Dynamically pull exact fitted categories from the pipeline preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    nom = preprocessor.named_transformers_["nom"]
    ordi = preprocessor.named_transformers_["ord"]
    return {
        "brands": sorted(list(nom.categories_[0])),
        "fuels": list(nom.categories_[1]),
        "transmissions": list(nom.categories_[2]),
        "owners": list(ordi.categories_[0]),
    }


# ── INR Currency Formatter ───────────────────────────────────────────
def fmt_inr(value: float) -> str:
    """Format numbers into Indian Rupee notation (e.g., ₹6,50,000)."""
    return f"₹{value:,.0f}"


# ── Brand-Specific Market Comps ──────────────────────────────────────
def get_brand_market_stats(df: pd.DataFrame, brand: str) -> Optional[Dict[str, Any]]:
    """Compute aggregate pricing metrics for a specific brand."""
    if df.empty:
        return None
    b_df = df[df["Brand"] == brand]
    if b_df.empty:
        return None
    return {
        "total_listings": len(b_df),
        "median_price": float(b_df["AskPrice"].median()),
        "min_price": float(b_df["AskPrice"].min()),
        "max_price": float(b_df["AskPrice"].max()),
        "top_fuel": b_df["FuelType"].mode().iloc[0] if not b_df["FuelType"].empty else "N/A",
    }


def get_top_models_summary(df: pd.DataFrame, brand: str, top_n: int = 5) -> pd.DataFrame:
    """
    Returns median and average asking prices for the top N models of a brand.
    Helps users benchmark their specific model (e.g., Swift vs Brezza).
    """
    if df.empty:
        return pd.DataFrame()
    b_df = df[df["Brand"] == brand]
    if b_df.empty:
        return pd.DataFrame()

    top_models = b_df["model"].value_counts().head(top_n).index
    summary = (
        b_df[b_df["model"].isin(top_models)]
        .groupby("model")
        .agg(
            Listings=("AskPrice", "count"),
            Median_Price=("AskPrice", "median"),
            Avg_Price=("AskPrice", "mean"),
        )
        .reset_index()
        .sort_values(by="Listings", ascending=False)
    )
    return summary


def get_comparable_listings(
    df: pd.DataFrame, brand: str, reg_year: int, limit: int = 5
) -> pd.DataFrame:
    """
    Pulls actual car listings from the dataset for the selected brand
    within +/- 2 years of the target registration year.
    """
    if df.empty:
        return pd.DataFrame()
    b_df = df[df["Brand"] == brand]
    if b_df.empty:
        return pd.DataFrame()

    comps = b_df[(b_df["Year"] >= reg_year - 2) & (b_df["Year"] <= reg_year + 2)]
    if len(comps) < limit:
        comps = b_df.sort_values(by="Year", ascending=False)

    cols = ["model", "Year", "FuelType", "Transmission", "kmDriven", "AskPrice"]
    available_cols = [c for c in cols if c in comps.columns]
    sample = comps.head(limit)[available_cols].copy()
    return sample
"""
app/app.py
Used Car Fair Value Engine — Production Streamlit application.
User-facing valuation tool with a separate technical reference section.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

# Resolve project root for modular imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import load_trained_pipeline, predict_fair_price

# ── Page configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Fair Value Engine",
    page_icon=":material/directions_car:",
    layout="centered",
)

CURRENT_YEAR = datetime.now().year
GITHUB_URL = "https://github.com/dipjyotisarma-dev/used-cars-price-prediction"


# ── Cached model loader ─────────────────────────────────────────────
@st.cache_resource
def get_model():
    """Load the serialized pipeline once and keep it in memory across reruns."""
    return load_trained_pipeline()


# ── Helper: format INR with lakhs ────────────────────────────────────
def fmt_inr(value: float) -> str:
    """Format a number into Indian Rupee notation with commas."""
    return f"₹{value:,.0f}"


# ── Helper: extract fitted categories from pipeline ──────────────────
def get_categories(pipeline):
    """Dynamically pull the exact categories the model was trained on."""
    preprocessor = pipeline.named_steps["preprocessor"]
    nom = preprocessor.named_transformers_["nom"]
    ordi = preprocessor.named_transformers_["ord"]
    return {
        "brands": sorted(list(nom.categories_[0])),
        "fuels": list(nom.categories_[1]),
        "transmissions": list(nom.categories_[2]),
        "owners": list(ordi.categories_[0]),
    }


# ── Main application ────────────────────────────────────────────────
def main():

    # Load model
    try:
        model = get_model()
    except Exception as err:
        st.error(f"Could not load the valuation model: {err}", icon=":material/error:")
        st.stop()

    cats = get_categories(model)

    # Smart defaults
    default_brand = cats["brands"].index("Maruti Suzuki") if "Maruti Suzuki" in cats["brands"] else 0
    default_fuel = cats["fuels"].index("petrol") if "petrol" in cats["fuels"] else 0
    default_trans = cats["transmissions"].index("Manual") if "Manual" in cats["transmissions"] else 0

    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header(":material/directions_car: Fair Value Engine")
        st.caption("Estimate what a used car is really worth.")

        st.markdown(
            "This tool provides a **market-calibrated price range** "
            "for pre-owned vehicles sold in India, based on key attributes "
            "like brand, age, mileage, fuel type, and ownership history."
        )

        st.markdown("---")
        st.subheader(":material/info: How it works")
        st.markdown(
            """
            1. Enter your vehicle's details in the form.
            2. Get a **fair market value** along with a recommended
               **negotiation range** (trade-in to retail).
            3. Prices are adjusted for the vehicle's segment — budget
               cars and luxury cars naturally carry wider price ranges
               than mid-range vehicles.
            """
        )

        st.markdown("---")
        st.subheader(":material/tips_and_updates: Tips for accuracy")
        st.markdown(
            """
            - Choose the **exact brand** as listed on your RC book.
            - Enter the **registration year**, not the manufacturing year.
            - Use the **current odometer reading** in kilometers.
            - The estimate does not account for accident history,
              aftermarket modifications, or regional demand spikes.
            """
        )

        st.markdown("---")
        st.caption(
            f"[View source code on GitHub]({GITHUB_URL})"
        )

    # ── Main area — two tabs ─────────────────────────────────────────
    tab_valuation, tab_about = st.tabs(
        [
            ":material/calculate: Estimate value",
            ":material/school: About this project",
        ]
    )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 1 — VALUATION
    # ═══════════════════════════════════════════════════════════════════
    with tab_valuation:
        st.header("Get your car's fair value")
        st.caption(
            "Fill in the details below and press **Estimate** to see the recommended price range."
        )

        with st.form("valuation_form"):
            col_l, col_r = st.columns(2)

            with col_l:
                brand = st.selectbox(
                    "Brand",
                    options=cats["brands"],
                    index=default_brand,
                    help="Select the manufacturer as shown on the registration certificate.",
                )

                years = list(range(CURRENT_YEAR, CURRENT_YEAR - 21, -1))
                reg_year = st.selectbox(
                    "Registration year",
                    options=years,
                    index=years.index(CURRENT_YEAR - 5),
                    help="Year of first registration (not manufacturing year).",
                )

                fuel = st.selectbox(
                    "Fuel type",
                    options=cats["fuels"],
                    index=default_fuel,
                    format_func=lambda x: x.title(),
                    help="Powertrain type as listed on the RC book.",
                )

            with col_r:
                km_driven = st.number_input(
                    "Odometer reading (km)",
                    min_value=0,
                    max_value=300000,
                    value=45000,
                    step=5000,
                    help="Current kilometers on the odometer. Capped at 3,00,000 km.",
                )

                transmission = st.selectbox(
                    "Transmission",
                    options=cats["transmissions"],
                    index=default_trans,
                    help="Manual or Automatic.",
                )

                owner = st.selectbox(
                    "Ownership",
                    options=cats["owners"],
                    index=0,
                    format_func=lambda x: f"{x.title()} owner",
                    help="Number of previous registered owners.",
                )

            submitted = st.form_submit_button(
                "Estimate fair value",
                icon=":material/search:",
                use_container_width=True,
                type="primary",
            )

        # ── Results ──────────────────────────────────────────────────
        if submitted:
            derived_age = CURRENT_YEAR - reg_year

            input_df = pd.DataFrame(
                [
                    {
                        "Brand": brand,
                        "Age": derived_age,
                        "kmDriven": float(km_driven),
                        "Transmission": transmission,
                        "Owner": owner,
                        "FuelType": fuel,
                    }
                ]
            )

            with st.spinner("Analysing market data..."):
                try:
                    result = predict_fair_price(input_df, model)
                except Exception as err:
                    st.error(f"Something went wrong: {err}", icon=":material/error:")
                    st.stop()

            # Vehicle summary
            st.caption(
                f"**{reg_year} {brand}** · {km_driven:,} km · "
                f"{fuel.title()} · {transmission} · {owner.title()} owner"
            )

            # Three-metric row
            col_lo, col_mid, col_hi = st.columns(3)
            with col_lo:
                st.metric(
                    label="Trade-in / Conservative",
                    value=fmt_inr(result["lower_bound"]),
                )
            with col_mid:
                st.metric(
                    label=":material/verified: Fair market value",
                    value=fmt_inr(result["predicted_price"]),
                )
            with col_hi:
                st.metric(
                    label="Retail / Dealer asking",
                    value=fmt_inr(result["upper_bound"]),
                )

            # Segment callout
            st.info(
                f"This vehicle is classified as **{result['bracket']}** "
                f"with a calibrated **±{result['margin_percentage']:.0f}%** price margin.",
                icon=":material/info:",
            )

            # Guardrail warning (Budget / Luxury)
            if result["warning"]:
                st.warning(result["warning"], icon=":material/warning:")

            # Educational expander
            with st.expander(
                "Why a range instead of a single price?",
                icon=":material/help:",
            ):
                st.markdown(
                    """
                    No algorithm can predict a car's exact resale value.
                    Several factors that significantly affect pricing are
                    not captured in structured datasets:

                    - **Physical condition** — scratches, dents, tyre wear
                    - **Service history** — full dealer records vs. none
                    - **Regional demand** — metro vs. tier-2 city premiums
                    - **Seasonal fluctuations** — festival-season spikes

                    The **trade-in value** represents a conservative
                    floor (what a dealer might offer), while the **retail
                    value** represents what you might see on a dealer
                    listing.  The **fair market value** is the model's
                    central estimate — a reasonable price for a private
                    sale between informed parties.

                    Vehicles under ₹5 Lakh and above ₹30 Lakh naturally
                    carry wider margins because small condition differences
                    swing a larger percentage of their base value.
                    """
                )

    # ═══════════════════════════════════════════════════════════════════
    # TAB 2 — ABOUT THIS PROJECT (Technical / Portfolio)
    # ═══════════════════════════════════════════════════════════════════
    with tab_about:
        st.header("About this project")
        st.markdown(
            "This application is the production front-end of a **classical supervised "
            "machine learning pipeline** built end-to-end — from raw data cleaning "
            "through model evaluation to deployment."
        )

        st.subheader(":material/precision_manufacturing: Model architecture")
        st.markdown(
            """
            | Attribute | Detail |
            |-----------|--------|
            | Algorithm | Tuned Random Forest Regressor |
            | Estimators | 300 trees, max depth 20, min samples split 10 |
            | Target space | Log-transformed (`log1p` → `expm1`) |
            | Training data | 10,900+ Indian pre-owned car listings |
            | Test holdout | 2,731 listings (20%) |
            | Test MAE | ₹3,09,522 |
            | Test RMSE | ₹11,69,145 |
            | Test R² | 0.54 |
            | Baseline delta | ~₹30,000 MAE improvement over Linear Regression |
            """
        )

        st.subheader(":material/query_stats: Segmented error analysis")
        st.markdown(
            "The model exhibits **heteroscedasticity** — prediction error varies "
            "by price segment. We calibrated per-segment uncertainty margins from "
            "holdout MAPE analysis:"
        )

        mape_data = pd.DataFrame(
            {
                "Price segment": [
                    "Budget (< ₹5L)",
                    "Mid-range (₹5L – ₹15L)",
                    "Premium (₹15L – ₹30L)",
                    "Luxury (> ₹30L)",
                ],
                "MAPE": ["37%", "23%", "25%", "34%"],
                "Margin applied": ["±37%", "±23%", "±25%", "±34%"],
            }
        )
        st.dataframe(mape_data, hide_index=True, use_container_width=True)

        # MAPE bar chart
        chart_df = pd.DataFrame(
            {
                "Segment": [
                    "Budget\n(< ₹5L)",
                    "Mid-range\n(₹5L–₹15L)",
                    "Premium\n(₹15L–₹30L)",
                    "Luxury\n(> ₹30L)",
                ],
                "MAPE (%)": [37, 23, 25, 34],
            }
        ).set_index("Segment")
        st.bar_chart(chart_df, use_container_width=True)

        st.subheader(":material/psychology: Key feature roles")
        st.markdown(
            """
            | Feature | Role in valuation |
            |---------|-------------------|
            | **Age** | Primary depreciation driver |
            | **Brand** | Determines pricing tier (economy vs. luxury) |
            | **Kilometers driven** | Usage-based value discount |
            | **Fuel type** | Diesel and hybrid models hold different residual values |
            | **Transmission** | Automatics trade at a premium in urban markets |
            | **Ownership** | Second-owner vehicles face steeper discounts |
            """
        )

        st.subheader(":material/construction: Known limitations")
        st.markdown(
            """
            - **Missing sub-model feature:** The specific model name (e.g. Swift
              vs. Brezza) was dropped due to high cardinality. A Maruti hatchback
              and a Maruti SUV currently share the same brand signal.
            - **No monotonic constraints:** Random Forests do not enforce that
              higher mileage → lower price. Rare input combinations may produce
              counter-intuitive steps.
            - **No condition scoring:** Physical inspection data, accident history,
              and service records are not available in the training data.
            """
        )

        st.subheader(":material/update: Future iterations")
        st.markdown(
            """
            - Introduce body-type categorisation (SUV, sedan, hatchback).
            - Explore gradient boosting with explicit monotonic constraints
              on `Age` and `kmDriven`.
            - Integrate a condition/inspection score input.
            """
        )

        st.caption(
            f"Full source code, notebooks, and methodology: [{GITHUB_URL}]({GITHUB_URL})"
        )


if __name__ == "__main__":
    main()
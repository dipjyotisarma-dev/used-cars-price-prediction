"""
app/views/project_view.py
Recruiter & ML Engineering view: model performance scorecard,
segmented error analysis, exploratory charts, and pipeline specs.
"""

import pandas as pd
import streamlit as st


def render_project_tab(dataset: pd.DataFrame, github_url: str):
    """Renders Tab 2: Recruiter scorecard, EDA visualizations, and engineering specs."""
    st.subheader("Model architecture & engineering scorecard")
    st.caption("End-to-end technical overview of the supervised regression pipeline.")

    # ── Top Scorecard ────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Cleaned dataset", "13,652 listings", "2000–2024")
    kpi2.metric("Test MAE", "₹3,09,522", "-₹30,000 vs baseline")
    kpi3.metric("Test R²", "0.54", "+17% vs baseline")
    kpi4.metric("Inference latency", "< 15 ms", "Joblib Pipeline")

    # ── Visual Analytics Row ─────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        with st.container(border=True):
            st.markdown("**:material/bar_chart: Top 8 pre-owned brands in India**")
            if not dataset.empty:
                top_brands = (
                    dataset["Brand"]
                    .value_counts()
                    .head(8)
                    .reset_index()
                )
                top_brands.columns = ["Brand", "Listings"]
                st.bar_chart(top_brands.set_index("Brand"))
            st.caption("Distribution across 13,652 processed market listings.")

    with col_r:
        with st.container(border=True):
            st.markdown("**:material/troubleshoot: Segmented uncertainty (MAPE)**")
            mape_df = pd.DataFrame(
                {
                    "Segment": ["Budget (< ₹5L)", "Mid-Range (₹5L-₹15L)", "Premium (₹15L-₹30L)", "Luxury (> ₹30L)"],
                    "Error Margin (%)": [37, 23, 25, 34],
                }
            ).set_index("Segment")
            st.bar_chart(mape_df)
            st.caption("Heteroscedasticity-adjusted dynamic price bands on holdout data.")

    # ── Architecture & Preprocessing ─────────────────────────────────
    with st.container(border=True):
        st.markdown("**:material/account_tree: Production pipeline specifications**")
        
        spec_col1, spec_col2 = st.columns(2)
        with spec_col1:
            st.markdown(
                """
                - **Algorithm:** Tuned `RandomForestRegressor` (300 trees, depth 20, min split 10)
                - **Target formulation:** `log1p(AskPrice)` → inverse `expm1()` post-processing
                - **Train / test split:** 80 / 20 stratified split (10,922 train / 2,732 test)
                """
            )
        with spec_col2:
            st.markdown(
                """
                - **Nominal features:** `OneHotEncoder(handle_unknown='ignore')` (`Brand`, `FuelType`, `Transmission`)
                - **Ordinal features:** `OrdinalEncoder` (`Owner`)
                - **Numerical scaling:** `StandardScaler` (`Age`, `kmDriven`)
                """
            )

    # ── Model Comparison Table ───────────────────────────────────────
    st.markdown("**:material/compare: Model benchmark progression**")
    comparison_data = pd.DataFrame(
        {
            "Model Stage": [
                "Baseline (Linear Regression)",
                "Default Random Forest",
                "Tuned Random Forest (Selected)",
            ],
            "Test MAE (INR)": ["₹3,39,520", "₹3,14,200", "₹3,09,522"],
            "Test RMSE (INR)": ["₹12,44,100", "₹11,85,400", "₹11,69,145"],
            "Test R²": [0.46, 0.52, 0.54],
            "Key Strengths": [
                "Fast, interpretable benchmark",
                "Captures non-linear relationships",
                "Optimized leaf depth, reduces overfitting",
            ],
        }
    )
    st.dataframe(comparison_data, hide_index=True)

    # ── Engineering Insights ─────────────────────────────────────────
    with st.expander("Known model constraints & future engineering roadmap", icon=":material/tips_and_updates:"):
        st.markdown(
            """
            - **Non-monotonicity:** Random Forests can occasionally yield non-monotonic leaf splits with respect to mileage. Future iterations will test monotonic gradient boosting (LightGBM/XGBoost).
            - **Model cardinality:** The granular `model` feature was omitted due to high cardinality across 13,000 samples. Body-type clustering (SUV vs Sedan) is the planned next iteration.
            - **No inspection data:** Paint condition, tyre health, and service records are unobserved in the public dataset.
            """
        )

    # ── Data Provenance & Dataset Attribution ────────────────────────
    with st.container(border=True):
        st.markdown("**:material/database: Data provenance & source**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown(
                """
                - **Primary source:** Indian Pre-Owned Car Market Dataset ([Kaggle](https://www.kaggle.com/datasets/mohitkumar282/used-car-dataset))
                - **Raw ingestion:** 14,993 multi-portal aggregated listings
                - **Cleaned volume:** 13,652 deduplicated records across 39 brands
                """
            )
        with p_col2:
            st.markdown(
                """
                - **Temporal range:** Vehicle registrations spanning 2000–2024
                - **Target variable:** `AskPrice` in Indian Rupees (INR ₹)
                - **Data processing:** Missing-value imputation, outlier capping (300k km), lowercase normalization
                """
            )

    st.caption(
        f"Full pipeline scripts, training notebooks, and dataset artifacts: [{github_url}]({github_url})"
    )
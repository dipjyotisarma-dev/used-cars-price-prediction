"""
app/views/valuation_view.py
Consumer valuation interface: input form, price prediction KPI cards,
and balanced market comps with top model benchmarks.
"""

from datetime import datetime
import pandas as pd
import streamlit as st

from src.pipeline import predict_fair_price
from app.data_loader import (
    fmt_inr,
    get_brand_market_stats,
    get_top_models_summary,
    get_comparable_listings,
)

CURRENT_YEAR = datetime.now().year


def render_valuation_tab(model, dataset: pd.DataFrame, cats: dict):
    """Renders Tab 1: Valuation form, prediction cards, and market comps."""
    st.subheader("Vehicle valuation & market benchmarks")
    st.caption("Configure vehicle parameters to compute the fair market price and inspect actual market comps.")

    default_brand = cats["brands"].index("Maruti Suzuki") if "Maruti Suzuki" in cats["brands"] else 0
    default_fuel = cats["fuels"].index("petrol") if "petrol" in cats["fuels"] else 0
    default_trans = cats["transmissions"].index("Manual") if "Manual" in cats["transmissions"] else 0

    # ── Valuation Form ───────────────────────────────────────────────
    with st.form("valuation_form", border=True):
        col_l, col_r = st.columns(2)

        with col_l:
            brand = st.selectbox(
                "Brand",
                options=cats["brands"],
                index=default_brand,
                help="Vehicle manufacturer as shown on the registration certificate.",
            )

            years = list(range(CURRENT_YEAR, CURRENT_YEAR - 21, -1))
            reg_year = st.selectbox(
                "Registration year",
                options=years,
                index=years.index(CURRENT_YEAR - 5),
                help="Year of first registration.",
            )

            fuel = st.selectbox(
                "Fuel type",
                options=cats["fuels"],
                index=default_fuel,
                format_func=lambda x: x.title(),
                help="Powertrain type.",
            )

        with col_r:
            km_driven = st.number_input(
                "Odometer reading (km)",
                min_value=0,
                max_value=300000,
                value=45000,
                step=5000,
                help="Current odometer reading in kilometers (capped at 3,00,000 km).",
            )

            transmission = st.selectbox(
                "Transmission",
                options=cats["transmissions"],
                index=default_trans,
                help="Transmission type.",
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
            type="primary",
        )

    # ── Prediction & Market Comps Display ────────────────────────────
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

        with st.spinner("Evaluating vehicle against market models..."):
            try:
                result = predict_fair_price(input_df, model)
            except Exception as err:
                st.error(f"Inference error: {err}", icon=":material/error:")
                return

        st.markdown(
            f"**Evaluation target:** `{reg_year} {brand}` · `{km_driven:,} km` · "
            f"`{fuel.title()}` · `{transmission}` · `{owner.title()} owner`"
        )

        # Three-metric KPI row
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

        # Dynamic tier banner
        st.info(
            f"Vehicle segment: **{result['bracket']}** · "
            f"Calibrated uncertainty margin: **±{result['margin_percentage']:.0f}%**",
            icon=":material/info:",
        )

        if result["warning"]:
            st.warning(result["warning"], icon=":material/warning:")

        # ── Grounded Real-World Comps Section ────────────────────────
        if not dataset.empty:
            with st.container(border=True):
                st.subheader(f":material/query_stats: Real market data for {brand}")
                st.caption(
                    "Our ML model predicts the brand tier average. Use these actual historical "
                    "transactions to calibrate for your specific car model."
                )

                # Brand Quick Stats (KPIs)
                brand_stats = get_brand_market_stats(dataset, brand)
                if brand_stats:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Brand listings in database", f"{brand_stats['total_listings']:,}")
                    c2.metric("Brand median price", fmt_inr(brand_stats["median_price"]))
                    c3.metric(
                        "Observed price range",
                        f"{fmt_inr(brand_stats['min_price'])} – {fmt_inr(brand_stats['max_price'])}",
                    )

                # Clean Sub-Tabs to eliminate whitespace asymmetry
                tab_top_models, tab_comps = st.tabs(
                    [
                        ":material/bar_chart: Top models price guide",
                        ":material/receipt_long: Comparable listings in dataset",
                    ]
                )

                with tab_top_models:
                    top_col1, top_col2 = st.columns([3, 2])
                    top_models_df = get_top_models_summary(dataset, brand, top_n=5)

                    if not top_models_df.empty:
                        with top_col1:
                            st.markdown("**Average asking price by model**")
                            chart_data = top_models_df.set_index("model")["Avg_Price"]
                            st.bar_chart(chart_data)

                        with top_col2:
                            st.markdown("**Model volume & median benchmark**")
                            display_top = top_models_df.copy()
                            display_top["Median Price"] = display_top["Median_Price"].apply(fmt_inr)
                            display_top["Average Price"] = display_top["Avg_Price"].apply(fmt_inr)
                            st.dataframe(
                                display_top[["model", "Listings", "Median Price", "Average Price"]],
                                hide_index=True,
                            )

                with tab_comps:
                    st.markdown(f"**Actual listings ({brand}, near {reg_year})**")
                    comps_df = get_comparable_listings(dataset, brand, reg_year, limit=6)
                    if not comps_df.empty:
                        comps_display = comps_df.copy()
                        comps_display["kmDriven"] = comps_display["kmDriven"].apply(lambda x: f"{x:,.0f} km")
                        comps_display["AskPrice"] = comps_display["AskPrice"].apply(fmt_inr)
                        comps_display.rename(
                            columns={
                                "model": "Model",
                                "Year": "Year",
                                "FuelType": "Fuel",
                                "Transmission": "Trans",
                                "kmDriven": "Mileage",
                                "AskPrice": "Actual Ask Price",
                            },
                            inplace=True,
                        )
                        st.dataframe(comps_display, hide_index=True)
                        st.caption("Historical pre-owned listings matching this brand and vintage.")

        # Negotiation Guide
        with st.expander("Negotiation strategy with these numbers", icon=":material/handshake:"):
            st.markdown(
                """
                - **Buying:** Use the **Conservative / Trade-in** price as your initial counter-offer.
                - **Selling:** List your vehicle close to the **Retail Asking** figure to leave headroom for bargaining.
                - **Variant adjustment:** Check the top models chart above; premium trims (e.g., SUVs) typically command the upper half of the price band.
                """
            )
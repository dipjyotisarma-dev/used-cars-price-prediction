"""
app/views/sidebar_view.py
Renders the navigation sidebar with quick badges, usage guidelines, and links.
"""

import streamlit as st


def render_sidebar(total_listings: int, github_url: str):
    """Renders a clean, structured sidebar."""
    with st.sidebar:
        st.header(":material/directions_car: Fair value engine")
        st.caption("Supervised ML engine for pre-owned car valuation in India.")

        # Quick stats badges
        with st.container(border=True):
            st.markdown(f"**Database scale:** `{total_listings:,}` listings")
            st.markdown("**Segment coverage:** `39` brands")
            st.markdown("**Data origin:** Kaggle (Indian Car Market)")
            st.markdown("**Core model:** Tuned Random Forest")

        st.subheader(":material/lightbulb: How to use")
        st.markdown(
            """
            - **Select exact brand & registration year** from your RC book.
            - **Enter current odometer reading** in kilometers.
            - The model provides a **fair value** and a **negotiation range**.
            - Check the **market comps section** to see how specific models benchmark.
            """
        )

        st.subheader(":material/verified: Fair value guide")
        st.markdown(
            """
            - **Conservative / Trade-in:** Realistic dealer offer floor.
            - **Fair market value:** Central benchmark for private sales.
            - **Retail asking:** Expected dealer listing price.
            """
        )

        st.caption(f"[View source code on GitHub]({github_url})")
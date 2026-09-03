"""
app/app.py
Used Car Fair Value Engine — Production Streamlit Entrypoint.
Coordinates caching, modular views, and page navigation.
"""

import sys
from pathlib import Path
import streamlit as st

# Ensure project root is available in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.data_loader import get_model, get_market_dataset, get_categories
from app.views.sidebar_view import render_sidebar
from app.views.valuation_view import render_valuation_tab
from app.views.project_view import render_project_tab

GITHUB_URL = "https://github.com/dipjyotisarma-dev/used-cars-price-prediction"

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Used Car Fair Value Engine",
    page_icon=":material/directions_car:",
    layout="wide",
)


def main():
    # 1. Load cached pipeline and market dataset
    try:
        model = get_model()
    except Exception as err:
        st.error(f"Failed to load ML pipeline: {err}", icon=":material/error:")
        st.stop()

    dataset = get_market_dataset()
    categories = get_categories(model)
    total_listings = len(dataset) if not dataset.empty else 13652

    # 2. Render Sidebar
    render_sidebar(total_listings, GITHUB_URL)

    # 3. Main Header
    st.title(":material/directions_car: Used car fair value engine")

    # 4. Two-tab Modular Presentation
    tab_valuation, tab_project = st.tabs(
        [
            ":material/calculate: Car valuation",
            ":material/analytics: Model architecture & metrics",
        ]
    )

    with tab_valuation:
        render_valuation_tab(model, dataset, categories)

    with tab_project:
        render_project_tab(dataset, GITHUB_URL)


if __name__ == "__main__":
    main()
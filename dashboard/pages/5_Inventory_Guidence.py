from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inventory.reorder import guidance_for_medicine

DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"

st.title("Inventory Reorder Guidance")

st.warning(
    "CSV मा वास्तविक current stock र supplier lead time छैन। "
    "त्यसैले current stock आफैँ राख्नुपर्छ र lead time अनुमानित "
    "रूपमा दिनुपर्छ। यो guidance मात्र हो।"
)

if not DATA_PATH.exists():
    st.error("पहिले ETL चलाऊ।")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

medicine_ids = sorted(df["Medicine_ID"].astype(str).unique())

medicine_id = st.selectbox(
    "Select Medicine",
    medicine_ids,
)

lead_time = st.number_input(
    "Assumed Lead Time (days)",
    min_value=1,
    max_value=180,
    value=7,
)

current_stock = st.number_input(
    "Current Stock (units)",
    min_value=0,
    value=0,
)

if st.button("Calculate Reorder Guidance"):
    try:
        result = guidance_for_medicine(
            df,
            medicine_id,
            lead_time_days=int(lead_time),
            current_stock=float(current_stock),
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Average Daily Demand",
            result["average_daily_demand"],
        )
        c2.metric(
            "Estimated Safety Stock",
            result["safety_stock_estimate"],
        )
        c3.metric(
            "Estimated Reorder Point",
            result["estimated_reorder_point"],
        )

        if result["stock_status"] == "Consider reorder":
            st.warning(result["stock_status"])
        else:
            st.info(result["stock_status"])

        st.json(result)

    except ValueError as exc:
        st.error(str(exc))
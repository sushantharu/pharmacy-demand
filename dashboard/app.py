from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"

st.set_page_config(
    page_title="Pharmacy Demand Intelligence",
    page_icon="💊",
    layout="wide",
)

st.title("💊 Pharmacy Demand Intelligence")
st.write(
    "Machine Learning-based medicine demand analysis "
    "and inventory guidance."
)

if not DATA_PATH.exists():
    st.warning(
        "Processed dataset भेटिएन। Project root बाट "
        "`python -m src.data.etl` चलाऊ।"
    )
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Quantity Used",
    f"{df['Quantity_Used'].sum():,.0f}",
)
col2.metric(
    "Unique Medicines",
    df["Medicine_ID"].nunique(),
)
col3.metric(
    "Categories",
    df["Category"].nunique(),
)

st.subheader("Daily Demand Trend")

daily = (
    df.groupby("Date")["Quantity_Used"]
    .sum()
    .sort_index()
)

st.line_chart(daily)

st.subheader("Category-wise Demand")

category = (
    df.groupby("Category")["Quantity_Used"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(category)

st.subheader("Top 10 Medicines")

top_medicines = (
    df.groupby("Medicine_ID")["Quantity_Used"]
    .sum()
    .nlargest(10)
    .reset_index()
)

st.dataframe(top_medicines, use_container_width=True)

st.caption(
    f"Historical period: {df['Date'].min().date()} "
    f"to {df['Date'].max().date()}"
)
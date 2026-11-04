from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"

st.title("Exploratory Data Analysis")

if not DATA_PATH.exists():
    st.error("पहिले ETL चलाऊ: python -m src.data.etl")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

start_date = st.date_input(
    "Start Date",
    value=df["Date"].min().date(),
)
end_date = st.date_input(
    "End Date",
    value=df["Date"].max().date(),
)

if start_date > end_date:
    st.error("Start Date, End Date भन्दा पछाडि हुन मिल्दैन।")
    st.stop()

filtered = df[
    df["Date"].between(
        pd.Timestamp(start_date),
        pd.Timestamp(end_date),
    )
]

if filtered.empty:
    st.warning("यो date range मा data छैन।")
    st.stop()

st.subheader("Dataset Summary")

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(filtered))
c2.metric("Total Quantity", int(filtered["Quantity_Used"].sum()))
c3.metric("Unique Medicines", filtered["Medicine_ID"].nunique())

st.subheader("Statistical Description")
st.dataframe(
    filtered["Quantity_Used"].describe().to_frame(),
    use_container_width=True,
)

daily = (
    filtered.groupby("Date", as_index=False)["Quantity_Used"]
    .sum()
)

fig = px.line(
    daily,
    x="Date",
    y="Quantity_Used",
    title="Daily Total Medicine Demand",
)
st.plotly_chart(fig, use_container_width=True)

category = (
    filtered.groupby("Category", as_index=False)["Quantity_Used"]
    .sum()
    .sort_values("Quantity_Used", ascending=False)
)

fig2 = px.bar(
    category,
    x="Category",
    y="Quantity_Used",
    title="Demand by Category",
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Data Quality")

quality = pd.DataFrame({
    "Metric": [
        "Missing values",
        "Duplicate rows",
        "Negative quantities",
    ],
    "Count": [
        int(filtered.isna().sum().sum()),
        int(filtered.duplicated().sum()),
        int((filtered["Quantity_Used"] < 0).sum()),
    ],
})

st.dataframe(quality, use_container_width=True)

st.download_button(
    "Download filtered data",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="pharmacy_filtered_data.csv",
    mime="text/csv",
)
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"

st.title("Medicine-wise Analysis")

if not DATA_PATH.exists():
    st.error("पहिले ETL चलाऊ।")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

medicine_ids = sorted(df["Medicine_ID"].astype(str).unique())

medicine_id = st.selectbox(
    "Select Medicine ID",
    medicine_ids,
)

medicine_df = df[
    df["Medicine_ID"].astype(str) == medicine_id
].sort_values("Date")

if medicine_df.empty:
    st.warning("Medicine को data भेटिएन।")
    st.stop()

first_row = medicine_df.iloc[0]

c1, c2, c3 = st.columns(3)

c1.metric(
    "Total Quantity",
    int(medicine_df["Quantity_Used"].sum()),
)
c2.metric(
    "Average Daily Demand",
    round(medicine_df["Quantity_Used"].mean(), 2),
)
c3.metric(
    "Maximum Daily Demand",
    int(medicine_df["Quantity_Used"].max()),
)

st.write("**Category:**", first_row["Category"])
st.write("**Unit:**", first_row["Unit"])
st.write("**Department:**", first_row["Department"])

fig = px.line(
    medicine_df,
    x="Date",
    y="Quantity_Used",
    title=f"Historical Demand: {medicine_id}",
)
st.plotly_chart(fig, use_container_width=True)

monthly = (
    medicine_df.set_index("Date")["Quantity_Used"]
    .resample("MS")
    .sum()
    .rename("Quantity_Used")
    .reset_index()
)

fig2 = px.bar(
    monthly,
    x="Date",
    y="Quantity_Used",
    title="Monthly Demand",
)
st.plotly_chart(fig2, use_container_width=True)

st.download_button(
    "Export Medicine History",
    data=medicine_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{medicine_id}_history.csv",
    mime="text/csv",
)
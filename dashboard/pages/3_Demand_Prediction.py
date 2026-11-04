from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.predict import predict_next_day

DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"

st.title("Medicine Demand Prediction")

if not DATA_PATH.exists():
    st.error("पहिले ETL चलाऊ।")
    st.stop()

if not (ROOT / "artifacts" / "models" / "best_regressor.joblib").exists():
    st.warning(
        "Regression model भेटिएन। Project root बाट "
        "`python -m src.models.train_regressor` चलाऊ।"
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

medicine_ids = sorted(df["Medicine_ID"].astype(str).unique())

medicine_id = st.selectbox(
    "Select Medicine",
    medicine_ids,
)

if st.button("Predict Next-day Demand", type="primary"):
    try:
        with st.spinner("Predicting..."):
            result = predict_next_day(medicine_id)

        st.success("Prediction completed.")

        c1, c2 = st.columns(2)

        c1.metric(
            "Predicted Quantity",
            f"{result['predicted_quantity']} {result['unit']}",
        )
        c2.metric(
            "Forecast Date",
            result["forecast_date"],
        )

        st.json(result)

        st.caption(
            "यो historical data मा आधारित अनुमान हो; "
            "वास्तविक demand फरक हुन सक्छ।"
        )

    except (ValueError, FileNotFoundError) as exc:
        st.error(str(exc))
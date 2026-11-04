from datetime import timedelta
import joblib
import pandas as pd

from src.features.feature_engineering import (
    make_features,
    FEATURES,
)
from src.models.utils import ROOT, MODEL_DIR


DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"
MODEL_PATH = MODEL_DIR / "best_regressor.joblib"


def predict_next_day(medicine_id: str) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Regression model not found. Run "
            "`python -m src.models.train_regressor` first."
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run ETL first."
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    df["Medicine_ID"] = df["Medicine_ID"].astype(str)
    medicine_df = df[
        df["Medicine_ID"] == str(medicine_id)
    ].copy()

    if medicine_df.empty:
        raise ValueError(f"Medicine ID not found: {medicine_id}")

    medicine_df = medicine_df.sort_values("Date")

    if medicine_df["Date"].duplicated().any():
        raise ValueError(
            "Duplicate daily records found for this medicine."
        )

    # Do not silently interpret missing dates as zero demand.
    expected_dates = pd.date_range(
        medicine_df["Date"].min(),
        medicine_df["Date"].max(),
        freq="D",
    )

    if not medicine_df["Date"].reset_index(drop=True).equals(
        pd.Series(expected_dates, name="Date")
    ):
        raise ValueError(
            "Some dates are missing for this medicine. "
            "Verify whether these dates mean zero demand or missing data."
        )

    history = make_features(medicine_df)

    last_date = history["Date"].max()
    next_date = last_date + timedelta(days=1)
    last_row = history.iloc[-1]

    next_row = {
        "Date": next_date,
        "Medicine_ID": last_row["Medicine_ID"],
        "Category": last_row["Category"],
        "Unit": last_row["Unit"],
        "Department": last_row["Department"],
        "Quantity_Used": 0,
    }

    # The placeholder quantity is not used as a feature:
    # lag/rolling features use only observations before the forecast date.
    extended = pd.concat(
        [medicine_df, pd.DataFrame([next_row])],
        ignore_index=True,
    )
    extended = make_features(extended)

    forecast_row = extended.iloc[[-1]]

    if forecast_row[FEATURES].isna().any().any():
        raise ValueError(
            "Not enough history to calculate lag and rolling features."
        )

    model = joblib.load(MODEL_PATH)
    prediction = max(
        0.0,
        float(model.predict(forecast_row[FEATURES])[0]),
    )

    return {
        "medicine_id": str(medicine_id),
        "forecast_date": next_date.date().isoformat(),
        "predicted_quantity": round(prediction, 2),
        "unit": str(last_row["Unit"]),
    }


if __name__ == "__main__":
    selected_id = input("Enter Medicine_ID: ").strip()

    try:
        print(predict_next_day(selected_id))
    except (ValueError, FileNotFoundError) as error:
        print("Prediction error:", error)
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "features.csv"

NUMERIC_FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "quarter",
    "is_weekend",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_mean_30",
]

CATEGORICAL_FEATURES = [
    "Medicine_ID",
    "Category",
    "Unit",
    "Department",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Quantity_Used"


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(
        ["Medicine_ID", "Date"]
    ).reset_index(drop=True)

    df["day_of_week"] = df["Date"].dt.dayofweek
    df["day_of_month"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    grouped = df.groupby("Medicine_ID")["Quantity_Used"]

    df["lag_1"] = grouped.shift(1)
    df["lag_7"] = grouped.shift(7)

    df["rolling_mean_7"] = (
        df.groupby("Medicine_ID")["Quantity_Used"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=7, min_periods=7
            ).mean()
        )
    )

    df["rolling_mean_30"] = (
        df.groupby("Medicine_ID")["Quantity_Used"]
        .transform(
            lambda s: s.shift(1).rolling(
                window=30, min_periods=30
            ).mean()
        )
    )

    return df


def run_feature_engineering():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Run ETL first: python -m src.data.etl"
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["Date"])
    feature_df = make_features(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    print("Feature engineering completed.")
    print("Shape:", feature_df.shape)
    print("Saved:", OUTPUT_PATH)

    return feature_df


if __name__ == "__main__":
    run_feature_engineering()
from pathlib import Path
import json
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from src.features.feature_engineering import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES,
)

ROOT = Path(__file__).resolve().parents[2]
FEATURE_PATH = ROOT / "data" / "processed" / "features.csv"
MODEL_DIR = ROOT / "artifacts" / "models"
REPORT_DIR = ROOT / "artifacts" / "reports"


def load_feature_data():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            "Run feature engineering first."
        )

    df = pd.read_csv(FEATURE_PATH, parse_dates=["Date"])

    required = FEATURES + ["Quantity_Used", "Date"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.dropna(subset=FEATURES + ["Quantity_Used"])
    return df.sort_values("Date").reset_index(drop=True)


def chronological_split(df, test_fraction=0.20):
    dates = sorted(df["Date"].unique())

    if len(dates) < 10:
        raise ValueError(
            "Not enough unique dates for a reliable time split."
        )

    split_index = int(len(dates) * (1 - test_fraction))
    split_index = max(1, min(split_index, len(dates) - 1))
    split_date = dates[split_index]

    train_df = df[df["Date"] < split_date].copy()
    test_df = df[df["Date"] >= split_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or testing set is empty.")

    return train_df, test_df, pd.Timestamp(split_date)


def build_preprocessor():
    transformer = ColumnTransformer(
        transformers=[
            (
                "numeric",
                "passthrough",
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return transformer


def build_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", model),
    ])


def ensure_artifact_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
import pandas as pd
import pytest

from src.data.cleaning import clean_data
from src.features.feature_engineering import make_features
from src.inventory.reorder import calculate_reorder_guidance


def sample_data():
    return pd.DataFrame({
        "Date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-02",
        ],
        "Medicine_ID": ["MED001", "MED001", "MED001"],
        "Category": ["Tablet", "Tablet", "Tablet"],
        "Unit": ["Piece", "Piece", "Piece"],
        "Department": ["General", "General", "General"],
        "Quantity_Used": [10, 5, 0],
    })


def test_clean_data_keeps_zero_quantity():
    df = clean_data(sample_data())

    assert len(df) == 3
    assert (df["Quantity_Used"] >= 0).all()
    assert 0 in df["Quantity_Used"].values


def test_clean_data_rejects_missing_columns():
    df = pd.DataFrame({"Date": ["2025-01-01"]})

    with pytest.raises(ValueError):
        clean_data(df)


def test_feature_lag_values():
    dates = pd.date_range(
        "2025-01-01",
        periods=35,
        freq="D",
    )

    df = pd.DataFrame({
        "Date": dates,
        "Medicine_ID": ["MED001"] * 35,
        "Category": ["Tablet"] * 35,
        "Unit": ["Piece"] * 35,
        "Department": ["General"] * 35,
        "Quantity_Used": list(range(1, 36)),
    })

    result = make_features(df)

    assert result.loc[7, "lag_1"] == 7
    assert result.loc[7, "lag_7"] == 1
    assert result.loc[7, "rolling_mean_7"] == 4


def test_reorder_guidance():
    dates = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    history = pd.DataFrame({
        "Date": dates,
        "Quantity_Used": [10] * 10,
    })

    result = calculate_reorder_guidance(
        history,
        lead_time_days=3,
        current_stock=20,
    )

    assert result["average_daily_demand"] == 10
    assert result["estimated_reorder_point"] == 30
    assert result["stock_status"] == "Consider reorder"


def test_reorder_rejects_missing_dates():
    history = pd.DataFrame({
        "Date": pd.to_datetime([
            "2025-01-01",
            "2025-01-03",
        ]),
        "Quantity_Used": [10, 12],
    })

    with pytest.raises(ValueError):
        calculate_reorder_guidance(
            history,
            lead_time_days=3,
        )
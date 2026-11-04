from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import math

def calculate_reorder_guidance(
    history: pd.DataFrame,
    lead_time_days: int = 7,
    safety_factor: float = 1.65,
    current_stock: float | None = None,
):
    if lead_time_days < 1:
        raise ValueError("Lead time must be at least 1 day.")

    if safety_factor < 0:
        raise ValueError("Safety factor cannot be negative.")

    if current_stock is not None and current_stock < 0:
        raise ValueError("Current stock cannot be negative.")

    required_columns = {"Date", "Quantity_Used"}

    if not required_columns.issubset(history.columns):
        raise ValueError(
            "Data must contain Date and Quantity_Used columns."
        )

    data = history[["Date", "Quantity_Used"]].copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Quantity_Used"] = pd.to_numeric(
        data["Quantity_Used"], errors="coerce"
    )

    data = data.dropna().sort_values("Date")

    if data.empty:
        raise ValueError("No valid demand history found.")

    if (data["Quantity_Used"] < 0).any():
        raise ValueError("Quantity_Used cannot contain negative values.")

    if data["Date"].dt.normalize().duplicated().any():
        raise ValueError(
            "Multiple records per day found. Aggregate daily demand first."
        )

    # Ensure each calendar day is represented.
    # Missing dates are not automatically treated as zero demand.
    data["Date"] = data["Date"].dt.normalize()
    data = data.set_index("Date").sort_index()

    full_dates = pd.date_range(
        data.index.min(), data.index.max(), freq="D"
    )

    if not data.index.equals(full_dates):
        raise ValueError(
            "Daily dates are missing. Complete the daily data before "
            "calculating reorder guidance."
        )

    average_daily_demand = data["Quantity_Used"].mean()
    daily_std = data["Quantity_Used"].std(ddof=1)

    if pd.isna(daily_std):
        daily_std = 0.0

    safety_stock = (
        safety_factor * daily_std * math.sqrt(lead_time_days)
    )

    reorder_point = (
        average_daily_demand * lead_time_days + safety_stock
    )

    result = {
        "average_daily_demand": round(float(average_daily_demand), 2),
        "lead_time_days_assumed": lead_time_days,
        "safety_stock_estimate": round(float(safety_stock), 2),
        "estimated_reorder_point": math.ceil(reorder_point),
        "current_stock_provided": current_stock,
        "stock_status": "Unknown",
        "note": (
            "Estimated guidance only. Confirm actual stock, supplier "
            "lead time, and pharmacy requirements."
        ),
    }

    if current_stock is not None:
        if current_stock <= reorder_point:
            result["stock_status"] = "Consider reorder"
        else:
            result["stock_status"] = (
                "Stock above estimated reorder point"
            )

    return result


def guidance_for_medicine(
    df: pd.DataFrame,
    medicine_id: str,
    lead_time_days: int = 7,
    current_stock: float | None = None,
):
    medicine_df = df[
        df["Medicine_ID"].astype(str) == str(medicine_id)
    ].copy()

    if medicine_df.empty:
        raise ValueError(f"Medicine ID not found: {medicine_id}")

    return calculate_reorder_guidance(
        medicine_df[["Date", "Quantity_Used"]],
        lead_time_days=lead_time_days,
        current_stock=current_stock,
    )
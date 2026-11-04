def validate_data(df):
    report = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(
            df.isna().sum().sum()
        ),
        "duplicate_rows": int(
            df.duplicated().sum()
        ),
        "unique_medicines": int(
            df["Medicine_ID"].nunique()
        ),
        "unique_dates": int(
            df["Date"].nunique()
        ),
        "negative_quantities": int(
            (df["Quantity_Used"] < 0).sum()
        ),
    

    }
    return report
import pandas as pd

REQUIRED_COLUMNS = [
    "Date",
    "Medicine_ID",
    "Category",
    "Unit",
    "Department",
    "Quantity_Used",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[REQUIRED_COLUMNS]

    # Convert data types
    df["Date"] = pd.to_datetime(
        df["Date"], errors="coerce"
    )

    df["Quantity_Used"] = pd.to_numeric(
        df["Quantity_Used"], errors="coerce"
    )

    # Clean text columns
    text_columns = [
        "Medicine_ID",
        "Category",
        "Unit",
        "Department",
    ]

    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace("", pd.NA)

    # Remove invalid rows
    df = df.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Quantity cannot be negative
    df = df[df["Quantity_Used"] >= 0]

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Convert quantity to integer
    df["Quantity_Used"] = (
        df["Quantity_Used"].round().astype(int)
    )

    return df.reset_index(drop=True)
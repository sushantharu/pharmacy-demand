from pathlib import Path
import pandas as pd

from src.data.cleaning import clean_data
from src.data.validation import validate_data

RAW_PATH = Path(r"C:\Users\Admin\Desktop\Data Science\Farmacy\pharmacy_demand.csv")
PROCESSED_PATH = Path("data/processed")

def run_etl():
    PROCESSED_PATH.mkdir(
        parents=True, exist_ok=True
    )

    # Extract
    raw_df = pd.read_csv(RAW_PATH)

    # Transform: clean
    cleaned_df = clean_data(raw_df)

    # Aggregate at Date + Medicine level
    daily_df = (
        cleaned_df.groupby(
            ["Date", "Medicine_ID"],
            as_index=False
        )
        .agg(
            Quantity_Used=("Quantity_Used", "sum"),
            Category=("Category", "first"),
            Unit=("Unit", "first"),
            Department=("Department", "first"),
        )
    )

    # Load
    output_path = (
        PROCESSED_PATH / "clean_daily_data.csv"
    )

    daily_df.to_csv(output_path, index=False)

    print("ETL completed.")
    print(validate_data(daily_df))
    print("Saved:", output_path)

    return daily_df


if __name__ == "__main__":
    run_etl()
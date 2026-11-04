from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "clean_daily_data.csv"
REPORT_PATH = ROOT / "artifacts" / "reports"


def run_eda():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Run ETL first: python -m src.data.etl"
        )

    REPORT_PATH.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])

    print("\n--- DATA INFORMATION ---")
    print(df.info())

    print("\n--- DESCRIPTIVE STATISTICS ---")
    print(df["Quantity_Used"].describe())

    print("\n--- CATEGORY DEMAND ---")
    print(
        df.groupby("Category")["Quantity_Used"]
        .sum()
        .sort_values(ascending=False)
    )

    # 1. Daily total demand trend
    daily = df.groupby("Date")["Quantity_Used"].sum()

    plt.figure(figsize=(14, 5))
    daily.plot()
    plt.title("Daily Pharmacy Medicine Demand")
    plt.xlabel("Date")
    plt.ylabel("Total Quantity Used")
    plt.tight_layout()
    plt.savefig(REPORT_PATH / "daily_demand.png", dpi=150)
    plt.close()

    # 2. Demand by category
    category = (
        df.groupby("Category")["Quantity_Used"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 5))
    category.plot(kind="bar")
    plt.title("Total Demand by Medicine Category")
    plt.xlabel("Category")
    plt.ylabel("Quantity Used")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_PATH / "category_demand.png", dpi=150)
    plt.close()

    # 3. Top 15 medicines
    top_medicines = (
        df.groupby("Medicine_ID")["Quantity_Used"]
        .sum()
        .nlargest(15)
        .sort_values()
    )

    plt.figure(figsize=(10, 7))
    top_medicines.plot(kind="barh")
    plt.title("Top 15 Medicines by Total Demand")
    plt.xlabel("Quantity Used")
    plt.tight_layout()
    plt.savefig(REPORT_PATH / "top_medicines.png", dpi=150)
    plt.close()

    # 4. Quantity distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(
        df["Quantity_Used"],
        bins=30,
        kde=True
    )
    plt.title("Distribution of Daily Medicine Quantity")
    plt.tight_layout()
    plt.savefig(REPORT_PATH / "quantity_distribution.png", dpi=150)
    plt.close()

    print("\nEDA completed.")
    print("Charts saved in:", REPORT_PATH)


if __name__ == "__main__":
    run_eda()
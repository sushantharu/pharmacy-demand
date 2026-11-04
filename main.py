import argparse
import sys
import traceback

from src.data.etl import run_etl
from src.analysis.eda import run_eda
from src.features.feature_engineering import run_feature_engineering
from src.models.train_classifier import train_classifiers
from src.models.train_regressor import train_regressors
from src.models.evaluate import main as show_evaluation


def run_pipeline(skip_eda=False):
    print("\nPHARMACY DEMAND FORECASTING SYSTEM")

    print("\n[1/5] Data cleaning and ETL")
    run_etl()

    if not skip_eda:
        print("\n[2/5] Exploratory Data Analysis")
        run_eda()
    else:
        print("\n[2/5] EDA skipped")

    print("\n[3/5] Feature Engineering")
    run_feature_engineering()

    print("\n[4/5] Classification Training")
    train_classifiers()

    print("\n[5/5] Regression Training")
    train_regressors()

    print("\nModel evaluation")
    show_evaluation()

    print("\nPipeline completed.")
    print("Run dashboard:")
    print("streamlit run dashboard/app.py")


def main():
    parser = argparse.ArgumentParser(
        description="Pharmacy Demand Forecasting System"
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Skip EDA charts",
    )

    args = parser.parse_args()

    try:
        run_pipeline(skip_eda=args.skip_eda)
    except Exception as exc:
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
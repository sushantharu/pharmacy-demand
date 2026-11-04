import json

from src.models.utils import REPORT_DIR


def show_report(filename, title):
    path = REPORT_DIR / filename

    print(f"\n{'=' * 50}")
    print(title)
    print("=" * 50)

    if not path.exists():
        print("Report not found. Train the model first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)

    print("Best model:", report.get("best_model"))
    print("Test split date:", report.get("split_date"))

    for name, metrics in report.get("models", {}).items():
        print(f"\n{name}")
        for metric, value in metrics.items():
            if metric not in [
                "confusion_matrix",
                "classification_report",
            ]:
                print(f"  {metric}: {value}")

        if "confusion_matrix" in metrics:
            print("  Confusion matrix:")
            print(metrics["confusion_matrix"])


def main():
    show_report(
        "classification_metrics.json",
        "CLASSIFICATION EVALUATION",
    )
    show_report(
        "regression_metrics.json",
        "REGRESSION EVALUATION",
    )


if __name__ == "__main__":
    main()
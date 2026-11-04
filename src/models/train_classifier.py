import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.models.utils import (
    load_feature_data,
    chronological_split,
    build_pipeline,
    ensure_artifact_dirs,
    MODEL_DIR,
    REPORT_DIR,
)

MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    ),
    "DecisionTree": DecisionTreeClassifier(
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}


def train_classifiers():
    ensure_artifact_dirs()

    df = load_feature_data()
    train_df, test_df, split_date = chronological_split(df)

    # Threshold is calculated from training data only.
    threshold = float(train_df["Quantity_Used"].median())

    y_train = (
        train_df["Quantity_Used"] > threshold
    ).astype(int)
    y_test = (
        test_df["Quantity_Used"] > threshold
    ).astype(int)

    # Some datasets may have only one class after filtering.
    if y_train.nunique() < 2:
        raise ValueError(
            "Training target has only one class. "
            "Check the quantity distribution."
        )

    X_train = train_df.drop(
        columns=["Date", "Quantity_Used"]
    )
    X_test = test_df.drop(
        columns=["Date", "Quantity_Used"]
    )

    results = {}
    best_model = None
    best_name = None
    best_f1 = -1

    for name, estimator in MODELS.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "precision": float(
                precision_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y_test, predictions, zero_division=0
                )
            ),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "classification_report": classification_report(
                y_test,
                predictions,
                labels=[0, 1],
                target_names=["Low Demand", "High Demand"],
                zero_division=0,
                output_dict=True,
            ),
        }

        results[name] = metrics

        print(f"\n--- {name} ---")
        print(json.dumps(metrics, indent=2))

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_model = pipeline
            best_name = name

    joblib.dump(
        best_model,
        MODEL_DIR / "best_classifier.joblib",
    )

    report = {
        "task": "high_low_demand_classification",
        "best_model": best_name,
        "threshold": threshold,
        "split_date": str(split_date),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "models": results,
    }

    with open(
        REPORT_DIR / "classification_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nBest classifier:", best_name)
    print("Saved model:", MODEL_DIR / "best_classifier.joblib")


if __name__ == "__main__":
    train_classifiers()
import json
import joblib
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=3,
        loss="squared_error",
        random_state=42,
    ),
}


def train_regressors():
    ensure_artifact_dirs()

    df = load_feature_data()
    train_df, test_df, split_date = chronological_split(df)

    X_train = train_df.drop(
        columns=["Date", "Quantity_Used"]
    )
    X_test = test_df.drop(
        columns=["Date", "Quantity_Used"]
    )

    y_train = train_df["Quantity_Used"]
    y_test = test_df["Quantity_Used"]

    results = {}
    best_model = None
    best_name = None
    best_mae = float("inf")

    for name, estimator in MODELS.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(X_train, y_train)

        # Negative predictions are not meaningful for demand.
        predictions = np.maximum(
            0, pipeline.predict(X_test)
        )

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )
        r2 = r2_score(y_test, predictions)

        metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }
        results[name] = metrics

        print(f"\n--- {name} ---")
        print(metrics)

        # MAE is used as the model selection criterion.
        if mae < best_mae:
            best_mae = mae
            best_model = pipeline
            best_name = name

    joblib.dump(
        best_model,
        MODEL_DIR / "best_regressor.joblib",
    )

    report = {
        "task": "quantity_regression",
        "best_model": best_name,
        "split_date": str(split_date),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "models": results,
    }

    with open(
        REPORT_DIR / "regression_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(report, f, indent=2)

    print("\nBest regressor:", best_name)
    print("Saved model:", MODEL_DIR / "best_regressor.joblib")


if __name__ == "__main__":
    train_regressors()
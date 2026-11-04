from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "artifacts" / "reports"

st.title("Machine Learning Model Evaluation")


def read_json_report(filename):
    path = REPORT_DIR / filename

    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


classification = read_json_report(
    "classification_metrics.json"
)
regression = read_json_report(
    "regression_metrics.json"
)

st.subheader("Classification Results")

if classification is None:
    st.info(
        "पहिले `python -m src.models.train_classifier` चलाऊ।"
    )
else:
    st.write("Best Classifier:", classification["best_model"])

    rows = []

    for name, metrics in classification["models"].items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1"],
        })

    result_df = pd.DataFrame(rows).sort_values(
        "F1 Score", ascending=False
    )

    st.dataframe(result_df, use_container_width=True)

    selected_model = st.selectbox(
        "Choose model for Confusion Matrix",
        list(classification["models"].keys()),
    )

    matrix = classification["models"][
        selected_model
    ]["confusion_matrix"]

    fig = px.imshow(
        matrix,
        text_auto=True,
        x=["Predicted Low", "Predicted High"],
        y=["Actual Low", "Actual High"],
        labels={"x": "Predicted", "y": "Actual"},
        title=f"{selected_model} Confusion Matrix",
    )

    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Regression Results")

if regression is None:
    st.info(
        "पहिले `python -m src.models.train_regressor` चलाऊ।"
    )
else:
    st.write("Best Regressor:", regression["best_model"])

    rows = [
        {"Model": name, **metrics}
        for name, metrics in regression["models"].items()
    ]

    result_df = pd.DataFrame(rows).sort_values("mae")

    st.dataframe(result_df, use_container_width=True)

    st.caption(
        "MAE र RMSE कम हुनु सामान्यतया राम्रो हो। "
        "R² लाई अन्य metrics र test period सँगै हेर्नुपर्छ।"
    )
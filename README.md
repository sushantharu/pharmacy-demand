💊 Pharmacy Medicine Demand Forecasting
Estimate medicine demand and get reorder guidance through a Streamlit dashboard.

Decision-support prototype only. Forecasts are estimates and must be reviewed by a pharmacy professional.

Problem
Pharmacies need enough stock to avoid stockouts without overstocking slow-moving medicines. Manual estimates are unreliable because demand is irregular.

Features
ETL pipeline: cleans raw CSV into daily demand data
EDA: trends, medicine comparison, data-quality checks
Regression model for demand prediction
Model evaluation (MAE, RMSE, R²) against a baseline
Reorder point and safety stock guidance
Multi-page Streamlit dashboard
Tech Stack
Python, pandas, scikit-learn, Streamlit, Plotly, Git/GitHub

Dataset
Date, Medicine_ID, Quantity_Used (one row per medicine per day). Source: add dataset source here

Project Structure
data/        raw and processed CSVs
src/         ETL, features, analysis, models, inventory
dashboard/   Streamlit app and pages
artifacts/   saved model
tests/       tests
main.py
Setup
python -m venv venv
.\venv\Scripts\Activate.ps1     # Windows
pip install -r requirements.txt
Run
python main.py                          # ETL
python -m src.models.train_regressor    # train model
python -m streamlit run dashboard/app.py
Evaluation
Model	MAE	RMSE	R²
Baseline	TBD	TBD	TBD
Regression	TBD	TBD	TBD
Reorder Logic
Safety stock  = z × σ × √L
Reorder point = (avg daily demand × L) + Safety stock
L = lead time (default 7 days), z = safety factor, σ = daily demand std. Status shows "Consider reorder" if current stock ≤ reorder point.

Limitations
Historical data only; no expiry, pack size, or supplier logic
Default lead time is generic
Prototype: no authentication or monitoring
Live Demo
Add Streamlit URL here
https://pharmacy-demand.streamlit.app/


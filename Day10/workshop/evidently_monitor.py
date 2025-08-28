import pandas as pd
import joblib
import numpy as np

from evidently.core.report import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import MinValue, MaxValue, MeanValue, StdValue, MAE, RMSE, R2Score

# 1. Load trained model
model = joblib.load("model.pkl")

# 2. Load reference and current datasets
reference_data = pd.read_csv("ice_cream.csv")
current_data = pd.read_csv("current_data.csv")

# 3. Generate predictions
reference_data["prediction"] = model.predict(reference_data[["temp"]])
current_data["prediction"] = model.predict(current_data[["temp"]])


# 4. Build the report
report = Report(metrics=[
    DataDriftPreset(),
    MinValue(column="temp"),
    MaxValue(column="temp"),
    MeanValue(column="temp"),
    StdValue(column="temp"),
    MinValue(column="price"),
    MaxValue(column="price"),
    MeanValue(column="price"),
    StdValue(column="price"),
])


# 5. Run and save the report

result = report.run(reference_data=reference_data, current_data=current_data)
result.save_html("evidently_report.html")
print("✅ Evidently report generated: evidently_report.html")
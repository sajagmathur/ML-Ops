import pandas as pd
import joblib
import numpy as np

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metrics import ColumnValueRangeMetric, ColumnHistogramMetric

# 1. Load trained model
model = joblib.load("model.pkl")

# 2. Load reference and current datasets
reference_data = pd.read_csv("ice_cream.csv")
current_data = pd.read_csv("current_data.csv")

# 3. Generate predictions
reference_data["prediction"] = model.predict(reference_data[["temp"]])
current_data["prediction"] = model.predict(current_data[["temp"]])

# 4. Build full report with multiple metric sets
report = Report(metrics=[
    DataDriftPreset(),  # Checks for feature distribution changes
    RegressionPreset(target_column="price", prediction_column="prediction"),  # MAE, RMSE, R²
    DataQualityPreset(),  # Missing values, types, etc.
    
    # 🔍 Custom distribution metrics
    ColumnHistogramMetric(column_name="temp"),
    ColumnValueRangeMetric(column_name="temp"),
    ColumnHistogramMetric(column_name="price"),
    ColumnValueRangeMetric(column_name="price")
])

# 5. Run and save the report
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("evidently_report.html")

print("✅ Evidently report generated: evidently_report.html")

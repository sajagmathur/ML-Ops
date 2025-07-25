import sys
sys.stdout.reconfigure(encoding='utf-8')

import time
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
import numpy as np
import subprocess
import webbrowser

# Load model and test data
model = joblib.load("model.pkl")
X_test, y_test = joblib.load("test_data.pkl")

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Start MLflow server in the background
process = subprocess.Popen(
    ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000"]
)

print("MLflow server starting in the background...")

# Wait a bit to ensure the MLflow server is ready before logging
time.sleep(5)  # you can increase this if needed

# Open MLflow UI in the browser
webbrowser.open("http://127.0.0.1:5000")
print("MLflow UI opened in the browser.")

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment
mlflow.set_experiment("IceCreamModel")

# Prepare input example and signature for logging
input_example = X_test[:5]
signature = infer_signature(X_test, y_pred)

with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Log and register model with 'name' param instead of deprecated artifact_path
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",  # still needed to specify path inside the run's artifacts
        input_example=input_example,
        signature=signature
    )

    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="IceCreamModel"
    )

print("Model registered to MLflow")

# Optional: keep the MLflow server running or kill it when done
# process.terminate()

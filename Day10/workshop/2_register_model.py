# register.py

import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

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

import subprocess

# Start MLflow server in the background
process = subprocess.Popen(
    ["mlflow", "server", "--host", "127.0.0.1", "--port", "5000"]
)

print("MLflow server started in the background.")

#open mlflow ui in browser
import webbrowser
webbrowser.open("http://127.0.0.1:5000")

print("MLflow UI opened in the browser.")

mlflow.set_tracking_uri("http://127.0.0.1:5000")


# Log to MLflow
mlflow.set_experiment("IceCreamModel")

with mlflow.start_run() as run:
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log and register model
    mlflow.sklearn.log_model(model, artifact_path="model")
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model",
        name="IceCreamModel"
    )

print("Model registered to MLflow")

import os
import zipfile
from pathlib import Path

project_name = "ml-project"
base_path = Path(project_name)
dirs = [
    base_path / ".github" / "workflows",
    base_path / "src",
    base_path / "tests",
    base_path / "k8s"
]

files = {
    base_path / "requirements.txt": """scikit-learn
pandas
mlflow
joblib
fastapi
uvicorn
pytest
flake8
""",
    base_path / "MLproject": """name: ml-regression-demo

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.5}
    command: "python src/train.py --alpha {alpha}"
""",
    base_path / ".flake8": """[flake8]
max-line-length = 88
exclude = .git,__pycache__,venv
""",
    base_path / ".github" / "workflows" / "ci-cd.yml": """name: CI/CD Pipeline with FastAPI and MLflow

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Lint code
      run: |
        pip install flake8
        flake8 src tests

    - name: Run unit tests
      run: |
        pip install pytest
        pytest tests

    - name: Train model and log with MLflow
      run: |
        mlflow run . -P alpha=0.5
""",
    base_path / "src" / "train.py": '''import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import argparse

def train_model(alpha):
    X = pd.DataFrame({"x": range(10)})
    y = X["x"] * 3 + 5
    model = Ridge(alpha=alpha)
    model.fit(X, y)

    joblib.dump(model, "src/model.pkl")

    mlflow.log_param("alpha", alpha)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("src/model.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("internal-regression")
    with mlflow.start_run():
        train_model(args.alpha)
''',
    base_path / "src" / "serve.py": '''from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()
model = joblib.load("src/model.pkl")

class Input(BaseModel):
    x: float

@app.post("/predict")
def predict(data: Input):
    prediction = model.predict([[data.x]])
    return {"prediction": prediction[0]}
''',
    base_path / "tests" / "test_train.py": '''import joblib
import os

def test_model_file_exists():
    assert os.path.exists("src/model.pkl")

def test_prediction_output():
    model = joblib.load("src/model.pkl")
    result = model.predict([[2]])
    assert isinstance(result[0], float)
''',
    base_path / "k8s" / "fastapi-deployment.yaml": '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-ml
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fastapi-ml
  template:
    metadata:
      labels:
        app: fastapi-ml
    spec:
      containers:
      - name: fastapi
        image: tiangolo/uvicorn-gunicorn-fastapi:python3.9
        ports:
        - containerPort: 80
        volumeMounts:
        - name: model-volume
          mountPath: /app/src
      volumes:
      - name: model-volume
        hostPath:
          path: /path/to/ml-project/src

---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-service
spec:
  type: NodePort
  selector:
    app: fastapi-ml
  ports:
    - protocol: TCP
      port: 80
      targetPort: 80
      nodePort: 30036
'''
}

# Create folders and write files
for d in dirs:
    os.makedirs(d, exist_ok=True)

for file_path, content in files.items():
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        f.write(content)

# Zip it
zip_filename = "ml_project_package.zip"
with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
    for foldername, _, filenames in os.walk(project_name):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            arcname = os.path.relpath(file_path, project_name)
            zipf.write(file_path, arcname)

print(f"Created {zip_filename}")

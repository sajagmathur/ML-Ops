import mlflow
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

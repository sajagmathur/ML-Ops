import joblib
import os

def test_model_file_exists():
    assert os.path.exists("src/model.pkl")

def test_prediction_output():
    model = joblib.load("src/model.pkl")
    result = model.predict([[2]])
    assert isinstance(result[0], float)

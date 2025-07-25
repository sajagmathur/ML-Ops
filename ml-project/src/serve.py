from fastapi import FastAPI
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

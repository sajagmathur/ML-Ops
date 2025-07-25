# app.py

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
import numpy as np
import joblib
import os

# Load trained model
try:
    model = joblib.load("model.pkl")
except Exception as e:
    import sys
    print(f"Error loading model: {e}", file=sys.stderr)
    raise


# Setup FastAPI app
app = FastAPI()

# Mount static files (if needed)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Serve homepage
@app.get("/", response_class=HTMLResponse)
def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Predict route (API)
class InputData(BaseModel):
    temps: List[float]

@app.post("/predict")
async def predict_sales(data: InputData):
    input_array = np.array(data.temps).reshape(-1, 1)
    prediction = model.predict(input_array)
    return {"predicted_sales": prediction.tolist()}

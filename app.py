# app.py

import os
# Hide TensorFlow/Keras info/warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('absl').setLevel('ERROR')  # Hide absl warnings

# app.py
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Initialize FastAPI app
app = FastAPI()

# Mount static folder (optional)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load trained LSTM model
model = load_model("lstm_model.h5")
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

# Predict route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, prices: str = Form(...)):
    try:
        # Convert comma-separated string to float list
        price_list = [float(x.strip()) for x in prices.split(",")]

        # Check length = 60
        if len(price_list) != 60:
            return templates.TemplateResponse(
                "home.html",
                {
                    "request": request,
                    "prediction_text": "Error: Please enter exactly 60 values."
                }
            )

        # Reshape to match scaler (60,1)
        X = np.array(price_list).reshape(-1,1)
        X_scaled = scaler.transform(X)

        # Reshape for LSTM (1 sample, 60 timesteps, 1 feature)
        X_lstm = X_scaled.reshape(1, 60, 1)

        # Predict next stock price
        prediction = model.predict(X_lstm)[0][0]

        return templates.TemplateResponse(
            "home.html",
            {
                "request": request,
                "prediction_text": f"Predicted next stock price: {prediction:.2f}"
            }
        )

    except Exception as e:
        return templates.TemplateResponse(
            "home.html",
            {"request": request, "prediction_text": f"Error predicting. Try again. ({str(e)})"}
        )

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

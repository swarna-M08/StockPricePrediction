from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("lstm_model.h5")
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Stock Price Prediction API")

# Input schema
class StockInput(BaseModel):
    last_60_days: list  # list of last 60 closing prices

@app.get("/")
def home():
    return {"message": "Welcome to Stock Price Prediction API"}

@app.post("/predict")
def predict_stock(input_data: StockInput):
    data = np.array(input_data.last_60_days).reshape(-1,1)
    scaled_data = scaler.transform(data)

    # Prepare LSTM input
    X_input = scaled_data.reshape(1, 60, 1)

    # Predict next 7 days
    predictions = []
    current_batch = X_input.copy()
    for i in range(7):
        pred = model.predict(current_batch)[0,0]
        predictions.append(pred)
        current_batch = np.append(current_batch[:,1:,:], [[[pred]]], axis=1)

    # Inverse transform to original scale
    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1,1)).flatten()

    # Calculate confidence band (±1.96*std)
    error_std = np.std(predicted_prices) * 0.05  # simple approx 5% error
    upper_band = predicted_prices + 1.96*error_std
    lower_band = predicted_prices - 1.96*error_std

    result = []
    for i in range(7):
        result.append({
            "day": i+1,
            "predicted_price": float(predicted_prices[i]),
            "lower_bound": float(lower_band[i]),
            "upper_bound": float(upper_band[i])
        })

    return {"next_7_days_prediction": result}

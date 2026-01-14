import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

data = pd.read_csv("aapl_ticker.csv") 
closing_prices = data['Close'].values.reshape(-1, 1)  

# Fit scaler
scaler = MinMaxScaler()
scaler.fit(closing_prices)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("scaler.pkl successfully created!")

import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, sequence_length=60):
    X = []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
    return np.array(X)

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset (Replace 'traffic_data.csv' with your actual dataset file)
df = pd.read_csv("traffic_data.csv")  

# Convert datetime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'])

# Extract useful time-based features
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month

# Create a traffic count feature by grouping data
df['traffic_count'] = df.groupby(['hour'])['ID'].transform('count')

# Drop unnecessary columns
df.drop(columns=['ID', 'DateTime'], inplace=True)

# Normalize traffic data
scaler = MinMaxScaler()
df[['traffic_count', 'hour', 'day_of_week', 'month']] = scaler.fit_transform(df[['traffic_count', 'hour', 'day_of_week', 'month']])

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
traffic_data = df['traffic_count'].values
X, y = create_sequences(traffic_data, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Define LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])

# Compile model
model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# Train model
model.fit(X, y, epochs=20, batch_size=16)

# Save the trained model
model.save("traffic_forecast_model.h5")
print("Model saved successfully!")

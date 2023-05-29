import numpy as np
import pandas as pd
import ccxt
import time
import openpyxl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Function to fetch data
def fetch_data(symbol, timeframe, from_date, to_date):
    exchange = ccxt.binance({
        'timeout': 30000,
        'enableRateLimit': True,
    })
    since = exchange.parse8601(from_date + 'T00:00:00Z')
    to = exchange.parse8601(to_date + 'T00:00:00Z')
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, to)
    data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    return data


# Fetch Bitcoin data
btc_data = fetch_data('BTC/USDT', '1d', '2014-01-01', '2022-12-31')

# Function to calculate Simple Moving Average
def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()

# Function to calculate Exponential Moving Average
def EMA(data, period=20, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Calculate SMA and EMA
btc_data['SMA'] = SMA(btc_data)
btc_data['EMA'] = EMA(btc_data)

# We'll use the 'close' prices to predict the next closing price
dataset = btc_data[['close', 'SMA', 'EMA']].values
dataset = dataset.astype('float32')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# Reshape into X=t and Y=t+1
look_back = 15
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)


# Save the model for later use
model.save('btc_price_prediction_model.keras')
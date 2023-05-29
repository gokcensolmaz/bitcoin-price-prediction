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

# We'll use the 'close' prices to predict the next closing price
dataset = btc_data['close'].values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# save the scaler
joblib.dump(scaler, 'scaler.pkl')

# Split the data into training and testing sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# Reshape into X=t and Y=t+1
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# Save the model for later use
model.save('btc_price_prediction_model.keras')

# Load the saved model
model = load_model('btc_price_prediction_model.keras', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load('scaler.pkl')

# Load the original dataset
btc_data = fetch_data('BTC/USDT', '1d', '2014-01-01', '2022-12-31')
dataset = btc_data['close'].values
dataset = dataset.astype('float32')
dataset = np.reshape(dataset, (-1, 1))

# Normalize the data using the loaded scaler
dataset = scaler.transform(dataset)

# Split the data into training and testing sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# Reshape the data for LSTM input
look_back = 7
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % trainScore)
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % testScore)

# Create indices for plotting
train_indices = range(look_back, look_back + len(trainPredict))
test_indices = range(len(dataset) - test_size + look_back, len(dataset))

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(dataset, label='Actual Price', color='blue')
plt.plot(train_indices, trainPredict, label='Train Predictions', color='green')
plt.plot(test_indices, testPredict[:, 0], label='Test Predictions', color='red')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

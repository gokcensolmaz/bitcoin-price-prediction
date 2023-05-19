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




def fetch_data(symbol, timeframe, from_date, to_date):
    exchange = ccxt.binance({
        'timeout': 30000,
        'enableRateLimit': True,
    })
    since = exchange.parse8601(from_date + 'T00:00:00Z')
    to = exchange.parse8601(to_date + 'T00:00:00Z')
    # Fetch OHLCV (Open/High/Low/Close/Volume) data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, to)

    # Convert to DataFrame for easier manipulation
    data = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

    # Convert timestamp to datetime format
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    return data




# Fetch Bitcoin data from 2020 to 2022
btc_data = fetch_data('BTC/USDT', '1d', '2020-01-01', '2022-12-31')
btc_data.isnull().sum()
btc_data.info()
# Fetch Ethereum data from 2020 to 2022
eth_data = fetch_data('ETH/USDT', '1d', '2020-01-01', '2022-12-31')

btc_data.to_excel("btc_data.xlsx")
eth_data.to_excel("eth_data.xlsx")

# Function to calculate Simple Moving Average
def SMA(data, period=30, column='close'):
    return data[column].rolling(window=period).mean()

# Function to calculate Exponential Moving Average
def EMA(data, period=20, column='close'):
    return data[column].ewm(span=period, adjust=False).mean()

# Add SMA and EMA to the dataframes
btc_data['SMA'] = SMA(btc_data)
btc_data['EMA'] = EMA(btc_data)

eth_data['SMA'] = SMA(eth_data)
eth_data['EMA'] = EMA(eth_data)


# Function to calculate Fibonacci Retracement Levels
def calculate_fibonacci_retracement_levels(data):
    high = max(data['high'])
    low = min(data['low'])

    diff = high - low
    levels = {
        'level_0': high,
        'level_1': high - 0.236 * diff,
        'level_2': high - 0.382 * diff,
        'level_3': high - 0.618 * diff,
        'level_4': low
    }
    return levels


# Calculate Fibonacci Retracement Levels for BTC and ETH
btc_fib_levels = calculate_fibonacci_retracement_levels(btc_data)
eth_fib_levels = calculate_fibonacci_retracement_levels(eth_data)

print("BTC Fibonacci Levels: ", btc_fib_levels)
print("ETH Fibonacci Levels: ", eth_fib_levels)

btc_data['Fib_Level_0'] = btc_fib_levels['level_0']
btc_data['Fib_Level_1'] = btc_fib_levels['level_1']
btc_data['Fib_Level_2'] = btc_fib_levels['level_2']
btc_data['Fib_Level_3'] = btc_fib_levels['level_3']
btc_data['Fib_Level_4'] = btc_fib_levels['level_4']

def SMA(data, period=30, column='close'):
    sma = data[column].rolling(window=period).mean()
    sma[:period] = sma[period]  # Fill missing SMA values with the first valid SMA value
    return sma

#selecting columns to use as parameter
data = btc_data[['close', 'SMA', 'EMA', 'Fib_Level_0', 'Fib_Level_1', 'Fib_Level_2', 'Fib_Level_3', 'Fib_Level_4']].values
btc_data['SMA'] = SMA(btc_data)
# Call SMA function again to fill missing SMA values
btc_data['SMA'] = SMA(btc_data)

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows
print(btc_data.head(30))

# Normalize the data
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Create method to create a time-series dataset
def create_dataset(data, steps):
    X = []
    y = []
    for i in range(len(data) - steps - 1):
        X.append(data[i:(i + steps), :])
        y.append(data[i + steps, 0])
    X = np.array(X)
    y = np.array(y)
    print("X shape:", X.shape)  # Add this line to print the shape of X
    return X, y

# We'll use the past 60 days' closing prices to predict the next closing price
steps = 60

X_train, y_train = create_dataset(train_data, steps)
X_test, y_test = create_dataset(test_data, steps)

# Reshape the features for the LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 8))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 8))


# Define the LSTM model
model = Sequential()

model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=100, return_sequences=False))
model.add(Dense(units=1))

# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Save the model for later use
model.save_weights('btc_price_prediction_model_weights.h5')


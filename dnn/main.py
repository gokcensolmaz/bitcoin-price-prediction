import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load BTC price data
btc_data = pd.read_excel('btc.xlsx')
# Convert the 'Date' column to datetime
btc_data['Date'] = pd.to_datetime(btc_data['Date'])
# Replace commas with periods and convert affected columns to numeric data type
btc_data['Open'] = btc_data['Open'].replace(',', '.', regex=True).astype(float)
btc_data['High'] = btc_data['High'].replace(',', '.', regex=True).astype(float)
btc_data['Low'] = btc_data['Low'].replace(',', '.', regex=True).astype(float)
btc_data['Close'] = btc_data['Close'].replace(',', '.', regex=True).astype(float)
btc_data['Adj Close'] = btc_data['Adj Close'].replace(',', '.', regex=True).astype(float)
btc_data['Volume'] = btc_data['Volume'].replace(',', '.', regex=True).astype(float)

# Load DXY data
dxy_data = pd.read_csv('dxy.csv', delimiter=';')
# Convert the 'Date' column to datetime
dxy_data['Date'] = pd.to_datetime(dxy_data['Date'])
# Calculate the percentage change in DXY
dxy_data['DXY_Change'] = dxy_data['Close'].pct_change()
# Replace NaN values with 0
dxy_data['DXY_Change'].fillna(0, inplace=True)
# Take the negative of DXY_Change to capture the inverse correlation
dxy_data['DXY_Change'] = -dxy_data['DXY_Change']

# Perform inner merge on 'Date' column to keep only common dates
merged_data = pd.merge(btc_data, dxy_data[['Date', 'DXY_Change']], on='Date', how='inner')

# Continue with the rest of your code...
X = merged_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'DXY_Change']]
y = merged_data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(X_test, y_test)
print('Mean Squared Error on Test Set:', loss)

# Perform predictions on the test set
predictions = model.predict(X_test)

# Plotting the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(predictions.flatten(), label='Predicted')
plt.xlabel('Index')
plt.ylabel('Close Price')
plt.title('Actual vs. Predicted Close Price')
plt.legend()
plt.show()

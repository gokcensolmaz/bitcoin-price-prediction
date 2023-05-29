import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from modelcreation import btc_data, pd, scaler

# Load the saved model
model = load_model('btc_price_prediction_model.keras', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare input data for prediction
look_back = 15

# Extract the last look_back data points from the dataset
last_sequence = btc_data[['close', 'SMA', 'EMA']].tail(look_back).values

# Scale the data
scaled_sequence = scaler.transform(last_sequence)

# Reshape the input data
prediction_input = np.reshape(scaled_sequence, (1, look_back, 3))

# Make predictions
predictions = []
for _ in range(30):
    predicted_price = model.predict(prediction_input)
    predictions.append(predicted_price[0, 0])
    # Prepare the input for the next prediction
    next_sequence = np.concatenate((prediction_input[:, 1:, :], np.reshape(predicted_price, (1, 1, 1))), axis=1)
    prediction_input = scaler.transform(next_sequence)

# Invert predictions
predictions = np.array(predictions)
predictions = scaler.inverse_transform(np.concatenate((predictions.reshape(-1, 1), np.zeros((30, 2))), axis=1))

# Generate dates for the next 30 days
last_date = btc_data['timestamp'].iloc[-1]
dates = pd.date_range(last_date, periods=30, freq='D')

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(btc_data['timestamp'], btc_data['close'], label='Actual Price', color='blue')
plt.plot(dates, predictions[:, 0], label='Predicted Prices', color='red')

# Plot true values of predicted range
next_30_days_actual_prices = btc_data['close'].tail(30)
plt.plot(dates, next_30_days_actual_prices, label='True Values', color='green')

plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from modelcreation import dataset, look_back,btc_data,pd
import joblib

# Load the saved model
model = load_model('btc_price_prediction_model.keras', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load('scaler.pkl')

# Prepare input data for prediction
last_sequence = dataset[-look_back:]
prediction_input = np.reshape(last_sequence, (1, 1, look_back))

# Make predictions
predictions = []
for _ in range(30):
    predicted_price = model.predict(prediction_input)
    predictions.append(predicted_price[0, 0])
    prediction_input = np.append(prediction_input[:, :, 1:], np.reshape(predicted_price, (1, 1, 1)), axis=2)

# Invert predictions
predictions = np.array(predictions)
predictions = scaler.inverse_transform([[p] for p in predictions])

# Generate dates for the next 30 days
last_date = btc_data['timestamp'].iloc[-1]
dates = pd.date_range(last_date, periods=30, freq='D')

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(btc_data['timestamp'], scaler.inverse_transform(dataset), label='Actual Price', color='blue')
plt.plot(dates, predictions, label='Predicted Prices', color='red')

# Plot true values of predicted range
next_30_days_actual_prices = scaler.inverse_transform(dataset[-30:])
plt.plot(dates, next_30_days_actual_prices, label='True Values', color='green')

plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
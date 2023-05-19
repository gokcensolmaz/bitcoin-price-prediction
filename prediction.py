import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from modelcreation import X_train, scaler, X_test, y_train, y_test, train_data, create_dataset, test_data

# Load the saved model
new_model = Sequential()
new_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1))) # Ensure input shape matches your model
new_model.add(LSTM(units=50, return_sequences=False))
new_model.add(Dense(units=1))

new_model.compile(optimizer='adam', loss='mean_squared_error')

# Load the weights into the new model
new_model.load_weights('btc_price_prediction_model_weights.h5')


# Make predictions
train_predictions = new_model.predict(X_train)
test_predictions = new_model.predict(X_test)

# Un-normalize the predicted values
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate RMSE to evaluate the model

train_RMSE = np.sqrt(mean_squared_error(y_train, train_predictions))
test_RMSE = np.sqrt(mean_squared_error(y_test, test_predictions))

print('Train RMSE: ', train_RMSE)
print('Test RMSE: ', test_RMSE)

# Get the last 60 days closing price values and convert the dataframe to an array
last_60_days = train_data[-60:]

# Append test data to the last 60 days data
df_test = np.concatenate([last_60_days, test_data])

# Create the testing data set
X_test_new, _ = create_dataset(df_test, 60)

# Convert the test data to numpy arrays
X_test= np.array(X_test)

# Reshape the data
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))

# Get the predicted values
predictions = new_model.predict(X_test_new)

# Undo the scaling
predictions = scaler.inverse_transform(predictions)

# Calculate the mean squared error
mse = np.mean((predictions - y_test)**2)
print('Mean Squared Error: ', mse)

# Un-normalize the actual values
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the data
plt.figure(figsize=(8,4))
plt.plot(y_test_actual, color='blue', label='Actual Bitcoin Price')
plt.plot(predictions , color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Bitcoin Price')
plt.legend()
plt.show()
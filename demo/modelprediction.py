import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
from modelcreation import dataset, trainX, testX, trainY,testY,look_back, test_size
import joblib

# Load the saved model
model = load_model('btc_price_prediction_model.keras', compile=False)
model.compile(optimizer='adam', loss='mean_squared_error')
scaler = joblib.load('scaler.pkl')

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Reshape the trainPredict and testPredict arrays
trainPredict = trainPredict.reshape(-1, 1)
testPredict = testPredict.reshape(-1, 1)

# Invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % trainScore)
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % testScore)

# Create indices for plotting
train_indices = range(look_back, look_back + len(trainPredict))
test_indices = range(len(dataset) - test_size + look_back, len(dataset) + 1)

# Plot baseline and predictions
plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(dataset), label='Actual Price', color='blue')
plt.plot(train_indices, trainPredict, label='Train Predictions', color='green')
plt.plot(test_indices, testPredict, label='Test Predictions', color='red')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
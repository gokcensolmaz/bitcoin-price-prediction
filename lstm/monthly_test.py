# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import RobustScaler
plt.style.use("bmh")
import datetime

# Technical Analysis library
import ta
# Neural Network library
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

df = pd.read_excel("btc.xlsx")

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df.Date)

# Set 'Date' as the index
df.set_index('Date', inplace=True)

# Drop any NaNs
df.dropna(inplace=True)

# Add all TA features
df = ta.add_all_ta_features(df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
df.to_excel("btc_with_ta_features.xlsx")
# Drop unnecessary columns
df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

# Take the last 1000 days of data
df = df.head(1000)

# Split the data into 80% for training and 20% for testing
train_size = int(len(df) * 0.8)
train_df = df[:train_size]
test_df = df[train_size:]

# Initialize a scaler with the training data
close_scaler = RobustScaler()
close_scaler.fit(train_df[['Close']])

# Normalize/Scale the train and test dataframes
scaler = RobustScaler()

train_df = pd.DataFrame(scaler.fit_transform(train_df), columns=train_df.columns, index=train_df.index)
test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns, index=test_df.index)

def split_sequence(seq, n_steps_in, n_steps_out):
    # Creating a list for both variables
    X, y = [], []

    for i in range(len(seq)):

        # Find the end of this sequence
        end = i + n_steps_in
        out_end = end + n_steps_out

        # Break if we have exceeded the dataset's length
        if out_end > len(seq):
            break

        # Split the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def visualize_training_results(results):
    history = results.history
    plt.figure(figsize=(16, 5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    plt.figure(figsize=(16, 5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


def layer_maker(n_layers, n_nodes, activation, drop=None, d_rate=.5):
    for x in range(1, n_layers + 1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass


def val_rmse(df1, df2):
    df = df1.copy()
    df['close2'] = df2.Close
    df.dropna(inplace=True)
    df['diff'] = df.Close - df.close2
    rms = (df[['diff']] ** 2).mean()
    return np.sqrt(rms.values[0])

# Define model parameters
n_per_in = 90
n_per_out = 30
n_features = train_df.shape[1]
# Split the training and test data into appropriate sequences
X_train, y_train = split_sequence(train_df.to_numpy(), n_per_in, n_per_out)
X_test, y_test = split_sequence(test_df.to_numpy(), n_per_in, n_per_out)

# Instatiate the model
model = Sequential()
activ = "tanh"

# Input layer
model.add(LSTM(90, activation=activ, return_sequences=True, input_shape=(n_per_in, n_features)))

# Hidden layers
layer_maker(n_layers=1, n_nodes=30, activation=activ)

# Final Hidden layer
model.add(LSTM(60, activation=activ))

# Output layer
model.add(Dense(n_per_out))

# Model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Fit the model to the training data
res = model.fit(X_train, y_train, epochs=50, batch_size=128, validation_split=0.1)

# Make predictions on the test data
predictions = model.predict(X_test)

# Get the root mean squared error for the predictions
print("RMSE:", np.sqrt(np.mean((predictions - y_test)**2)))

# Flatten the predictions
flat_predictions = [item for sublist in predictions for item in sublist]

# Considering only first prediction from each sequence for comparison with actual values
predicted_prices = flat_predictions[::n_per_out]

# Transform the actual values to their original price
# Transform the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(test_df[["Close"]]),
                      index=test_df.index,
                      columns=[test_df.columns[0]])

# Match the length of actual prices and predicted prices
actual_prices = actual[n_per_in + n_per_out - 1: n_per_in + n_per_out - 1 + len(predicted_prices)]

# Transform the predicted prices to their original price
predicted_prices = close_scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Create a DataFrame for the predicted prices with the shifted index
predicted_prices_df = pd.DataFrame(predicted_prices, index=actual_prices.index, columns=[actual.columns[0]])

# Plotting
plt.figure(figsize=(16, 6))

# Plotting the actual prices
plt.plot(actual_prices, label="Actual Prices")

# Plotting the predicted prices
plt.plot(predicted_prices_df, label="Predicted Prices")
plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()

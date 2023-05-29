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

btc_data = pd.read_excel("btc.xlsx")
dxy_data = pd.read_csv("dxy.csv", sep=';')

## Datetime conversion
btc_data['Date'] = pd.to_datetime(btc_data.Date)

# Convert date column in dxy_data to yyyy-mm-dd format
dxy_data['Date'] = pd.to_datetime(dxy_data['Date'])

merged_data = pd.merge(btc_data, dxy_data, on="Date", how="inner")

# Set 'Date' column as the index
merged_data.set_index('Date', inplace=True)

# Drop any NaN values
merged_data.dropna(inplace=True)
print(merged_data.columns)

# Apply technical indicators using TA-Lib
merged_data = ta.add_all_ta_features(merged_data, open="Open_x", high="High_x", low="Low_x", close="Close_x", volume="Volume_x", fillna=True)

# Dropping everything else besides 'Close' and the Indicators
merged_data.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

# Only using the last 1000 days of data to get a more accurate representation of the current market climate
merged_data = merged_data.tail(1000)

## Scaling

# Scale fitting the close prices separately for inverse_transformations purposes later
close_scaler = RobustScaler()

close_scaler.fit(btc_data[['Close']])

# Normalizing/Scaling the DF
scaler = RobustScaler()

scaled_data = pd.DataFrame(scaler.fit_transform(merged_data), columns=merged_data.columns, index=merged_data.index)

def split_sequence(seq, n_steps_in, n_steps_out):
    """
    Splits the multivariate time sequence
    """

    # Creating a list for both variables
    X, y = [], []

    for i in range(len(seq)):

        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out

        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break

        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]

        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)

def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
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

def layer_maker(model, n_layers, n_nodes, activation, drop=None, d_rate=.5):
    """
    Creates a specified number of hidden layers for an RNN
    Optional: Adds regularization option - the dropout layer to prevent potential overfitting (if necessary)
    """

    # Creating the specified number of hidden layers with the specified number of nodes
    for x in range(1, n_layers + 1):
        model.add(LSTM(n_nodes, activation=activation, return_sequences=True))

        # Adds a Dropout layer after every Nth hidden layer (the 'drop' variable)
        try:
            if x % drop == 0:
                model.add(Dropout(d_rate))
        except:
            pass

def validater(model, n_per_in, n_per_out, btc_data, close_scaler):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """

    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=btc_data.index, columns=[btc_data.columns[0]])

    for i in range(n_per_in, len(btc_data) - n_per_in, n_per_out):
        # Creating rolling intervals to predict off of
        x = btc_data[-i - n_per_in:-i]

        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

        # Transforming values back to their normal prices
        yhat = close_scaler.inverse_transform(yhat)[0]

        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhat,
                               index=pd.date_range(start=x.index[-1],
                                                   periods=len(yhat),
                                                   freq="B"),
                               columns=[x.columns[0]])

        # Updating the predictions DF
        predictions.update(pred_df)

    return predictions

def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()

    # Adding a new column with the closing prices from the second DF
    df['close2'] = df2.Close

    # Dropping the NaN values
    df.dropna(inplace=True)

    # Adding another column containing the difference between the two DFs' closing prices
    df['diff'] = df.Close - df.close2

    # Squaring the difference and getting the mean
    rms = (df[['diff']] ** 2).mean()

    # Returning the square root of the root mean square
    return np.sqrt(rms.values[0])

# How many periods looking back to learn
n_per_in  = 90
# How many periods to predict
n_per_out = 30
# Features
n_features = scaled_data.shape[1]
# Splitting the data into appropriate sequences
X, y = split_sequence(scaled_data.to_numpy(), n_per_in, n_per_out)

## Creating the NN

# Instantiate the model
model = Sequential()

# Input layer
model.add(LSTM(90, activation='tanh', return_sequences=True, input_shape=(n_per_in, n_features)))

# Hidden layers
layer_maker(model, n_layers=1, n_nodes=30, activation='tanh')

# Final Hidden layer
model.add(LSTM(60, activation='tanh'))

# Output layer
model.add(Dense(n_per_out))

# Model summary
model.summary()

# Compiling the data with selected specifications
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

## Fitting and Training
res = model.fit(X, y, epochs=50, batch_size=128, validation_split=0.1)

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(scaled_data[["Close"]]),
                      index=scaled_data.index,
                      columns=[scaled_data.columns[0]])

# Getting a DF of the predicted values to validate against
predictions = validater(model, n_per_in, n_per_out, btc_data, close_scaler)

# Printing the RMSE
print("RMSE:", val_rmse(actual, predictions))

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(predictions, label='Predicted')
plt.plot(actual, label='Actual')
plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predicting off of the most recent days from the original DF
yhat = model.predict(np.array(merged_data.tail(n_per_in)).reshape(1, n_per_in, n_features))

# Transforming the predicted values back to their original format
yhat = close_scaler.inverse_transform(yhat)[0]

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat,
                     index=pd.date_range(start=merged_data.index[-1] + timedelta(days=1),
                                         periods=len(yhat),
                                         freq="B"),
                     columns=[merged_data.columns[0]])

# Number of periods back to plot the actual values
pers = n_per_in

# Transforming the actual values to their original price
actual = pd.DataFrame(close_scaler.inverse_transform(btc_data[["Close"]].tail(pers)),
                      index=btc_data.Close.tail(pers).index,
                      columns=[btc_data.columns[0]])

# Printing the predicted prices
print(preds)

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(actual, label="Actual Prices")
plt.plot(preds, label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()

import datetime as dt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpl

from sklearn.preprocessing import MinMaxScaler
from keras.api._v2 import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Chose Crypto Asset:
crypto = 'ETH'
currency = 'USD'

prediction_days = 60

start_time = dt.datetime(2023, 1, 1).strftime("%Y-%m-%d")
end_time = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")

# Download dataset:

data = yf.download(tickers=f'{crypto}-{currency}',
                   start=start_time, end=end_time)

# TRAIN MODEL WITH DATASET

x_train, y_train = [], []

scalar = MinMaxScaler(feature_range=(0, 1))
scaled_data = scalar.fit_transform(data['Close'].values.reshape(-1, 1))

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True,
          input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=25, batch_size=32)

# DOWNLOAD ACTUAL PRICES

test_start = dt.datetime(2023, 1, 1).strftime("%Y-%m-%d")
test_end = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")

# Download dataset:

test_data = yf.download(tickers=f'{crypto}-{currency}',
                        start=test_start, end=test_end)
actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# PREPARE DATA

model_inputs = total_dataset[len(
    total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scalar.fit_transform(model_inputs)

x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scalar.inverse_transform(prediction_prices)

# PLOT DATA

plt.plot(actual_prices, color='black', label='Actual Price')
plt.plot(prediction_prices, color='green', label='AI Predicted Price')
plt.title(f'{crypto} AI and Actual Price Comparison')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend(loc='upper left')
plt.show()

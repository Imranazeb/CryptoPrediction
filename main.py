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

crypto = 'ETH'
currency = 'USD'

prediction_days = 60

start_time = dt.datetime(2023, 1, 1).strftime("%Y-%m-%d")
end_time = (dt.datetime.now() - dt.timedelta(days=30)).strftime("%Y-%m-%d")

# Download dataset:

data = yf.download(tickers=f'{crypto}-{currency}',
                   start=start_time, end=end_time)

mpl.plot(data, type='candle', style='yahoo', volume=True)

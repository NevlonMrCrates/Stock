import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

start = '1990-01-01'
end = '2023-12-25'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 1990-01-01 to 2023-12-25')
st.write(df.describe())

st.subheader('Closing Prices vs Time Chart')
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Prices vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
plt.plot(ma100)
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Prices vs Time Chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_train = pd.DataFrame(df['Close'][0:int(len(df) * 0.90)])
data_test = pd.DataFrame(df['Close'][int(len(df) * 0.90):])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
data_train_arr = scaler.fit_transform(data_train)

x_train, y_train = [], []

for i in range(100, len(data_train_arr)):
    x_train.append(data_train_arr[i - 100:i, 0])
    y_train.append(data_train_arr[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='relu', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(units=80, activation='relu', return_sequences=True))
model.add(Dropout(0.4))

model.add(LSTM(units=120, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=50)

data500 = data_train.tail(500)
final_df = data500.append(data_test, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i, 0])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

y_predicted = model.predict(x_test)
nm = scaler.scale_
scale_factor = 1 / nm[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Original vs Predicted')
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

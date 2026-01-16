import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    layout="wide"
)

st.title("🚗 Tesla Stock Price Prediction Dashboard")
st.markdown("**SimpleRNN vs LSTM | Time Series Forecasting**")

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

# --------------------------------------------------
# Prepare Data (same as training)
# --------------------------------------------------
data = df[['Adj Close']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

WINDOW_SIZE = 60

def create_sequences(data, window):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, WINDOW_SIZE)

split = int(0.8 * len(X))
X_test = X[split:]
y_test = y[split:]

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# --------------------------------------------------
# Load Models (NO TRAINING)
# --------------------------------------------------
simple_rnn = load_model("simple_rnn_best.h5")
lstm_model = load_model("lstm_best.h5")

# --------------------------------------------------
# Predictions
# --------------------------------------------------
rnn_pred = simple_rnn.predict(X_test)
lstm_pred = lstm_model.predict(X_test)

rnn_pred = scaler.inverse_transform(rnn_pred)
lstm_pred = scaler.inverse_transform(lstm_pred)
y_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("📊 Controls")

model_choice = st.sidebar.radio(
    "Select Model",
    ["SimpleRNN", "LSTM"]
)

# --------------------------------------------------
# Plot: Actual vs Predicted
# --------------------------------------------------
st.subheader("📈 Actual vs Predicted Tesla Stock Prices")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_actual, label="Actual Price", color="black")

if model_choice == "SimpleRNN":
    ax.plot(rnn_pred, label="SimpleRNN Prediction", color="blue")
else:
    ax.plot(lstm_pred, label="LSTM Prediction", color="orange")

ax.set_xlabel("Time")
ax.set_ylabel("Stock Price (USD)")
ax.legend()

st.pyplot(fig)

# --------------------------------------------------
# Future Forecast
# --------------------------------------------------
def forecast_future(model, last_seq, days):
    future = []
    seq = last_seq.copy()

    for _ in range(days):
        pred = model.predict(seq.reshape(1, WINDOW_SIZE, 1))
        future.append(pred[0, 0])
        seq = np.append(seq[1:], pred)

    return scaler.inverse_transform(np.array(future).reshape(-1, 1))

st.subheader("🔮 Future Stock Price Forecast")

forecast_days = st.selectbox("Select Forecast Horizon", [1, 5, 10])

last_sequence = scaled_data[-WINDOW_SIZE:]

if model_choice == "SimpleRNN":
    future_prices = forecast_future(simple_rnn, last_sequence, forecast_days)
else:
    future_prices = forecast_future(lstm_model, last_sequence, forecast_days)

forecast_df = pd.DataFrame(
    future_prices,
    columns=["Predicted Price (USD)"]
)

st.dataframe(forecast_df)

# --------------------------------------------------
# Forecast Plot
# --------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.plot(forecast_df["Predicted Price (USD)"], marker='o')
ax2.set_xlabel("Days Ahead")
ax2.set_ylabel("Stock Price (USD)")
ax2.set_title("Future Price Prediction")

st.pyplot(fig2)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "📌 **Project:** Tesla Stock Price Prediction using SimpleRNN & LSTM  \n"
    "📌 **Domain:** Financial Services  \n"
    "📌 **Deployment:** Streamlit Frontend"
)

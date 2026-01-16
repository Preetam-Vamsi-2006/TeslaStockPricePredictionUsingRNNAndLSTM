# 🚗 Tesla Stock Price Prediction using Deep Learning

This project focuses on predicting **Tesla stock prices** using **time-series deep learning models**, specifically **SimpleRNN and LSTM**.  
The goal is to analyze historical stock data and forecast future closing prices for different forecast horizons.

The project also includes a **Streamlit-based frontend dashboard** for interactive visualization and deployment.

---

## 📌 Project Overview

- **Domain:** Financial Services  
- **Company:** Tesla  
- **Problem Type:** Time-Series Forecasting (Regression)  
- **Models Used:** SimpleRNN, LSTM  
- **Target Variable:** Adjusted Closing Price (`Adj Close`)  
- **Forecast Horizons:** 1 Day, 5 Days, 10 Days  

---

## 🎯 Objectives

- Perform exploratory data analysis (EDA) on Tesla stock data  
- Preprocess and scale time-series data  
- Build and compare **SimpleRNN** and **LSTM** models  
- Predict future stock prices using multi-step forecasting  
- Evaluate models using **MSE and RMSE**  
- Deploy the trained models using **Streamlit**  

---

## 📂 Dataset

- **Source:** Yahoo Finance (Tesla Stock Data)
- **Features:**
  - Date
  - Open
  - High
  - Low
  - Close
  - Adj Close
  - Volume

> The **Adjusted Close price** is used as the target variable as it accounts for stock splits and dividends.

---

## 🧠 Methodology

### 1️⃣ Data Preprocessing
- Converted `Date` column to datetime format and set it as index
- Handled missing values using forward fill
- Selected `Adj Close` as the target feature
- Applied **MinMaxScaler** for normalization
- Created sliding window sequences (60 days)

### 2️⃣ Model Development
- Built models using **Keras Sequential API**
- **SimpleRNN Model**
  - SimpleRNN layer
  - Dropout layer
  - Dense output layer
- **LSTM Model**
  - Two stacked LSTM layers
  - Dropout layers
  - Dense output layer

### 3️⃣ Model Training
- Loss Function: Mean Squared Error (MSE)
- Optimizer: Adam
- Used **EarlyStopping** to prevent overfitting
- Used **ModelCheckpoint** to save best models

### 4️⃣ Model Evaluation
- Evaluated on unseen test data
- Metrics used:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
- Visualized **Actual vs Predicted prices**

---

## 📊 Results

| Model | RMSE (USD) |
|------|------------|
| SimpleRNN | 27.03 |
| LSTM | 27.67 |

**Key Observations:**
- Both models capture the overall stock price trend
- SimpleRNN slightly outperformed LSTM in this setup
- Predictions are smoother than actual prices, especially during sudden market spikes
- Performance is reasonable given Tesla’s high volatility

---

## 🌐 Streamlit Deployment

An interactive **Streamlit dashboard** was developed to:

- Visualize actual vs predicted stock prices
- Compare SimpleRNN and LSTM predictions
- Forecast future prices for:
  - 1 day
  - 5 days
  - 10 days

### Run the app locally:
```bash
streamlit run app.py

# 📈 Time Series Demand Forecasting using XGBoost and LSTM

This project focuses on simulating and forecasting retail product demand over a 4-year period using synthetic data. The goal is to compare the effectiveness of traditional machine learning (XGBoost) and deep learning (LSTM) models in predicting future demand patterns.

---

## 🔧 Project Structure

## 📊 Features Simulated

- Seasonal demand trends (sinusoidal pattern)
- Random holidays affecting demand
- Weather conditions (Sunny, Cloudy, Rainy)
- Product launch boosts
- Noise to mimic real-world randomness

---

## 🧠 Models Used

### 🔹 XGBoost
- Uses `DayOfYear` as the input feature.
- Suitable for tabular regression tasks.
- Offers fast training and explainability.

### 🔹 LSTM (Long Short-Term Memory)
- Neural network that learns sequential patterns.
- Takes `DayOfYear`, weather, and holiday indicators as input.
- Scaled and reshaped data for time-series compatibility.

---

## 📈 Evaluation Metrics

- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Visualizations of actual vs predicted demand**

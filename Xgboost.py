import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# Load dataset
data = pd.read_csv("synthetic_demand_data.csv")
data["Date"] = pd.to_datetime(data["Date"])
data["DayOfYear"] = data["Date"].dt.dayofyear

features = data[["DayOfYear"]]
target = data["Demand"]

# Split data: 3 years train, 1 year test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, shuffle=False)

# Initialize and train XGBoost
regressor = xgb.XGBRegressor(n_estimators=100, max_depth=5, objective="reg:squarederror", random_state=0)
regressor.fit(X_train, y_train)

# Predictions and metrics
y_pred = regressor.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.2f}")

# On-demand prediction
query_date = input("Enter a date (YYYY-MM-DD): ")
query_date = pd.to_datetime(query_date)
query_day = query_date.dayofyear
prediction = regressor.predict(np.array([[query_day]]))

if query_date in data["Date"].values:
    actual = data[data["Date"] == query_date]["Demand"].values[0]
    print(f"Actual: {actual:.2f}")
else:
    print("No actual value available.")

print(f"Predicted Demand: {prediction[0]:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.plot(data["Date"].iloc[len(X_train):], y_test, label="Actual", color="green")
plt.plot(data["Date"].iloc[len(X_train):], y_pred, label="XGBoost Forecast", color="red")
plt.title("XGBoost Forecast vs Actual Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()
plt.show()

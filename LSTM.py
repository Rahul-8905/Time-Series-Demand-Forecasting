import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("demand_data_with_random_holidays_and_weather.csv")
data["Date"] = pd.to_datetime(data["Date"])
data["DayOfYear"] = data["Date"].dt.dayofyear

# Encode categorical data (Weather Condition and Holiday)
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

# Assuming your data is already loaded into the `data` dataframe
encoder = OneHotEncoder(sparse_output=False)

# Combine the columns for encoding
columns_to_encode = data[["WeatherCondition", "Holiday"]]
encoded_data = encoder.fit_transform(columns_to_encode)

# Get the names of the encoded features
encoded_columns = encoder.get_feature_names_out(columns_to_encode.columns)

# Create a new dataframe for the encoded columns
encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

# Add the encoded columns back to the original dataframe
data = pd.concat([data, encoded_df], axis=1)

# Prepare the features and target variable
X = data[["DayOfYear"] + list(encoded_columns)]
y = data["Demand"]

# Train-test split (using first 3 years for training, and last year for testing)
train_size = int(len(X) * 0.75)  # Use the first 75% (3 years) for training
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1))

# Reshape for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=1)

# Make predictions
predictions_scaled = model.predict(X_test_scaled)
predictions = scaler_y.inverse_transform(predictions_scaled)  # Inverse transform to get actual values
y_test_actual = scaler_y.inverse_transform(y_test_scaled)  # Inverse transform test data to actual values


mse = mean_squared_error(y_test_actual, predictions)
print(f"Mean Squared Error: {mse:.2f}")
mae = mean_absolute_error(y_test_actual, predictions)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
mape = mean_absolute_percentage_error(y_test_actual,predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}")

# Plot the predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(data["Date"].iloc[train_size:], y_test_actual, label="Actual Demand", color="green")
plt.plot(data["Date"].iloc[train_size:], predictions, label="LSTM Predictions", color="red")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("Actual vs Predicted Demand (LSTM)")
plt.legend()
plt.show()

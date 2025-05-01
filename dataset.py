import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate a sequence of dates
total_days = 1460
dates = pd.date_range(start="2020-01-01", periods=total_days)

# Create base demand signal with seasonal trends and noise
base_value = 300
season_effect = 100
noise = np.random.normal(0, 10, total_days)
seasonal_wave = np.sin(2 * np.pi * (np.arange(total_days) % 365) / 365)
demand_values = base_value + season_effect * seasonal_wave + noise

# Assemble DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Demand": demand_values,
    "DayOfYear": dates.dayofyear
})

# Assign random weather types
df["Weather"] = np.random.choice(["Sunny", "Cloudy", "Rainy"], size=total_days, p=[0.7, 0.2, 0.1])

# Randomly mark days with a product launch
df["LaunchDay"] = np.random.choice([0, 1], size=total_days, p=[0.95, 0.05])

# Random holiday generation across years
holiday_list = []
for year in range(2020, 2024):
    for _ in range(10):
        random_date = datetime(year, 1, 1) + timedelta(days=np.random.randint(0, 365))
        holiday_list.append(random_date)

df["HolidayFlag"] = df["Date"].isin(holiday_list).astype(int)

# Add feature effects on demand
df["Demand"] += df["HolidayFlag"] * 50
df["Demand"] -= (df["Weather"] == "Rainy") * 30
df["Demand"] += df["LaunchDay"] * 100

# Save to CSV
df.to_csv("synthetic_demand_data.csv", index=False)
print("Synthetic dataset saved as 'synthetic_demand_data.csv'")

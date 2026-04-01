import pandas as pd
import numpy as np
import requests

print("Step 1: Loading IEX data...")

df = pd.read_csv("clean_iex_data.csv")
df['datetime'] = pd.to_datetime(df['datetime'])

# ---------------- WEATHER ----------------
print("Step 2: Fetching weather data...")

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 17.3850,
    "longitude": 78.4867,
    "start_date": "2026-01-01",
    "end_date": "2026-01-31",
    "hourly": [
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloudcover",
        "windspeed_10m",
        "surface_pressure",
        "shortwave_radiation"
    ],
    "timezone": "Asia/Kolkata"
}

response = requests.get(url, params=params)
data = response.json()

weather_df = pd.DataFrame(data['hourly'])

weather_df['time'] = pd.to_datetime(weather_df['time'])

weather_df = weather_df.rename(columns={
    "time": "datetime",
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "precipitation": "rain",
    "cloudcover": "cloud",
    "windspeed_10m": "wind",
    "surface_pressure": "pressure",
    "shortwave_radiation": "solar"
})

# Save raw weather
weather_df.to_csv("weather_data.csv", index=False)

# ---------------- RESAMPLE ----------------
print("Step 3: Resampling to 15-min...")

weather_df = weather_df.set_index('datetime')
weather_15min = weather_df.resample('15min').interpolate()
weather_15min = weather_15min.reset_index()

# ---------------- ALIGN TIME ----------------
print("Step 4: Aligning time range...")

start = df['datetime'].min()
end = df['datetime'].max()

full_range = pd.date_range(start=start, end=end, freq='15min')
full_df = pd.DataFrame({'datetime': full_range})

# Merge IEX
merged_df = pd.merge(full_df, df, on='datetime', how='left')

# Merge weather
merged_df = pd.merge(merged_df, weather_15min, on='datetime', how='left')

# Fill missing
merged_df = merged_df.ffill().bfill()

# ---------------- SAVE ----------------
merged_df.to_csv("merged_data.csv", index=False)

print("SUCCESS: merged_data.csv created")
print("Shape:", merged_df.shape)
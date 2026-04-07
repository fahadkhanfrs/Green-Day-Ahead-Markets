import pandas as pd
import numpy as np

df = pd.read_excel("final_merged_data.xlsx")

df['start_time'] = df['Time Period'].str.split('-').str[0]

df['datetime'] = pd.to_datetime(
    df['Delivery Date'] + ' ' + df['start_time'],
    dayfirst=True
)

df = df.sort_values('datetime')

df = df.rename(columns={
    'Price (Rs./MWh)': 'price',
    'Cleared Buy (MW)': 'buy_mw',
    'Cleared Sell (MW)': 'sell_mw',
    'temperature_2m (°C)': 'temp',
    'relative_humidity_2m (%)': 'humidity',
    'rain (mm)': 'rain',
    'cloud_cover (%)': 'cloud',
    'wind_speed_100m (km/h)': 'wind',
    'shortwave_radiation (W/m²)': 'solar'
})


df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

df['price_lag_1'] = df['price'].shift(1)
df['price_lag_4'] = df['price'].shift(4)
df['price_lag_96'] = df['price'].shift(96)
df['solar_hour_interaction'] = df['solar'] * df['hour_sin']
df['price_rolling_mean_4'] = df['price'].shift(1).rolling(4).mean()
df['price_rolling_std_4'] = df['price'].shift(1).rolling(4).std()

df['demand_supply_ratio'] = df['buy_mw'] / (df['sell_mw'] + 1)
df['price_diff'] = df['price'] - df['price_lag_1']

df = df.dropna()

df.to_csv("final_data.csv", index=False)

print("SUCCESS: final_data.csv created")
print("Shape:", df.shape)
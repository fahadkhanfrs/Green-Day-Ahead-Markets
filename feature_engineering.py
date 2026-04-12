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
df['price_rolling_mean_96'] = df['price'].shift(96).rolling(96).mean()
df['price_rolling_std_96'] = df['price'].shift(1).rolling(96).std()
df['price_lag_96'] = df['price'].shift(96)
df['price_lag_192'] = df['price'].shift(192)
df['is_low_solar'] = (df['solar'] < 50).astype(int)
df['price_change_96'] = df['price_lag_96'] - df['price_lag_192']
df['lag96_solar'] = df['price_lag_96'] * df['solar']
df['is_spike'] = (df['price'] > df['price_lag_96'] + 1000).astype(int)
df['hour'] = df['datetime'].dt.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)
df['weekday'] = df['datetime'].dt.weekday
df['is_peak_hour'] = df['hour'].isin([18,19,20,21]).astype(int)

df['solar_hour_interaction'] = df['solar'] * df['hour_sin']

df['demand_supply_ratio'] = df['buy_mw'] / (df['sell_mw'] + 1)

df = df.dropna()

df.to_csv("final_data.csv", index=False)
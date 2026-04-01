import pandas as pd
import numpy as np

df = pd.read_csv("merged_data.csv")

df['datetime'] = pd.to_datetime(df['datetime'])

# Time features
df['hour'] = df['datetime'].dt.hour
df['weekday'] = df['datetime'].dt.weekday

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)

# Lag features
df['price_lag_1'] = df['price'].shift(1)
df['price_lag_4'] = df['price'].shift(4)
df['price_lag_96'] = df['price'].shift(96)

df['net_demand'] = df['buy_mw'] - df['sell_mw']
df['demand_supply_ratio'] = df['buy_mw'] / (df['sell_mw'] + 1)

# Drop NaNs
df = df.dropna()

df.to_csv("final_data.csv", index=False)
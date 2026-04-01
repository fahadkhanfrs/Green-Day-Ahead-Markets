import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("final_data.csv")

features = [
    'price_lag_1',
    'price_lag_4',
    'price_lag_96',
    'hour',
    'hour_sin',
    'hour_cos',
    'solar',
    'cloud',
    'temp',
    'demand_supply_ratio'
]

X = df[features]
y = df['price']

# IMPORTANT: no shuffle (time series)
split = int(len(df)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = LinearRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("MSE:", mean_squared_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

df = pd.read_csv("final_data.csv")

features = [
    'price_lag_1',
    'price_lag_4',
    'price_lag_96',
    'hour_sin',
    'hour_cos',
    'solar',
    'cloud',
    'temp',
    'hour',
    'demand_supply_ratio'
]

X = df[features]
y = df['price']

split = int(len(df)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBRegressor(n_estimators=200, max_depth=6)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
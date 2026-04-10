import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("final_data.csv")

features = [
    'price_lag_1',
    'price_lag_4',
    'price_lag_96',
    'price_rolling_mean_4',
    'price_rolling_std_4',
    'solar_hour_interaction',
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

split = int(len(df)*0.5)

X_train, X_test = X[:split], X[split:]
y_test = y[split:]
y_train = y[:split]

model = XGBRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

plt.figure(figsize=(12,6))

plt.plot(y_test.values[:300], label='Actual')
plt.plot(preds[:300], label='Predicted')

plt.legend()
plt.title("Actual vs Predicted MCP")
plt.xlabel("Time")
plt.ylabel("Price")

plt.show()

errors = y_test - preds

plt.hist(errors, bins=50)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")

plt.show()

plot_importance(model)
plt.show()
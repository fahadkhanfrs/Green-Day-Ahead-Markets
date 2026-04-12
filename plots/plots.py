import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor

model = Pipeline([
    ('scaler', StandardScaler()),
    ('huber', HuberRegressor())
])

df = pd.read_csv("final_data.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

features = [
    'is_spike',
    'is_low_solar',
    'price_change_96',
    'lag96_solar',
    'weekday',
    'is_peak_hour',
    'price_lag_96',
    'price_lag_192',
    'price_rolling_mean_96',
    'price_rolling_std_96',
    'solar_hour_interaction',
    'hour_sin',
    'hour_cos',
    'solar',
    'cloud',
    'temp',
    'demand_supply_ratio'
]

X = df[features]
y = df['price']
train_df = df.iloc[:int(len(df)*0.6)]
test_df = df.iloc[int(len(df)*0.8):]

X_train = train_df[features]
y_train = train_df['price']

X_test = test_df[features]
y_test = test_df['price']

model.fit(X_train, y_train)

preds = model.predict(X_test)

residuals = y_test - preds

plt.figure(figsize=(12,6))
plt.plot(y_test.values[:300], label='Actual')
plt.plot(preds[:300], label='Predicted')
plt.legend()
plt.title("Actual vs Predicted MCP")
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(preds, residuals, alpha=0.3)
plt.axhline(0, linestyle='--')
plt.title("Residual vs Prediction")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.show()

plt.figure(figsize=(12,5))
plt.plot(residuals.values[:500])
plt.axhline(0, linestyle='--')
plt.title("Residual over Time")
plt.xlabel("Time")
plt.ylabel("Residual")
plt.show()

plt.figure(figsize=(8,5))
plt.hist(residuals, bins=50)
plt.title("Residual Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(8,5))
plt.scatter(X_test['solar'], residuals, alpha=0.3)
plt.axhline(0, linestyle='--')
plt.title("Residual vs Solar")
plt.xlabel("Solar Radiation")
plt.ylabel("Residual")
plt.show()

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

print("MAE:", mae)
print("R2:", r2)
print("MAPE:", mape)

coef = model.named_steps['huber'].coef_

importance_df = pd.DataFrame({
    'feature': features,
    'importance': coef
})

importance_df['abs_importance'] = importance_df['importance'].abs()
importance_df = importance_df.sort_values(by='abs_importance', ascending=False)

print("\nFeature Importance:\n")
print(importance_df[['feature', 'importance']])

plt.figure(figsize=(8,6))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.title("Feature Importance (Huber Regression)")
plt.xlabel("Coefficient Value")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
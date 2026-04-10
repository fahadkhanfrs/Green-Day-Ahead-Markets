from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pandas as pd

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

df= df.sort_values('datetime')

train_df = df[df['datetime'] < '2023-01-01']
test_df = df[df['datetime'] > '2023-06-01']

X_train = train_df[features]
y_train = train_df['price']

X_test = test_df[features]
y_test = test_df['price']

model = XGBRegressor(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1,
    reg_lambda=2,
    random_state=42
)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))
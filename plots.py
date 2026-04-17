import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import HuberRegressor, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score, classification_report

df = pd.read_csv("final_data.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.sort_values('datetime')

features = [
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

train_df = df.iloc[:int(len(df)*0.6)]
test_df  = df.iloc[int(len(df)*0.8):]

X_train = train_df[features]
y_train = train_df['price']

X_test = test_df[features]
y_test = test_df['price']

reg_model = Pipeline([
    ('scaler', StandardScaler()),
    ('huber', HuberRegressor(max_iter=1000))
])
reg_model.fit(X_train, y_train)

clf_model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])
clf_model.fit(X_train, train_df['is_spike'])

base_preds = reg_model.predict(X_test)
spike_probs = clf_model.predict_proba(X_test)[:, 1]

# final_preds = []

# for i in range(len(base_preds)):
#     pred = base_preds[i]
#     prob = spike_probs[i]

#     if prob > 0.3:
#         pred = pred * (1 + 0.3 * prob)

#     final_preds.append(pred)

# final_preds = np.array(final_preds)

# mae_base = mean_absolute_error(y_test, base_preds)
# mae_hybrid = mean_absolute_error(y_test, final_preds)

# r2_base = r2_score(y_test, base_preds)
# r2_hybrid = r2_score(y_test, final_preds)

# mape_base = np.mean(np.abs((y_test - base_preds) / y_test)) * 100
# mape_hybrid = np.mean(np.abs((y_test - final_preds) / y_test)) * 100

# print("\nBASE MODEL:")
# print("MAE:", mae_base)
# print("R2:", r2_base)
# print("MAPE:", mape_base)

# print("\nHYBRID MODEL:")
# print("MAE:", mae_hybrid)
# print("R2:", r2_hybrid)
# print("MAPE:", mape_hybrid)

# print("\nSPIKE CLASSIFICATION:")
# print(classification_report(test_df['is_spike'], clf_model.predict(X_test)))

# res_base = y_test - base_preds
# res_hybrid = y_test - final_preds

# plt.figure(figsize=(12,6))
# plt.plot(y_test.values[:300], label='Actual')
# plt.plot(base_preds[:300], label='Base')
# plt.plot(final_preds[:300], label='Hybrid')
# plt.legend()
# plt.title("Actual vs Predicted (Base vs Hybrid)")
# plt.show()

# plt.figure(figsize=(8,5))
# plt.scatter(base_preds, res_base, alpha=0.3, label='Base')
# plt.scatter(final_preds, res_hybrid, alpha=0.3, label='Hybrid')
# plt.axhline(0, linestyle='--')
# plt.legend()
# plt.title("Residual vs Prediction")
# plt.show()

# plt.figure(figsize=(12,5))
# plt.plot(res_base.values[:500], label='Base')
# plt.plot(res_hybrid.values[:500], label='Hybrid')
# plt.axhline(0, linestyle='--')
# plt.legend()
# plt.title("Residual over Time")
# plt.show()

# plt.figure(figsize=(8,5))
# plt.hist(res_base, bins=50, alpha=0.6, label='Base')
# plt.hist(res_hybrid, bins=50, alpha=0.6, label='Hybrid')
# plt.legend()
# plt.title("Residual Distribution")
# plt.show()

# plt.figure(figsize=(8,5))
# plt.scatter(X_test['solar'], res_base, alpha=0.3, label='Base')
# plt.scatter(X_test['solar'], res_hybrid, alpha=0.3, label='Hybrid')
# plt.axhline(0, linestyle='--')
# plt.legend()
# plt.title("Residual vs Solar")
# plt.show()

# coef = reg_model.named_steps['huber'].coef_

# importance_df = pd.DataFrame({
#     'feature': features,
#     'importance': coef
# })

# importance_df['abs'] = importance_df['importance'].abs()
# importance_df = importance_df.sort_values(by='abs', ascending=False)

# plt.figure(figsize=(8,6))
# plt.barh(importance_df['feature'], importance_df['importance'])
# plt.gca().invert_yaxis()
# plt.title("Feature Importance (Huber)")
# plt.show()

# thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# results = []

# for threshold in thresholds:
#     adjusted_preds = []
#     for i in range(len(base_preds)):
#         pred = base_preds[i]
#         prob = spike_probs[i]

#         if prob > threshold:
#             pred = pred * (1 + 0.3 * prob)

#         adjusted_preds.append(pred)

#     adjusted_preds = np.array(adjusted_preds)
#     mae = mean_absolute_error(y_test, adjusted_preds)
#     r2 = r2_score(y_test, adjusted_preds)
#     mape = np.mean(np.abs((y_test - adjusted_preds) / y_test)) * 100

#     results.append((threshold, mae, r2, mape))

# print("\nThreshold Testing Results:")
# for threshold, mae, r2, mape in results:
#     print(f"Threshold: {threshold}, MAE: {mae:.4f}, R2: {r2:.4f}, MAPE: {mape:.4f}%")

# print("\nSpike probability distribution:")
# print(pd.Series(spike_probs).describe())
# for threshold in thresholds:
#     print(f"Adjusted points at threshold {threshold}: {np.sum(spike_probs > threshold)}")

spike_train = train_df[train_df['is_spike'] == 1]

X_spike = spike_train[features]
y_spike = spike_train['delta']

delta_model = Pipeline([
    ('scaler', StandardScaler()),
    ('huber', HuberRegressor())
])

delta_model.fit(X_spike, y_spike)

base = reg_model.predict(X_test)
prob = clf_model.predict_proba(X_test)[:,1]

final = []

for i in range(len(X_test)):
    pred = base[i]
    
    if prob[i] > 0.3:
        delta_pred = delta_model.predict(X_test.iloc[i:i+1])[0]
        pred = pred + delta_pred
    
    final.append(pred)

final = np.array(final)

mae_delta = mean_absolute_error(y_test, final)
r2_delta = r2_score(y_test, final)

print("\nDELTA HYBRID MODEL:")
print("MAE:", mae_delta)
print("R2:", r2_delta)
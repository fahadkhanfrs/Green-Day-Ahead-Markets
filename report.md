Residual vs solar

High spread when solar is close to 0
tighter when solar is high

Low solar (night / low renewable) → unpredictable market → larger errors

model works for normal conditions, fails on extreme events

Underpredicting high prices
Slight overprediction in some mid ranges
Overall unbiased, but variance increases with price

struggles without short term immediate past features

MAE: 686.3816708474038
R2: 0.8109971670523476
MAPE: 23.711670668077282

Feature Importance:

                   feature   importance
0             price_lag_96  1701.575487
1            price_lag_192   722.399385
7                    solar  -302.126442
9                     temp   213.818699
10     demand_supply_ratio  -124.802275
4   solar_hour_interaction   110.377030
5                 hour_sin   -84.228123
3     price_rolling_std_96    76.380121
8                    cloud   -73.726802
2    price_rolling_mean_96    18.057237
6                 hour_cos   -11.912679

Weather now matters more:
solar negative
temp positive

model is forced to use exogenous variables

weekly features made the performance worse, added more noice instead of contributing

added is_low_solar after residual analysis showed high spread

used huber regression (robust to spikes)

model performance improved after removing leakage and adding robust regression, remaining error is concentrated in spike regimes, so moving on to toward regime based modeling
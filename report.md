Residual vs solar

High spread when solar is close to 0
tighter when solar is high

Underpredicting high prices
Slight overprediction in some mid ranges
Overall unbiased, but variance increases with price

struggles without short term immediate past features

MAE: 468.9677035668098
R2: 0.8736344827482706
MAPE: 15.25164590824688

Feature Importance:

                   feature   importance
6             price_lag_96  1627.858978
7            price_lag_192  1263.222647
0                 is_spike   677.054183
2          price_change_96   626.558844
4                  weekday   -95.164657
3              lag96_solar   -80.197383
13                   solar    49.166776
16     demand_supply_ratio   -37.607528
15                    temp    35.127036
14                   cloud   -20.373261
12                hour_cos    17.633412
10  solar_hour_interaction    16.468350
9     price_rolling_std_96   -11.010229
8    price_rolling_mean_96     9.779000
11                hour_sin    -9.279413
1             is_low_solar     6.330981
5             is_peak_hour     0.661166


weekly features made the performance worse, added more noice instead of contributing

added is_low_solar because residual analysis showed high spread when solar was low

used huber regression (robust to spikes)

model performance improved after removing leakage and adding robust regression, remaining error is concentrated in spike regimes, so moving on to toward regime based modeling

Aggressive spike detection degrades performance due to false positives, while conservative spike correction improves prediction stability.

Simple models fail to capture spike regimes. Introducing a spike aware hybrid approach improves prediction performance under realistic constraints.
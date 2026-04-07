Raw cleared buy/sell volumes showed weak correlation with MCP. However, engineered imbalance features significantly improved predictive performance.

Your data is:

GDAM (Green market)
Supply dominated by renewables (solar/wind)

So:

Weather (solar) already captures supply variation

That’s why:

solar → strong negative (-0.34)
sell_mw → stronger than buy_mw
ratio works better than difference

-----------------------

Key insight (very important)

Your data is showing:

Price ↓ when renewable supply ↑ (solar high)

This matches real-world:

Oversupply → prices fall (even negative sometimes)

So your model is capturing correct physics

Even after removing the immediate autoregressive term (lag-1), the model retains high predictive power, indicating strong temporal structure in GDAM prices.

Price is NOT driven primarily by weather - ablation study result.
Weather improves model only when combined with lag features

GDAM MCP is primarily driven by autoregressive temporal dynamics, with weather variables contributing only marginal predictive improvement.
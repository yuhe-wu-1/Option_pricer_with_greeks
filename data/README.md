# Data

Market data is **not committed** to this repository: OptionMetrics data (via WRDS) is licensed and may not be
redistributed, and Yahoo Finance terms restrict redistribution. Place the files below in these folders to run
analysis scripts 01 to 04. Scripts 05 and 06 and the unit tests need no data.

## data/optionmetrics/  (WRDS -> OptionMetrics -> IvyDB US)

| File | WRDS table | Query |
|---|---|---|
| `spx_options_2024-08-29_30.csv` | Option Prices | SPX (secid 108105), 2024-08-29 to 2024-08-30, all Option Price and Option Information variables |
| `spx_security_prices.csv` | Security Prices | secid 108105, same dates |
| `zero_coupon_yield.csv` | Zero Coupon Yield Curve | same dates |
| `spx_index_dividend_yield.csv` | Index Dividend Yield | secid 108105, same dates |

If you download the full Option Prices universe, extract SPX with:
```python
import pandas as pd
parts = [c[c.secid == 108105] for c in pd.read_csv("option_prices.csv", chunksize=500_000, low_memory=False)]
pd.concat(parts).to_csv("data/optionmetrics/spx_options_2024-08-29_30.csv", index=False)
```

## data/yahoo/

`SPX_european_clean.csv`: SPX option chain snapshot from Yahoo Finance, 2026-08-27 (used by `03_yahoo_snapshot.py`).

## data/history/ and data/model_grid/

`daily_prices.csv`: SPX daily closes. `options_timeseries.csv` is generated from it by
`scripts/preprocessor_timeseries.py --input data/history/daily_prices.csv`; `priced_bs.csv`, `priced_bin.csv`
and `priced_mc.csv` are the three models' prices on that grid (used by `04_model_consistency.py`).

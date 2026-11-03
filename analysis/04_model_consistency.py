"""
Consistency of the three pricing methods on a historical-volatility grid (SPX closes 2023-01 to 2025-08,
20-day rolling vol, K/S 0.8-1.2, maturities 30/90/180 days; built by scripts/preprocessor_timeseries.py).

Checks:
  1. Binomial tree vs Black-Scholes.
  2. Monte Carlo vs Black-Scholes, in standard errors.

Output: printed summary.
"""
import numpy as np
import pandas as pd
from common import DATA

g = DATA / "model_grid"
bsp, tree, mc = (pd.read_csv(g / f) for f in ("priced_bs.csv", "priced_bin.csv", "priced_mc.csv"))
grid = pd.read_csv(g / "options_timeseries.csv")
print(f"{len(grid)} options, {grid.date.min()} to {grid.date.max()}, K/S {sorted((grid.K/grid.S).round(2).unique())}, "
      f"T days {sorted((grid['T']*365).round().astype(int).unique())}")
d = (tree.price - bsp.price).abs()
print(f"tree vs BS: median relative diff {100*np.median(d/bsp.price.clip(lower=1e-8)):.3f}%, max |diff| {d.max():.4f}")
ok = (mc.se > 0) & (bsp.price > 0.01); z = ((mc.price - bsp.price) / mc.se)[ok]
print(f"MC vs BS on {ok.sum()} options with SE > 0: {100*np.mean(np.abs(z) < 2):.1f}% |z| < 2, max |z| {np.abs(z).max():.2f}")
print(f"  ({(~ok).sum()} far out-of-the-money options excluded: every path pays zero, so SE = 0 and z is undefined)")

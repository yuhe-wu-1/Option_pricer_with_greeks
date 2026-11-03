"""
Monte Carlo Greeks vs analytic Black-Scholes Greeks, for calls and puts across moneyness and maturity.

Checks:
  1. Pathwise delta and vega, and likelihood-ratio delta and gamma, in standard errors.
  2. Pathwise gamma, which is zero on every path.
  3. Standard error of likelihood-ratio vs pathwise delta.

Output: results/05_mc_greeks.csv.
"""
import numpy as np
import pandas as pd
from common import RESULTS, bs
from pricer.mc_greeks import mc_greeks_likelihood_ratio, mc_greeks_pathwise

N = 200_000
CASES = [  # label, S, K, r, q, sigma, T, is_call
    ("ATM call 1y", 100, 100, 0.05, 0.00, 0.20, 1.00, True),
    ("OTM call 3m", 100, 110, 0.05, 0.00, 0.20, 0.25, True),
    ("ITM call 6m", 100, 90, 0.05, 0.02, 0.25, 0.50, True),
    ("ATM put 1y", 100, 100, 0.05, 0.00, 0.20, 1.00, False),
    ("OTM put 3m", 100, 90, 0.05, 0.00, 0.30, 0.25, False),
    ("SPX-like put 22d", 5600, 5300, 0.052, 0.0098, 0.16, 22 / 365, False),
]

rows = []
for i, (label, S, K, r, q, v, T, c) in enumerate(CASES):
    exact = dict(delta=(bs.delta_call if c else bs.delta_put)(S, K, r, v, T, q),
                 vega=bs.vega(S, K, r, v, T, q), gamma=bs.gamma(S, K, r, v, T, q))
    pw = mc_greeks_pathwise(S, K, r, q, v, T, c, N, seed=100 + i)
    lr = mc_greeks_likelihood_ratio(S, K, r, q, v, T, c, N, seed=200 + i)
    for method, greek, (est, se) in [("pathwise", "delta", pw["delta"]), ("pathwise", "vega", pw["vega"]),
                                     ("pathwise", "gamma", pw["gamma"]), ("likelihood ratio", "delta", lr["delta"]),
                                     ("likelihood ratio", "gamma", lr["gamma"])]:
        z = (est - exact[greek]) / se if se > 0 else np.nan
        rows.append(dict(case=label, method=method, greek=greek, analytic=exact[greek], mc=est, se=se, z=z))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "05_mc_greeks.csv", index=False)
pd.set_option("display.width", 200)
print(R.round(5).to_string(index=False))

ok = R.z.notna()
print(f"\n|z| < 2 for {100*np.mean(np.abs(R.z[ok]) < 2):.0f}% of the {ok.sum()} estimates with a standard error "
      f"(max |z| {np.abs(R.z[ok]).max():.2f})")
pg = R[(R.method == "pathwise") & (R.greek == "gamma")]
print(f"Pathwise gamma is exactly 0 in every case, against analytic values from {pg.analytic.min():.5f} to {pg.analytic.max():.5f}")
d = R[R.greek == "delta"].pivot(index="case", columns="method", values="se")
print(f"Delta standard error, likelihood ratio / pathwise: {(d['likelihood ratio']/d['pathwise']).min():.1f}x to "
      f"{(d['likelihood ratio']/d['pathwise']).max():.1f}x (pathwise is less noisy when the payoff allows it)")


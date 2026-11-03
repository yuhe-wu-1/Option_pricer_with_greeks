"""
Variance reduction in Monte Carlo pricing by moneyness (calls, S = 100, sigma = 20%, T = 1).

Checks:
  1. Control variate (discounted terminal price): measured variance reduction vs the theoretical 1 / (1 - rho^2).
  2. Antithetic variates, alone and combined with the control variate.

Output: results/06_variance_reduction.csv.
"""
import numpy as np
import pandas as pd
from common import RESULTS, bs
from pricer.monte_carlo import mc_euro_vectorized

S, r, q, v, T = 100.0, 0.05, 0.0, 0.20, 1.0
N, RUNS = 20_000, 1000
CASES = [("deep ITM call", 70), ("ITM call", 90), ("ATM call", 100), ("OTM call", 110), ("deep OTM call", 130)]

rows = []
for label, K in CASES:
    # correlation from one large sample
    z = np.random.default_rng(1).standard_normal(1_000_000)
    ST = S * np.exp((r - q - 0.5 * v * v) * T + v * np.sqrt(T) * z)
    rho = np.corrcoef(np.exp(-r * T) * np.maximum(ST - K, 0), np.exp(-r * T) * ST)[0, 1]
    est = {m: [] for m in ("plain", "antithetic", "control variate", "both")}
    for seed in range(RUNS):
        est["plain"].append(mc_euro_vectorized(S, K, r, q, v, T, True, N, seed, antithetic=False, control_variate=False)[0])
        est["antithetic"].append(mc_euro_vectorized(S, K, r, q, v, T, True, N, seed, antithetic=True, control_variate=False)[0])
        est["control variate"].append(mc_euro_vectorized(S, K, r, q, v, T, True, N, seed, antithetic=False, control_variate=True)[0])
        est["both"].append(mc_euro_vectorized(S, K, r, q, v, T, True, N, seed, antithetic=True, control_variate=True)[0])
    var = {m: np.var(x, ddof=1) for m, x in est.items()}
    rows.append(dict(case=label, K=K, bs_price=bs.call_price(S, K, r, v, T, q), rho=rho,
                     theory_cv=1 / (1 - rho ** 2), measured_cv=var["plain"] / var["control variate"],
                     measured_antithetic=var["plain"] / var["antithetic"], measured_both=var["plain"] / var["both"]))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "06_variance_reduction.csv", index=False)
pd.set_option("display.width", 200)
print(f"S = {S}, r = {r}, sigma = {v}, T = {T}; {N:,} paths per run, variance measured across {RUNS} independent runs")
print(R.round(3).to_string(index=False))
print("\nmeasured_cv / theory_cv: " + ", ".join(f"{c} {m/t:.2f}" for c, m, t in zip(R.case, R.measured_cv, R.theory_cv)))

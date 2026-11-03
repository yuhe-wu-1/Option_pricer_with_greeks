"""
Baseline validation against OptionMetrics SPX quotes (2024-08-29, AM-settled expiries at 22, 50 and 78 days;
out-of-the-money options, K/F 0.80-1.10). Forward and discount factor are implied from put-call parity;
time to expiry is calendar days / 365 (refined in 02).

Checks:
  1. Parity-implied forward vs OptionMetrics' zero curve and dividend yield.
  2. Model implied vols vs OptionMetrics; call vs put implied vol at the same strike.
  3. Model Greeks vs OptionMetrics Greeks, both using OptionMetrics' implied vols.
  4. Pricing error by moneyness for a single ATM vol, the best single vol, and a quadratic smile
     tested on held-out strikes.
  5. The 2024-08-29 smile applied to 2024-08-30 quotes.

Output: results/01_greeks_vs_optionmetrics.csv, results/01_pricing_errors.csv.
"""
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from common import (RESULTS, bs, bs_prices, error_table, implied_vols, load_optionmetrics, om_quotes,
                    out_of_the_money, parity_forward, time_to_expiry, vegas, zero_rate)

opts, spot, zc, div = load_optionmetrics()
DATE, NEXT = "2024-08-29", "2024-08-30"
EXPS = ["2024-09-20", "2024-10-18", "2024-11-15"]
S, q = spot[DATE], div[DATE]
print(f"{DATE}: SPX close {S}, OptionMetrics dividend yield {100*q:.3f}%")

rows, greek_rows, smiles = [], [], {}
for exd in EXPS:
    s = om_quotes(opts, DATE, exd)
    T = time_to_expiry(DATE, exd, True, "calendar"); days = round(T * 365)
    F, DF = parity_forward(s, S); r_ = -math.log(DF) / T
    F_om = S * math.exp((zero_rate(zc, DATE, days) - q) * T)
    o = out_of_the_money(s, F); o["iv"], ok = implied_vols(o, F, DF, T)
    o = o[ok & o.impl_volatility.notna()].reset_index(drop=True)

    # call vs put IV at the same strike, near the money
    c = s[s.cp_flag == "C"].set_index("K").mid; p = s[s.cp_flag == "P"].set_index("K").mid
    b = pd.concat([c, p], axis=1, keys=["C", "P"]).dropna(); b = b[(b.index > 0.97 * F) & (b.index < 1.03 * F)]
    S_ = F * DF
    gap = [100 * (bs.implied_vol(S_, K, r_, T, C, True)[0] - bs.implied_vol(S_, K, r_, T, P, False)[0]) for K, C, P in zip(b.index, b.C, b.P)]
    print(f"\n### {exd} ({days}d): parity F={F:.2f} vs OptionMetrics-input F={F_om:.2f} ({100*(F/F_om-1):+.3f}%), "
          f"parity r={100*r_:.2f}% vs zero curve {100*zero_rate(zc, DATE, days):.2f}%")
    print(f"   {len(o)} OTM options | your IV - OM IV: median {np.median(100*(o.iv-o.impl_volatility)):+.3f} pts | "
          f"call-put IV gap near ATM: max |.| {np.max(np.abs(gap)):.3f} pts")

    # Greeks vs OptionMetrics, using OptionMetrics' vols
    qi = r_ - math.log(F / S) / T; v = o.impl_volatility.values; cp = o.cp_flag.values
    mine = dict(delta=[(bs.delta_call if c_ == "C" else bs.delta_put)(S, K, r_, s_, T, qi) for K, s_, c_ in zip(o.K, v, cp)],
                gamma=[bs.gamma(S, K, r_, s_, T, qi) for K, s_ in zip(o.K, v)],
                vega=[bs.vega(S, K, r_, s_, T, qi) for K, s_ in zip(o.K, v)],
                theta=[(bs.theta_call if c_ == "C" else bs.theta_put)(S, K, r_, s_, T, qi) for K, s_, c_ in zip(o.K, v, cp)])
    gr = dict(exp=exd, n=len(o), delta_max_abs_diff=np.max(np.abs(np.array(mine["delta"]) - o.delta)))
    for g in ("gamma", "vega", "theta"):
        gr[f"{g}_median_rel_diff_pct"] = 100 * np.median(np.abs(np.array(mine[g]) / o[g] - 1))
    greek_rows.append(gr)

    # volatility inputs
    atm = float(np.interp(0, o.k, o.iv))
    best = minimize_scalar(lambda x: np.sum((bs_prices(o, F, DF, T, x) - o.mid.values) ** 2), bounds=(0.02, 1.5), method="bounded").x
    tr = np.asarray(o.index % 2 == 0); vw = vegas(o, F, DF, T)
    quad = np.polyfit(o.k[tr], o.iv[tr], 2, w=np.sqrt(vw[tr])); smiles[exd] = np.polyfit(o.k, o.iv, 2, w=np.sqrt(vw))
    print(f"   ATM vol {100*atm:.2f}%, best single vol {100*best:.2f}%")
    for label, sig, mask in [("atm_single", np.full(len(o), atm), np.ones(len(o), bool)),
                             ("best_single", np.full(len(o), best), np.ones(len(o), bool)),
                             ("quadratic_heldout", np.polyval(quad, o.k), ~tr)]:
        rows += error_table(o, bs_prices(o, F, DF, T, sig), sig, mask, label, exp=exd)

G = pd.DataFrame(greek_rows); G.to_csv(RESULTS / "01_greeks_vs_optionmetrics.csv", index=False)
print("\n=== Greeks vs OptionMetrics (their vols, calendar-day T) ==="); print(G.round(4).to_string(index=False))
R = pd.DataFrame(rows); R.to_csv(RESULTS / "01_pricing_errors.csv", index=False)
print("\n=== Pricing errors by volatility input ==="); print(R.round(2).to_string(index=False))

print(f"\n=== Next day: {DATE} quadratic smile on {NEXT} quotes (SPX {100*(spot[NEXT]/S-1):+.2f}%) ===")
for exd in EXPS:
    s = om_quotes(opts, NEXT, exd); T = time_to_expiry(NEXT, exd, True, "calendar")
    F, DF = parity_forward(s, spot[NEXT]); o = out_of_the_money(s, F); o["iv"], ok = implied_vols(o, F, DF, T); o = o[ok]
    for label, sig in [("yesterday ATM vol", np.full(len(o), np.polyval(smiles[exd], 0.0))), ("yesterday smile", np.polyval(smiles[exd], o.k))]:
        e = 100 * (sig - o.iv)
        print(f"  {exd} {label:18s}: IV RMSE {np.sqrt(np.mean(e**2)):.2f} pts, mean {e.mean():+.2f}")

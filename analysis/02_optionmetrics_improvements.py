"""
Improvements on the baseline (01), same OptionMetrics SPX data and expiries.

Checks:
  1. Time to expiry measured to settlement (09:30 ET for AM-settled SPX, 16:00 ET for PM-settled SPXW):
     AM vs PM implied vol on the same expiry date, and model implied vols vs OptionMetrics.
  2. SVI vs quadratic smile on held-out strikes, a butterfly no-arbitrage check, and the 2024-08-29 fits
     applied to 2024-08-30 quotes.
  3. Held-out options priced with SVI vols by Black-Scholes, the European binomial tree and Monte Carlo,
     compared with each other and with market mids.

Output: results/02_svi_vs_quadratic.csv, results/02_european_three_models.csv.
"""
import math
import numpy as np
import pandas as pd
from common import (RESULTS, bs, bs_prices, durrleman_g, error_table, fit_svi, implied_vols, load_optionmetrics,
                    om_quotes, out_of_the_money, parity_forward, svi_vol, time_to_expiry, vegas)
from pricer.binomial import binomial_price
from pricer.monte_carlo import mc_euro_vectorized

opts, spot, zc, div = load_optionmetrics()
DATE, NEXT = "2024-08-29", "2024-08-30"
EXPS = ["2024-09-20", "2024-10-18", "2024-11-15"]


def setup(date, exd, am, mode):
    s = om_quotes(opts, date, exd, am)
    F, DF = parity_forward(s, spot[date]); T = time_to_expiry(date, exd, am, mode)
    o = out_of_the_money(s, F); o["iv"], ok = implied_vols(o, F, DF, T)
    return o[ok].reset_index(drop=True), F, DF, T


print("=" * 100 + "\nIMPROVEMENT 1: time to expiry to the settlement moment\n" + "=" * 100)
print("(a) Same expiry date, AM-settled SPX vs PM-settled SPXW: ATM implied vol at K = F")
for exd in EXPS:
    for mode in ("calendar", "settlement"):
        res = {}
        for am in (True, False):
            o, F, DF, T = setup(DATE, exd, am, mode); res[am] = (float(np.interp(0, o.k, o.iv)), T * 365)
        print(f"  {exd} {mode:10s}: AM {100*res[True][0]:.2f}% (T={res[True][1]:.2f}d)  PM {100*res[False][0]:.2f}% "
              f"(T={res[False][1]:.2f}d)  gap {100*(res[False][0]-res[True][0]):+.2f} pts")

print("\n(b) Your IV - OptionMetrics IV (median, vol points), AM-settled")
for exd in EXPS:
    out = []
    for mode in ("calendar", "settlement"):
        o, F, DF, T = setup(DATE, exd, True, mode); o = o[o.impl_volatility.notna()]
        out.append(f"{mode}: {np.median(100*(o.iv-o.impl_volatility)):+.3f}")
    o, F, DF, T = setup(DATE, exd, True, "calendar"); T1 = T - 1 / 365
    iv1, _ = implied_vols(o, F, DF, T1); o = o.assign(iv1=iv1)[o.impl_volatility.notna()]
    out.append(f"to the last trading day's close, (days-1)/365: {np.median(100*(o.iv1-o.impl_volatility)):+.3f}")
    print(f"  {exd}  " + " | ".join(out))

print("\n" + "=" * 100 + "\nIMPROVEMENT 2: SVI vs quadratic smile, held-out strikes, settlement-time T\n" + "=" * 100)
rows, fits = [], {}
for exd in EXPS:
    o, F, DF, T = setup(DATE, exd, True, "settlement")
    vw = vegas(o, F, DF, T); tr = np.asarray(o.index % 2 == 0)
    quad = np.polyfit(o.k[tr], o.iv[tr], 2, w=np.sqrt(vw[tr]))
    svi = fit_svi(o.k[tr], o.iv[tr], T, vw[tr] / vw[tr].max())
    fits[exd] = dict(T=T, svi=fit_svi(o.k, o.iv, T, vw / vw.max()), quad=np.polyfit(o.k, o.iv, 2, w=np.sqrt(vw)))
    for label, sig in [("quadratic", np.polyval(quad, o.k.values)), ("SVI", svi_vol(svi, o.k.values, T))]:
        rows += error_table(o, bs_prices(o, F, DF, T, sig), sig, ~tr, label, exp=exd)
    a, b, rho, m, sg = fits[exd]["svi"]; g = durrleman_g(fits[exd]["svi"], np.linspace(-0.5, 0.3, 801))
    print(f"  {exd}: SVI a={a:.5f} b={b:.4f} rho={rho:+.3f} m={m:+.4f} sigma={sg:.4f} | min g(k) on [-0.5, 0.3] = "
          f"{g.min():+.4f} -> {'no butterfly arbitrage' if g.min() >= 0 else 'ARBITRAGE'}")
R = pd.DataFrame(rows); R.to_csv(RESULTS / "02_svi_vs_quadratic.csv", index=False); print(R.round(2).to_string(index=False))

print(f"\n  Next day: {DATE} fits on {NEXT} quotes (sticky moneyness)")
for exd in EXPS:
    o, F, DF, T = setup(NEXT, exd, True, "settlement"); right = o.KF.values > 1.03
    for label, sig in [("quadratic", np.polyval(fits[exd]["quad"], o.k.values)), ("SVI", svi_vol(fits[exd]["svi"], o.k.values, fits[exd]["T"]))]:
        e = 100 * (sig - o.iv)
        print(f"    {exd} {label:9s}: IV RMSE {np.sqrt(np.mean(e**2)):.2f} pts (K/F > 1.03: {np.sqrt(np.mean(e[right]**2)):.2f}), mean {e.mean():+.2f}")

print("\n" + "=" * 100 + "\nEUROPEAN TEST: SVI vols, Black-Scholes vs binomial tree (European, 500 steps) vs Monte Carlo\n" + "=" * 100)
rows = []
for exd in EXPS:
    o, F, DF, T = setup(DATE, exd, True, "settlement")
    vw = vegas(o, F, DF, T); tr = np.asarray(o.index % 2 == 0)
    h = o[~tr].reset_index(drop=True); sig = svi_vol(fit_svi(o.k[tr], o.iv[tr], T, vw[tr] / vw[tr].max()), h.k.values, T)
    S_, r_ = F * DF, -math.log(DF) / T
    p_bs = bs_prices(h, F, DF, T, sig)
    p_tr = np.array([binomial_price(S_, K, r_, v, T, 500, c == "C", american=False, q=0.0) for K, v, c in zip(h.K, sig, h.cp_flag)])
    # independent seed per option so MC errors are independent across strikes
    mc = [mc_euro_vectorized(S_, K, r_, 0.0, v, T, c == "C", n_paths=200_000, seed=1000 + i) for i, (K, v, c) in enumerate(zip(h.K, sig, h.cp_flag))]
    p_mc, se = np.array([m[0] for m in mc]), np.array([m[1] for m in mc])
    ok = se > 0; z = (p_mc[ok] - p_bs[ok]) / se[ok]
    print(f"  {exd}: tree vs BS max |diff| {np.max(np.abs(p_tr-p_bs)):.4f} pts, median rel {100*np.median(np.abs(p_tr/p_bs-1)):.3f}% | "
          f"MC vs BS: {100*np.mean(np.abs(z) < 2):.0f}% of {ok.sum()} with |z| < 2")
    atm = ((h.KF >= 0.97) & (h.KF < 1.03)).values
    for label, p in [("BS", p_bs), ("tree", p_tr), ("MC", p_mc)]:
        pct = 100 * (p - h.mid.values) / h.mid.values
        rows.append(dict(exp=exd, model=label, n=len(h), median_abs_pct_err=np.median(np.abs(pct)),
                         median_abs_pct_err_ATM=np.median(np.abs(pct[atm])),
                         pct_inside_spread=100 * np.mean((p >= h.bid.values) & (p <= h.ask.values))))
E = pd.DataFrame(rows); E.to_csv(RESULTS / "02_european_three_models.csv", index=False); print(E.round(2).to_string(index=False))

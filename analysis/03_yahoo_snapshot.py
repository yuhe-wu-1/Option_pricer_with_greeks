"""
First market attempt on a Yahoo Finance SPX snapshot (2026-08-27), kept to document an input problem.

Checks:
  1. Call vs put implied vol at the same strike, using the file's spot, rate and dividend yield
     vs a forward implied from put-call parity.
  2. Yahoo's own implied-vol column vs implied vols solved from mids.
  3. Held-out pricing error of a quadratic smile under each set of inputs.

Output: results/03_yahoo_snapshot.csv.
"""
import math
import numpy as np
import pandas as pd
from common import DATA, RESULTS, bs, bs_prices, error_table, implied_vols, out_of_the_money, parity_forward, vegas

y = pd.read_csv(DATA / "yahoo" / "SPX_european_clean.csv")
y = y[(y.bid > 0) & (y.spread_pct <= 0.5)].copy(); y["mid"] = y.mid_price
print(f"Yahoo SPX snapshot {y.asof_date.iloc[0]}: S in file = {y.S.iloc[0]}, r = {y.r.iloc[0]}, q = {y.q.iloc[0]}")
print(f"Yahoo implied-vol placeholders (< 0.1%): {(y.yahoo_implied_vol < 1e-3).sum()} of {len(y)} rows")

rows = []
for exd in ["2026-09-18", "2026-10-16", "2026-11-20"]:
    s = y[y.expiration == exd]; S, T, r, q = s.S.iloc[0], s["T"].iloc[0], s.r.iloc[0], s.q.iloc[0]
    F_par, DF_par = parity_forward(s, S, typ="type")
    for label, F, DF in [("file inputs", S * math.exp((r - q) * T), math.exp(-r * T)), ("parity forward", F_par, DF_par)]:
        S_, r_ = F * DF, -math.log(DF) / T
        c = s[s.type == "C"].set_index("K").mid; p = s[s.type == "P"].set_index("K").mid
        b = pd.concat([c, p], axis=1, keys=["C", "P"]).dropna(); b = b[(b.index > 0.97 * F) & (b.index < 1.03 * F)]
        gap = [100 * (bs.implied_vol(S_, K, r_, T, C, True)[0] - bs.implied_vol(S_, K, r_, T, P, False)[0]) for K, C, P in zip(b.index, b.C, b.P)]
        o = out_of_the_money(s, F, typ="type"); o["iv"], ok = implied_vols(o, F, DF, T, typ="type"); o = o[ok].reset_index(drop=True)
        tr = np.asarray(o.index % 2 == 0); vw = vegas(o, F, DF, T)
        quad = np.polyfit(o.k[tr], o.iv[tr], 2, w=np.sqrt(vw[tr])); sig = np.polyval(quad, o.k)
        rows += error_table(o, bs_prices(o, F, DF, T, sig, typ="type"), sig, ~tr, f"quadratic, {label}", exp=exd)
        yahoo_gap = np.median(100 * (o.yahoo_implied_vol - o.iv))
        print(f"  {exd} {label:14s}: F={F:.2f} | call-put IV gap near ATM: mean {np.mean(gap):+.2f} pts | "
              f"Yahoo IV - IV from mids: median {yahoo_gap:+.2f} pts")
R = pd.DataFrame(rows); R = R[R.bucket == "ALL"]; R.to_csv(RESULTS / "03_yahoo_snapshot.csv", index=False)
print(R.round(2).to_string(index=False))

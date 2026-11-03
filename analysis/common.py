"""Shared paths and helpers for the analysis scripts."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pricer import blackscholes as bs  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
OM = DATA / "optionmetrics"
YEAR = 365 * 24 * 3600.0
BUCKETS = [(0, 0.90, "<0.90"), (0.90, 0.97, "0.90-0.97"), (0.97, 1.03, "0.97-1.03"), (1.03, 9, ">1.03")]
pd.set_option("display.width", 200)


def bucket(m):
    return next(n for lo, hi, n in BUCKETS if lo <= m < hi)


# ---------------------------------------------------------------- OptionMetrics loaders
def load_optionmetrics():
    opts = pd.read_csv(OM / "spx_options_2024-08-29_30.csv")
    spot = pd.read_csv(OM / "spx_security_prices.csv").set_index("date").close
    zc = pd.read_csv(OM / "zero_coupon_yield.csv")
    div = pd.read_csv(OM / "spx_index_dividend_yield.csv").set_index("date").rate / 100
    return opts, spot, zc, div


def zero_rate(zc, date, days):
    z = zc[zc.date == date]
    return np.interp(days, z.days, z.rate) / 100


def om_quotes(opts, date, exdate, am=True, max_rel_spread=0.5):
    s = opts[(opts.date == date) & (opts.exdate == exdate) & (opts.am_settlement == int(am))].copy()
    s["K"] = s.strike_price / 1000
    s["mid"] = (s.best_bid + s.best_offer) / 2
    s = s.rename(columns={"best_bid": "bid", "best_offer": "ask"})
    return s[(s.bid > 0) & (s.ask > s.bid) & ((s.ask - s.bid) / s.mid <= max_rel_spread)]


def time_to_expiry(date, exdate, am, mode):
    """'calendar': whole days / 365. 'settlement': 16:00 ET quote to 09:30 ET (AM) or 16:00 ET (PM) on expiry day."""
    d0, d1 = pd.Timestamp(date), pd.Timestamp(exdate)
    if mode == "calendar":
        return (d1 - d0).days / 365
    t1 = d1 + (pd.Timedelta(hours=9, minutes=30) if am else pd.Timedelta(hours=16))
    return (t1 - (d0 + pd.Timedelta(hours=16))).total_seconds() / YEAR


# ---------------------------------------------------------------- core steps
def parity_forward(s, spot, band=0.10, typ="cp_flag"):
    """Regress C - P on K near the money: slope = -DF, intercept = DF*F. Robust to a few bad quotes."""
    c = s[s[typ] == "C"].set_index("K").mid
    p = s[s[typ] == "P"].set_index("K").mid
    b = pd.concat([c, p], axis=1, keys=["C", "P"]).dropna()
    b = b[(b.index > (1 - band) * spot) & (b.index < (1 + band) * spot)]
    K, y = b.index.values.astype(float), (b.C - b.P).values
    keep = np.ones(len(K), bool)
    for _ in range(5):
        sl, ic = np.polyfit(K[keep], y[keep], 1)
        res = y - (sl * K + ic)
        mad = np.median(np.abs(res[keep] - np.median(res[keep]))) * 1.4826 + 1e-12
        nk = np.abs(res) < 3 * mad
        if (nk == keep).all():
            break
        keep = nk
    return ic / (-sl), -sl


def out_of_the_money(s, F, lo=0.80, hi=1.10, typ="cp_flag"):
    o = s[((s[typ] == "C") & (s.K >= F)) | ((s[typ] == "P") & (s.K < F))].copy()
    o["KF"] = o.K / F
    o = o[(o.KF >= lo) & (o.KF <= hi)].sort_values("K").reset_index(drop=True)
    o["k"] = np.log(o.KF)
    o["bucket"] = o.KF.apply(bucket)
    return o


def implied_vols(o, F, DF, T, typ="cp_flag"):
    """Black-76 inputs through your BSM functions: S*exp(-qT) = DF*F with q = 0 and r = -ln(DF)/T."""
    S_, r_ = F * DF, -math.log(DF) / T
    out = [bs.implied_vol(S_, K, r_, T, m, c == "C") for K, m, c in zip(o.K, o.mid, o[typ])]
    return np.array([v for v, _ in out]), np.array([ok for _, ok in out])


def bs_prices(o, F, DF, T, sig, typ="cp_flag"):
    S_, r_ = F * DF, -math.log(DF) / T
    sig = np.broadcast_to(sig, len(o))
    return np.array([bs.bs_price(S_, K, r_, 0.0, v, T, c == "C") for K, v, c in zip(o.K, sig, o[typ])])


def vegas(o, F, DF, T):
    S_, r_ = F * DF, -math.log(DF) / T
    return np.array([bs.vega(S_, K, r_, v, T, 0.0) for K, v in zip(o.K, o.iv)])


def error_table(o, model_price, model_vol, mask, label, **extra):
    d = o[mask].copy()
    d["mp"], d["sig"] = np.asarray(model_price)[mask], np.asarray(model_vol)[mask]
    d["pct"] = 100 * (d.mp - d.mid) / d.mid
    d["ivp"] = 100 * (d.sig - d.iv)
    d["inside"] = (d.mp >= d.bid) & (d.mp <= d.ask)
    rows = []
    for name, g in [("ALL", d)] + [(n, d[d.bucket == n]) for _, _, n in BUCKETS]:
        if len(g):
            rows.append(dict(**extra, method=label, bucket=name, n=len(g),
                             median_abs_pct_err=np.median(np.abs(g.pct)), mean_iv_err_pts=g.ivp.mean(),
                             rmse_iv_pts=np.sqrt(np.mean(g.ivp ** 2)), pct_inside_spread=100 * g.inside.mean()))
    return rows


# ---------------------------------------------------------------- SVI
def svi_total_variance(p, k):
    a, b, rho, m, sig = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sig ** 2))


def svi_vol(p, k, T):
    return np.sqrt(np.maximum(svi_total_variance(p, k), 1e-12) / T)


def fit_svi(k, iv, T, weights):
    """Raw SVI fitted in implied-vol space (weighted least squares), several starting points."""
    k, iv, weights = map(np.asarray, (k, iv, weights))
    best = None
    for rho0 in (-0.9, -0.6, -0.3):
        for sig0 in (0.05, 0.15):
            x0 = [max((iv ** 2 * T).min() * 0.5, 1e-6), 0.1, rho0, 0.0, sig0]
            resid = lambda p: np.sqrt(weights) * (svi_vol(p, k, T) - iv)
            r = least_squares(resid, x0, bounds=([-1, 1e-6, -0.999, -1, 1e-4], [1, 5, 0.999, 1, 2]))
            if best is None or r.cost < best.cost:
                best = r
    return best.x


def durrleman_g(p, k):
    """Butterfly-arbitrage-free iff g(k) >= 0 (Gatheral & Jacquier)."""
    a, b, rho, m, sig = p
    x = k - m
    s = np.sqrt(x ** 2 + sig ** 2)
    w = svi_total_variance(p, k)
    w1, w2 = b * (rho + x / s), b * sig ** 2 / s ** 3
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2

"""
Black-Scholes-Merton European option pricer, with full Greeks for hedging.

Inputs:
  S     spot price of the underlying
  K     strike
  r     risk-free rate (continuous, e.g. 0.0394)
  sigma implied volatility (e.g. 0.22), normally taken from the market
  T     time to expiry in YEARS (e.g. 50/365)
  q     continuous dividend yield (default 0.0)

Convention for the Greeks helper `greeks()` -> trader units:
  delta : per 1 share              (hedge ratio)
  gamma : delta change per $1 move in spot
  vega  : price change per 1 vol POINT  (i.e. per 1%, = raw_vega / 100)
  theta : price change per CALENDAR DAY (= raw_theta / 365)
  rho   : price change per 1% move in rates (= raw_rho / 100)
European options only.
"""

import math


# --- helpers ---
def norm_cdf(x):
    return 0.5 * math.erfc(-x * (1.0 / math.sqrt(2.0)))


def norm_pdf(x):
    return 0.3989422804014327 * math.exp(-0.5 * x * x)  # 1/sqrt(2pi) * e^...


def compute_d1d2(S, K, r, sigma, T, q=0.0):
    sigma_sqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sigma_sqrtT
    d2 = d1 - sigma_sqrtT
    return d1, d2


# --- prices ---
def call_price(S, K, r, sigma, T, q=0.0):
    d1, d2 = compute_d1d2(S, K, r, sigma, T, q)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def put_price(S, K, r, sigma, T, q=0.0):
    d1, d2 = compute_d1d2(S, K, r, sigma, T, q)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def bs_price(S, K, r, q, sigma, T, is_call):
    return call_price(S, K, r, sigma, T, q) if is_call else put_price(S, K, r, sigma, T, q)


# --- Greeks (raw / analytic units) ---
def delta_call(S, K, r, sigma, T, q=0.0):
    d1, _ = compute_d1d2(S, K, r, sigma, T, q)
    return math.exp(-q * T) * norm_cdf(d1)


def delta_put(S, K, r, sigma, T, q=0.0):
    d1, _ = compute_d1d2(S, K, r, sigma, T, q)
    return math.exp(-q * T) * (norm_cdf(d1) - 1.0)


def gamma(S, K, r, sigma, T, q=0.0):
    # same for call and put
    d1, _ = compute_d1d2(S, K, r, sigma, T, q)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S, K, r, sigma, T, q=0.0):
    # dPrice/dsigma, per 1.00 of vol (raw). Same for call and put.
    d1, _ = compute_d1d2(S, K, r, sigma, T, q)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T)


def theta_call(S, K, r, sigma, T, q=0.0):
    # dPrice/d(calendar time), per YEAR (raw). Usually negative for long options.
    d1, d2 = compute_d1d2(S, K, r, sigma, T, q)
    term1 = -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    term2 = q * S * math.exp(-q * T) * norm_cdf(d1)
    term3 = -r * K * math.exp(-r * T) * norm_cdf(d2)
    return term1 + term2 + term3


def theta_put(S, K, r, sigma, T, q=0.0):
    d1, d2 = compute_d1d2(S, K, r, sigma, T, q)
    term1 = -S * math.exp(-q * T) * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    term2 = -q * S * math.exp(-q * T) * norm_cdf(-d1)
    term3 = r * K * math.exp(-r * T) * norm_cdf(-d2)
    return term1 + term2 + term3


def rho_call(S, K, r, sigma, T, q=0.0):
    # dPrice/dr, per 1.00 of rate (raw).
    _, d2 = compute_d1d2(S, K, r, sigma, T, q)
    return K * T * math.exp(-r * T) * norm_cdf(d2)


def rho_put(S, K, r, sigma, T, q=0.0):
    _, d2 = compute_d1d2(S, K, r, sigma, T, q)
    return -K * T * math.exp(-r * T) * norm_cdf(-d2)


def theta_1day(S, K, r, sigma, T, is_call, q=0.0):
    # Bloomberg-style theta: actual price change over one calendar day forward
    # (rolls the date by 1/365). Differs from the analytic theta_* above, which
    # is the instantaneous derivative. Both are valid; this one matches OVME.
    dt = 1.0 / 365.0
    p0 = bs_price(S, K, r, q, sigma, T, is_call)
    p1 = bs_price(S, K, r, q, sigma, T - dt, is_call)
    return p1 - p0


# --- Greeks in trader units (the hedging-friendly view) ---
def greeks(S, K, r, sigma, T, is_call, q=0.0, theta_mode="analytic", units="trader"):
    """
    theta_mode: "analytic" (instantaneous dP/dt) or "1day" (Bloomberg-style,
                actual price change over one calendar day).
    units:      "trader"    -> delta per 1 share, gamma per $1 move, vega per
                              1 vol point, theta per day, rho per 1% rate move.
                "bloomberg" -> matches OVME display: delta %, gamma % (delta-%
                              per 1% spot move), vega per vol point, theta per
                              day (1-day if theta_mode="1day"), rho per 1 bp.
    """
    if is_call:
        price = call_price(S, K, r, sigma, T, q)
        dlt = delta_call(S, K, r, sigma, T, q)
        th_an = theta_call(S, K, r, sigma, T, q)
        rh = rho_call(S, K, r, sigma, T, q)
    else:
        price = put_price(S, K, r, sigma, T, q)
        dlt = delta_put(S, K, r, sigma, T, q)
        th_an = theta_put(S, K, r, sigma, T, q)
        rh = rho_put(S, K, r, sigma, T, q)
    gma = gamma(S, K, r, sigma, T, q)
    vg = vega(S, K, r, sigma, T, q)

    # theta per day
    if theta_mode == "1day":
        theta_day = theta_1day(S, K, r, sigma, T, is_call, q)
    else:
        theta_day = th_an / 365.0

    if units == "bloomberg":
        return {
            "price":     price,
            "delta_pct": dlt * 100.0,        # %
            "gamma_pct": gma * S,            # delta-% per 1% spot move
            "vega":      vg / 100.0,         # per 1 vol point
            "theta":     theta_day,          # per calendar day
            "rho_bp":    rh / 10000.0,       # per 1 basis point
        }
    return {
        "price": price,
        "delta": dlt,                        # per 1 share
        "gamma": gma,                        # per $1 move (raw, per point)
        "vega":  vg / 100.0,                 # per 1 vol point
        "theta": theta_day,                  # per calendar day
        "rho":   rh / 100.0,                 # per 1% rate move
    }


# --- Implied volatility: Newton with bisection fallback ---
def implied_vol_call_bisect(S, K, r, T, market_price, q=0.0,
                            lo=1e-6, hi=5.0, tol=1e-8, max_iters=100):
    def f(sig):
        return call_price(S, K, r, sig, T, q) - market_price
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) * 0.5 < tol:
            return mid
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def implied_vol_call(S, K, r, T, market_price, q=0.0,
                     sigma0=0.2, tol=1e-8, max_newton=20):
    sigma = sigma0
    for _ in range(max_newton):
        diff = call_price(S, K, r, sigma, T, q) - market_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, r, sigma, T, q)
        if v < 1e-12 or not math.isfinite(v):
            break
        sigma -= diff / v
        if not math.isfinite(sigma) or sigma < 1e-6 or sigma > 5.0:
            break
    return implied_vol_call_bisect(S, K, r, T, market_price, q)


def implied_vol_put_bisect(S, K, r, T, market_price, q=0.0,
                           lo=1e-6, hi=5.0, tol=1e-8, max_iters=100):
    def f(sig):
        return put_price(S, K, r, sig, T, q) - market_price
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo * fhi > 0.0:
        return lo if abs(flo) < abs(fhi) else hi
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi - lo) * 0.5 < tol:
            return mid
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5 * (lo + hi)


def implied_vol_put(S, K, r, T, market_price, q=0.0,
                    sigma0=0.2, tol=1e-8, max_newton=20):
    sigma = sigma0
    for _ in range(max_newton):
        diff = put_price(S, K, r, sigma, T, q) - market_price
        if abs(diff) < tol:
            return sigma
        v = vega(S, K, r, sigma, T, q)
        if v < 1e-12 or not math.isfinite(v):
            break
        sigma -= diff / v
        if not math.isfinite(sigma) or sigma < 1e-6 or sigma > 5.0:
            break
    return implied_vol_put_bisect(S, K, r, T, market_price, q)


# --- Portfolio aggregator (net Greeks for hedging) ---
def portfolio_greeks(positions, S, r, q=0.0, contract_size=100):
    """
    positions: list of dicts, each with
        K, T, is_call, sigma, qty   (qty negative = short)
    Returns net Greeks scaled by contract_size, i.e. position-level numbers.
    Net delta is the number of SHARES of exposure (hedge with the opposite).
    """
    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0, "value": 0.0}
    for p in positions:
        g = greeks(S, p["K"], r, p["sigma"], p["T"], p["is_call"], q)
        scale = p["qty"] * contract_size
        net["value"] += scale * g["price"]
        net["delta"] += scale * g["delta"]
        net["gamma"] += scale * g["gamma"]
        net["vega"]  += scale * g["vega"]
        net["theta"] += scale * g["theta"]
        net["rho"]   += scale * g["rho"]
    return net


# --- Implied vol with a reliability flag ---
def implied_vol(S, K, r, T, market_price, is_call, q=0.0, min_vega_per_point=1e-3, price_tol=1e-4):
    """
    Returns (sigma, reliable). `reliable` is False when the price is outside no-arbitrage bounds,
    the solved vol does not reprice the option, or vega is so small that the price carries
    almost no information about volatility (e.g. deep in-the-money, short-dated options).
    """
    disc_S, disc_K = S * math.exp(-q * T), K * math.exp(-r * T)
    lower = max(disc_S - disc_K, 0.0) if is_call else max(disc_K - disc_S, 0.0)
    upper = disc_S if is_call else disc_K
    if not (lower < market_price < upper):
        return float("nan"), False
    solver = implied_vol_call if is_call else implied_vol_put
    sigma = solver(S, K, r, T, market_price, q)
    repriced = bs_price(S, K, r, q, sigma, T, is_call)
    reliable = abs(repriced - market_price) < price_tol and vega(S, K, r, sigma, T, q) / 100.0 > min_vega_per_point
    return sigma, reliable

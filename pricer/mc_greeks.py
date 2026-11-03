"""
Monte Carlo Greeks for European options under geometric Brownian motion, computed from a single simulation.

Pathwise method: differentiate each path's payoff with respect to the input.
    delta: e^{-rT} 1{S_T > K} S_T / S                      (call; put uses -1{S_T < K})
    vega : e^{-rT} 1{S_T > K} S_T (sqrt(T) z - sigma T)
    gamma: the derivative of the indicator is zero on every path, so the pathwise estimate is exactly 0.
           (The true gamma comes from the jump in the payoff's slope at the strike, which the method cannot see.)

Likelihood-ratio method: keep the payoff, differentiate the probability density of S_T instead.
    delta weight: z / (S sigma sqrt(T))
    gamma weight: (z^2 - 1 - z sigma sqrt(T)) / (S^2 sigma^2 T)

Every estimate comes with a standard error. Paths are independent draws (no antithetic pairing),
so the usual SE formula applies.
"""
import numpy as np


def _simulate(S, r, q, sigma, T, n_paths, seed):
    z = np.random.default_rng(seed).standard_normal(n_paths)
    ST = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)
    return z, ST, np.exp(-r * T)


def _mean_se(x):
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def mc_greeks_pathwise(S, K, r, q, sigma, T, is_call, n_paths=200_000, seed=0):
    """Returns {'delta': (est, se), 'vega': (est, se), 'gamma': (0.0, 0.0)}."""
    z, ST, disc = _simulate(S, r, q, sigma, T, n_paths, seed)
    itm = (ST > K) if is_call else (ST < K)
    sign = 1.0 if is_call else -1.0
    delta = disc * sign * itm * ST / S
    vega = disc * sign * itm * ST * (np.sqrt(T) * z - sigma * T)
    return {"delta": _mean_se(delta), "vega": _mean_se(vega), "gamma": (0.0, 0.0)}


def mc_greeks_likelihood_ratio(S, K, r, q, sigma, T, is_call, n_paths=200_000, seed=0):
    """Returns {'delta': (est, se), 'gamma': (est, se)}."""
    z, ST, disc = _simulate(S, r, q, sigma, T, n_paths, seed)
    payoff = disc * (np.maximum(ST - K, 0.0) if is_call else np.maximum(K - ST, 0.0))
    sd = sigma * np.sqrt(T)
    delta = payoff * z / (S * sd)
    gamma = payoff * (z ** 2 - 1.0 - z * sd) / (S ** 2 * sd ** 2)
    return {"delta": _mean_se(delta), "gamma": _mean_se(gamma)}

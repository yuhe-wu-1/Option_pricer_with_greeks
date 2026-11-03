"""Unit tests for the pricing library. Run from the repo root:  python -m pytest -q"""
import math
import numpy as np
import pytest
from pricer import blackscholes as bs
from pricer.binomial import binomial_price
from pricer.monte_carlo import mc_euro_vectorized, mc_euro_option_price

CASES = [  # S, K, r, q, sigma, T
    (100, 100, 0.05, 0.00, 0.20, 1.0),
    (100, 80, 0.03, 0.01, 0.25, 0.5),
    (100, 120, 0.04, 0.02, 0.30, 2.0),
    (5600, 5000, 0.052, 0.0098, 0.14, 0.06),
    (50, 55, 0.01, 0.00, 0.45, 0.25),
]


def test_textbook_values():
    assert bs.call_price(100, 100, 0.05, 0.20, 1.0) == pytest.approx(10.4506, abs=1e-4)
    assert bs.put_price(100, 100, 0.05, 0.20, 1.0) == pytest.approx(5.5735, abs=1e-4)


@pytest.mark.parametrize("S,K,r,q,v,T", CASES)
def test_put_call_parity(S, K, r, q, v, T):
    lhs = bs.call_price(S, K, r, v, T, q) - bs.put_price(S, K, r, v, T, q)
    rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10 * max(S, 1)


@pytest.mark.parametrize("S,K,r,q,v,T", CASES)
def test_greeks_match_finite_differences(S, K, r, q, v, T):
    h = 1e-4 * S
    for is_call in (True, False):
        price = lambda s=S, sig=v, t=T, rr=r: bs.bs_price(s, K, rr, q, sig, t, is_call)
        delta = bs.delta_call(S, K, r, v, T, q) if is_call else bs.delta_put(S, K, r, v, T, q)
        assert delta == pytest.approx((price(S + h) - price(S - h)) / (2 * h), rel=1e-5, abs=1e-8)
        assert bs.gamma(S, K, r, v, T, q) == pytest.approx((price(S + h) - 2 * price() + price(S - h)) / h ** 2, rel=1e-3, abs=1e-8)
        dv = 1e-5
        assert bs.vega(S, K, r, v, T, q) == pytest.approx((price(sig=v + dv) - price(sig=v - dv)) / (2 * dv), rel=1e-6, abs=1e-8)
        dt = 1e-5
        theta = bs.theta_call(S, K, r, v, T, q) if is_call else bs.theta_put(S, K, r, v, T, q)
        assert theta == pytest.approx(-(price(t=T + dt) - price(t=T - dt)) / (2 * dt), rel=1e-4, abs=1e-6)
        dr = 1e-6
        rho = bs.rho_call(S, K, r, v, T, q) if is_call else bs.rho_put(S, K, r, v, T, q)
        assert rho == pytest.approx((price(rr=r + dr) - price(rr=r - dr)) / (2 * dr), rel=1e-5, abs=1e-6)


def test_implied_vol_round_trip_and_flag():
    """Grid of 420 cases. Every reliable result must recover the true vol; unreliable ones must have tiny vega."""
    recovered = flagged = 0
    for K in (60, 80, 90, 100, 110, 120, 140):
        for T in (0.02, 0.1, 0.25, 1.0, 2.0):
            for v in (0.1, 0.2, 0.4, 0.8):
                for r in (0.0, 0.03, 0.06):
                    for is_call in (True,):
                        p = bs.bs_price(100, K, r, 0.0, v, T, is_call)
                        sig, ok = bs.implied_vol(100, K, r, T, p, is_call)
                        if ok:
                            assert sig == pytest.approx(v, abs=1e-5)
                            recovered += 1
                        else:
                            assert bs.vega(100, K, r, v, T) / 100 < 1e-2  # only near-zero-vega cases may be flagged
                            flagged += 1
    assert recovered + flagged == 420 and recovered > 300  # this grid includes very deep, very short-dated options


@pytest.mark.parametrize("S,K,r,q,v,T", CASES)
def test_binomial_converges_to_black_scholes(S, K, r, q, v, T):
    for is_call in (True, False):
        exact = bs.bs_price(S, K, r, q, v, T, is_call)
        tree = binomial_price(S, K, r, v, T, 800, is_call, american=False, q=q)
        assert abs(tree - exact) < 2e-3 * S


def test_american_premium():
    S, K, r, v, T = 100, 100, 0.05, 0.2, 1.0
    eu_c = binomial_price(S, K, r, v, T, 500, True, False); am_c = binomial_price(S, K, r, v, T, 500, True, True)
    eu_p = binomial_price(S, K, r, v, T, 500, False, False); am_p = binomial_price(S, K, r, v, T, 500, False, True)
    assert am_c == pytest.approx(eu_c, abs=1e-10)   # no dividends: never optimal to exercise a call early
    assert am_p > eu_p                              # early exercise is valuable for the put


@pytest.mark.parametrize("antithetic", [True, False])
def test_monte_carlo_standard_error_is_honest(antithetic):
    """z-scores across independent seeds should look standard normal (checks the SE, not just the mean)."""
    S, K, r, q, v, T = 100, 105, 0.04, 0.01, 0.25, 0.5
    exact = bs.call_price(S, K, r, v, T, q)
    z = []
    for seed in range(200):
        p, se = mc_euro_vectorized(S, K, r, q, v, T, True, n_paths=20_000, seed=seed, antithetic=antithetic)
        z.append((p - exact) / se)
    z = np.array(z)
    assert abs(z.mean()) < 0.25 and 0.8 < z.std() < 1.2


def test_monte_carlo_se_halves_with_four_times_paths():
    args = (100, 100, 0.05, 0.0, 0.2, 1.0, True)
    _, se1 = mc_euro_vectorized(*args, n_paths=50_000, seed=1, control_variate=False)
    _, se4 = mc_euro_vectorized(*args, n_paths=200_000, seed=1, control_variate=False)
    assert se1 / se4 == pytest.approx(2.0, rel=0.05)


def test_loop_and_vectorized_monte_carlo_agree():
    a = mc_euro_option_price(100, 95, 0.03, 0.0, 0.3, 0.5, False, n_paths=40_000, seed=3)
    b = mc_euro_vectorized(100, 95, 0.03, 0.0, 0.3, 0.5, False, n_paths=40_000, seed=3, control_variate=False)
    assert abs(a[0] - b[0]) < 4 * math.hypot(a[1], b[1])


# ---------------------------------------------------------------- Monte Carlo Greeks and variance reduction
from pricer.mc_greeks import mc_greeks_likelihood_ratio, mc_greeks_pathwise


@pytest.mark.parametrize("K,is_call", [(110, True), (95, False)])
def test_mc_greeks_unbiased_with_honest_se(K, is_call):
    """Across independent seeds, z-scores vs the analytic Greeks should have mean ~0 and sd ~1."""
    S, r, q, v, T = 100, 0.05, 0.0, 0.2, 0.25
    exact = dict(delta=(bs.delta_call if is_call else bs.delta_put)(S, K, r, v, T, q),
                 vega=bs.vega(S, K, r, v, T, q), gamma=bs.gamma(S, K, r, v, T, q))
    z = {"pw_delta": [], "pw_vega": [], "lr_delta": [], "lr_gamma": []}
    for seed in range(150):
        pw = mc_greeks_pathwise(S, K, r, q, v, T, is_call, 20_000, seed)
        lr = mc_greeks_likelihood_ratio(S, K, r, q, v, T, is_call, 20_000, seed + 10_000)
        z["pw_delta"].append((pw["delta"][0] - exact["delta"]) / pw["delta"][1])
        z["pw_vega"].append((pw["vega"][0] - exact["vega"]) / pw["vega"][1])
        z["lr_delta"].append((lr["delta"][0] - exact["delta"]) / lr["delta"][1])
        z["lr_gamma"].append((lr["gamma"][0] - exact["gamma"]) / lr["gamma"][1])
    for name, zz in z.items():
        zz = np.array(zz)
        assert abs(zz.mean()) < 0.3, name
        assert 0.75 < zz.std() < 1.25, name


def test_pathwise_gamma_is_zero_but_likelihood_ratio_is_not():
    pw = mc_greeks_pathwise(100, 100, 0.05, 0.0, 0.2, 1.0, True, 50_000, 1)
    lr = mc_greeks_likelihood_ratio(100, 100, 0.05, 0.0, 0.2, 1.0, True, 50_000, 1)
    assert pw["gamma"] == (0.0, 0.0)
    assert lr["gamma"][0] == pytest.approx(bs.gamma(100, 100, 0.05, 0.2, 1.0), abs=4 * lr["gamma"][1])


def test_control_variate_reduction_matches_theory():
    """Measured variance ratio should match 1 / (1 - rho^2) for an at-the-money call."""
    S, K, r, v, T = 100, 100, 0.05, 0.2, 1.0
    z = np.random.default_rng(0).standard_normal(500_000)
    ST = S * np.exp((r - 0.5 * v * v) * T + v * np.sqrt(T) * z)
    rho = np.corrcoef(np.maximum(ST - K, 0), ST)[0, 1]
    plain = [mc_euro_vectorized(S, K, r, 0, v, T, True, 10_000, s, antithetic=False, control_variate=False)[0] for s in range(800)]
    cv = [mc_euro_vectorized(S, K, r, 0, v, T, True, 10_000, s, antithetic=False, control_variate=True)[0] for s in range(800)]
    assert np.var(plain) / np.var(cv) == pytest.approx(1 / (1 - rho ** 2), rel=0.15)

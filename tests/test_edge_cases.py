"""Edge cases: expiry (T = 0), zero volatility, and impossible inputs.

A production pricer must NEVER return nan. One nan in a portfolio makes the
whole book's total nan (nan + anything = nan), and the risk report breaks.
Tests are written by Claude**
"""

import warnings                  # Python's warnings system (used to catch RuntimeWarnings)

import numpy as np
import pytest

from options_pricer.pricing.black_scholes import bs_price
from options_pricer.pricing.greeks import bs_greeks


SPOTS = np.array([80.0, 100.0, 120.0])     # OTM, ATM, ITM for a call with K = 100


# ---------------------------------------------------------------------------
# 1. At expiry, the price is just the payoff
# ---------------------------------------------------------------------------
def test_price_at_expiry_is_payoff():
    call = bs_price(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
    put = bs_price(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2, option_type="put")

    assert call == pytest.approx([0, 0, 20])     # max(S − K, 0)
    assert put == pytest.approx([20, 0, 0])      # max(K − S, 0)


# ---------------------------------------------------------------------------
# 2. Zero vol: the future is certain (your very first Q3 answer!)
# ---------------------------------------------------------------------------
def test_zero_vol_price():
    # With σ = 0 and no dividends: call = max(S − K·e^(−rT), 0)
    T, r = 1.0, 0.05
    call = bs_price(S=SPOTS, K=100, T=T, r=r, sigma=0.0)

    expected = np.maximum(SPOTS - 100 * np.exp(-r * T), 0)
    assert call == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. No jumps: a tiny bit before expiry ≈ at expiry
# ---------------------------------------------------------------------------
def test_price_is_continuous_near_expiry():
    almost_expired = bs_price(S=SPOTS, K=100, T=1e-8, r=0.05, sigma=0.2)
    expired = bs_price(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2)

    assert almost_expired == pytest.approx(expired, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. No nan and no noisy warnings
# ---------------------------------------------------------------------------
def test_no_nan_and_no_warnings_at_expiry():
    # "error" turns any warning into a crash, so this test fails if numpy
    # prints a "divide by zero" warning anywhere inside the pricer
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        price = bs_price(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2)
        greeks = bs_greeks(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2)

    assert not np.any(np.isnan(price))                     # np.isnan: True where value is nan
    for name, values in greeks.items():
        assert not np.any(np.isnan(values)), name


# ---------------------------------------------------------------------------
# 5. Greeks at expiry: the "cliff"
# ---------------------------------------------------------------------------
def test_greeks_at_expiry():
    call = bs_greeks(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2, option_type="call")
    put = bs_greeks(S=SPOTS, K=100, T=0, r=0.05, sigma=0.2, option_type="put")

    # OTM call: delta 0. ITM call: delta 1 (it IS the stock now)
    assert call["delta"][0] == 0
    assert call["delta"][2] == 1
    # ITM put (spot 80): delta −1. OTM put (spot 120): delta 0
    assert put["delta"][0] == -1
    assert put["delta"][2] == 0
    # No time left -> no curvature and no vol exposure
    assert call["gamma"] == pytest.approx([0, 0, 0])
    assert call["vega"] == pytest.approx([0, 0, 0])


def test_zero_vol_greeks_itm_call():
    # σ = 0, ITM call: price = S − K·e^(−rT), a stock minus a bond.
    # So delta = 1, theta = −r·K·e^(−rT), rho = K·T·e^(−rT)
    T, r, K = 1.0, 0.05, 100
    g = bs_greeks(S=120, K=K, T=T, r=r, sigma=0.0)

    assert g["delta"] == pytest.approx(1.0)
    assert g["gamma"] == 0
    assert g["vega"] == 0
    assert g["theta"] == pytest.approx(-r * K * np.exp(-r * T))
    assert g["rho"] == pytest.approx(K * T * np.exp(-r * T))


# ---------------------------------------------------------------------------
# 6. Impossible inputs fail loudly
# ---------------------------------------------------------------------------
# Each line is one bad set of inputs; the test runs once per line
@pytest.mark.parametrize(
    "bad_inputs",
    [
        dict(S=100, K=100, T=-1, r=0.05, sigma=0.2),     # negative time
        dict(S=100, K=100, T=1, r=0.05, sigma=-0.2),     # negative vol
        dict(S=0, K=100, T=1, r=0.05, sigma=0.2),        # zero spot
        dict(S=100, K=-100, T=1, r=0.05, sigma=0.2),     # negative strike
    ],
)
def test_impossible_inputs_raise(bad_inputs):
    with pytest.raises(ValueError):
        bs_price(**bad_inputs)
    with pytest.raises(ValueError):
        bs_greeks(**bad_inputs)

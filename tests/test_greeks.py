"""Tests for the analytic Greeks (pricing/greeks.py).

Main idea: two completely different methods must agree.
  - analytic:  the Greek formulas in greeks.py
  - numerical: bump the input, re-price, measure the change (numerical.py)
If a formula had a typo, the two would disagree and the test would fail.
"""

import numpy as np
import pytest

from options_pricer.pricing.greeks import bs_greeks, to_desk_units
from options_pricer.pricing.numerical import numerical_greeks


GREEK_NAMES = ["delta", "gamma", "vega", "theta", "rho"]


# ---------------------------------------------------------------------------
# 1. Analytic Greeks == numerical Greeks, across many scenarios
# ---------------------------------------------------------------------------
# 2 option types × 3 spots (OTM/ATM/ITM) × 3 expiries × 2 dividend yields = 36 cases
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("S", [80, 100, 120])
@pytest.mark.parametrize("T", [0.1, 1.0, 2.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_analytic_matches_numerical(option_type, S, T, q):
    inputs = dict(S=S, K=100, T=T, r=0.05, sigma=0.25, q=q, option_type=option_type)

    analytic = bs_greeks(**inputs)          # ** unpacks the dict into named arguments
    numeric = numerical_greeks(**inputs)

    for name in GREEK_NAMES:
        # rel=1e-4: within 0.01% of each other
        # abs=1e-6: OR within 0.000001 (for Greeks that are close to zero)
        assert analytic[name] == pytest.approx(numeric[name], rel=1e-4, abs=1e-6), name


# ---------------------------------------------------------------------------
# 2. Relationships between Greeks that must always hold
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("S", [80, 100, 120])
def test_call_minus_put_delta(S):
    # From put-call parity: call delta − put delta = e^(−qT)
    T, q = 1.0, 0.03
    call = bs_greeks(S=S, K=100, T=T, r=0.05, sigma=0.2, q=q, option_type="call")
    put = bs_greeks(S=S, K=100, T=T, r=0.05, sigma=0.2, q=q, option_type="put")

    assert call["delta"] - put["delta"] == pytest.approx(np.exp(-q * T))


@pytest.mark.parametrize("S", [80, 100, 120])
def test_gamma_and_vega_same_for_call_and_put(S):
    # Call − put = stock − bond, which has no curvature and no vol exposure
    call = bs_greeks(S=S, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put = bs_greeks(S=S, K=100, T=1, r=0.05, sigma=0.2, option_type="put")

    assert call["gamma"] == pytest.approx(put["gamma"])
    assert call["vega"] == pytest.approx(put["vega"])


def test_vega_gamma_link():
    # Vega = Gamma × S² × σ × T  (the interview fact)
    S, T, sigma = 100, 1.0, 0.2
    g = bs_greeks(S=S, K=100, T=T, r=0.05, sigma=sigma)

    assert g["vega"] == pytest.approx(g["gamma"] * S**2 * sigma * T)


# ---------------------------------------------------------------------------
# 3. Signs and ranges every trader expects
# ---------------------------------------------------------------------------
def test_greek_signs():
    spots = np.linspace(60, 140, 81)       # check across a whole range of spots
    call = bs_greeks(S=spots, K=100, T=0.5, r=0.05, sigma=0.2, option_type="call")
    put = bs_greeks(S=spots, K=100, T=0.5, r=0.05, sigma=0.2, option_type="put")

    assert np.all((call["delta"] >= 0) & (call["delta"] <= 1))    # call delta in [0, 1]
    assert np.all((put["delta"] >= -1) & (put["delta"] <= 0))     # put delta in [−1, 0]
    assert np.all(call["gamma"] >= 0)                              # long options: gamma ≥ 0
    assert np.all(call["vega"] >= 0)                               # long options: vega ≥ 0
    assert np.all(call["rho"] >= 0)                                # calls gain when rates rise
    assert np.all(put["rho"] <= 0)                                 # puts lose when rates rise


def test_gamma_peaks_near_expiry():
    # The prediction you got wrong: ATM gamma RISES as expiry approaches
    gammas = [bs_greeks(S=100, K=100, T=T, r=0.05, sigma=0.2)["gamma"] for T in [1, 0.25, 1/12, 1/52]]

    # each value is bigger than the one before it
    assert all(later > earlier for earlier, later in zip(gammas, gammas[1:]))


# ---------------------------------------------------------------------------
# 4. Desk units conversion
# ---------------------------------------------------------------------------
def test_desk_units():
    raw = bs_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2)
    desk = to_desk_units(raw)

    assert desk["delta"] == raw["delta"]                     # unchanged
    assert desk["vega"] == pytest.approx(raw["vega"] / 100)    # per vol point
    assert desk["theta"] == pytest.approx(raw["theta"] / 365)  # per day
    assert desk["rho"] == pytest.approx(raw["rho"] / 100)      # per 1%
    assert desk["vega"] == pytest.approx(0.3752, abs=1e-4)     # known value

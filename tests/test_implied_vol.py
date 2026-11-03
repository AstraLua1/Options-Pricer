"""Tests for the implied vol solver (volatility/implied.py).

Main idea - a "round trip":
  pick a vol -> price the option -> solve for implied vol -> we should get the same vol back.
Tests are written by Claude**
"""

import numpy as np
import pytest

from options_pricer.pricing.black_scholes import bs_price
from options_pricer.volatility.implied import bs_implied_vol


# ---------------------------------------------------------------------------
# 1. Round trip across many scenarios
# ---------------------------------------------------------------------------
# 2 types × 5 strikes × 4 expiries × 4 vols = 160 cases
@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K", [50, 80, 100, 120, 200])          # deep ITM ... deep OTM
@pytest.mark.parametrize("T", [7 / 365, 0.25, 1.0, 5.0])        # 1 week ... 5 years
@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.0])        # 5% (calm) ... 200% (crypto)
def test_round_trip(option_type, K, T, sigma):
    S, r = 100, 0.05
    price = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)

    # If the option is so far out of the money that it's worth ~0, the price
    # carries no information about vol - nobody could recover it. Skip those.
    otm_price = bs_price(S=S, K=K, T=T, r=r, sigma=sigma,
                         option_type="put" if K < S else "call")
    if otm_price < 1e-6:
        pytest.skip("option worth ~0: price has no vol information left")

    implied = bs_implied_vol(price, S=S, K=K, T=T, r=r, option_type=option_type)
    assert implied == pytest.approx(sigma, abs=1e-6)


# ---------------------------------------------------------------------------
# 2. A known example
# ---------------------------------------------------------------------------
def test_reference_example():
    # The call we priced in Milestone 1: 10.4506 at 20% vol
    assert bs_implied_vol(10.4506, S=100, K=100, T=1, r=0.05) == pytest.approx(0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. Newton normally does the work, in a few steps
# ---------------------------------------------------------------------------
def test_newton_is_used_and_fast():
    price = bs_price(S=100, K=100, T=1, r=0.05, sigma=0.3)
    vol, info = bs_implied_vol(price, S=100, K=100, T=1, r=0.05, return_info=True)

    assert info["method"] == "newton"
    assert info["iterations"] <= 6
    assert vol == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 4. Brent backup gives the same answer
# ---------------------------------------------------------------------------
def test_brent_fallback_gives_same_answer():
    price = bs_price(S=100, K=120, T=0.5, r=0.05, sigma=0.6)

    # newton_max_iter=0 means "skip Newton entirely" -> forces Brent
    vol, info = bs_implied_vol(price, S=100, K=120, T=0.5, r=0.05,
                               newton_max_iter=0, return_info=True)

    assert info["method"] == "brent"
    assert vol == pytest.approx(0.6, abs=1e-8)


# ---------------------------------------------------------------------------
# 5. Impossible prices -> nan (a "bad quote" flag)
# ---------------------------------------------------------------------------
def test_price_below_payoff_has_no_solution():
    # An ITM call can't be worth less than its discounted payoff
    payoff_value = 120 - 100 * np.exp(-0.05)         # ≈ 24.88
    vol = bs_implied_vol(payoff_value - 1, S=120, K=100, T=1, r=0.05)
    assert np.isnan(vol)


def test_price_above_stock_has_no_solution():
    # A call can never be worth more than the stock itself
    vol = bs_implied_vol(150, S=100, K=100, T=1, r=0.05)
    assert np.isnan(vol)


def test_price_at_payoff_gives_zero_vol():
    payoff_value = 120 - 100 * np.exp(-0.05)
    assert bs_implied_vol(payoff_value, S=120, K=100, T=1, r=0.05) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Bad inputs fail loudly
# ---------------------------------------------------------------------------
def test_expired_option_raises():
    with pytest.raises(ValueError):
        bs_implied_vol(5.0, S=100, K=100, T=0, r=0.05)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        bs_implied_vol(5.0, S=100, K=100, T=1, r=0.05, option_type="Call")

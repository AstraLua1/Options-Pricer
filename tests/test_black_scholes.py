"""Tests for the Black-Scholes pricer (pricing/black_scholes.py).

Run them all from the PyCharm terminal with:   pytest
pytest finds every file named test_*.py and runs every function named test_*.
"""

import numpy as np
import pytest                                   # the testing tool

from options_pricer.pricing.black_scholes import bs_price


# ---------------------------------------------------------------------------
# 1. Known answer: Hull's textbook example
# ---------------------------------------------------------------------------
def test_reference_values():
    # S=K=100, T=1, r=5%, sigma=20% is a standard example: call 10.4506, put 5.5735
    call = bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="call")
    put = bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")

    # pytest.approx = "equal, allowing a tiny rounding difference"
    # (never use == to compare decimals: 0.1 + 0.2 == 0.3 is False in Python!)
    assert call == pytest.approx(10.4506, abs=1e-4)
    assert put == pytest.approx(5.5735, abs=1e-4)


# ---------------------------------------------------------------------------
# 2. Put-call parity: Call − Put = S·e^(−qT) − K·e^(−rT)
# ---------------------------------------------------------------------------
# @pytest.mark.parametrize runs the SAME test many times with different inputs.
# Here: 3 spots × 3 expiries × 2 dividend yields = 18 test cases from one function.
@pytest.mark.parametrize("S", [80, 100, 120])
@pytest.mark.parametrize("T", [0.1, 1.0, 3.0])
@pytest.mark.parametrize("q", [0.0, 0.03])
def test_put_call_parity(S, T, q):
    K, r, sigma = 100, 0.05, 0.25

    call = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="call")
    put = bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type="put")

    # Right-hand side: (stock minus dividends) − (strike discounted to today)
    expected = S * np.exp(-q * T) - K * np.exp(-r * T)

    assert call - put == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# 3. No-arbitrage bounds: a call is never worth less than its zero-vol value
# ---------------------------------------------------------------------------
def test_call_lower_bound():
    # From the very first question: with zero vol, call = max(S − K·e^(−rT), 0).
    # Any real call must be worth at least that.
    S = np.linspace(50, 150, 101)          # 101 spots from 50 to 150
    K, T, r = 100, 1.0, 0.05

    call = bs_price(S=S, K=K, T=T, r=r, sigma=0.2)
    lower_bound = np.maximum(S - K * np.exp(-r * T), 0)

    # np.all(...) = "true for every single spot"; tiny tolerance for rounding
    assert np.all(call >= lower_bound - 1e-12)
    # ...and a call can never be worth more than the stock itself
    assert np.all(call <= S)


# ---------------------------------------------------------------------------
# 4. Vectorisation: pricing an array gives the same as pricing one at a time
# ---------------------------------------------------------------------------
def test_vectorised_matches_scalar():
    spots = [90, 100, 110]

    all_at_once = bs_price(S=spots, K=100, T=1, r=0.05, sigma=0.2)
    one_by_one = [bs_price(S=s, K=100, T=1, r=0.05, sigma=0.2) for s in spots]

    assert all_at_once == pytest.approx(one_by_one)


# ---------------------------------------------------------------------------
# 5. Bad input fails loudly
# ---------------------------------------------------------------------------
def test_invalid_option_type_raises():
    # pytest.raises checks that the code inside DOES throw this error.
    # The test FAILS if no error happens.
    with pytest.raises(ValueError):
        bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="Call")

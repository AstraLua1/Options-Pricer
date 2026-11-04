"""Tests for the arbitrage checks (volatility/arbitrage.py)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.volatility.arbitrage import butterfly_g, check_butterfly, check_calendar
from options_pricer.volatility.svi import SVIParams, fit_svi_by_expiry

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


def fit_row(p, T, expiry, k_min=-1.0, k_max=1.0):
    """Build one row like fit_svi_by_expiry() makes, from hand-picked knobs."""
    return {"expiry": pd.Timestamp(expiry, tz="UTC"), "T": T, "forward": 100.0,
            "a": p.a, "b": p.b, "rho": p.rho, "m": p.m, "sigma": p.sigma,
            "rmse_pts": 0.0, "points": 10, "k_min": k_min, "k_max": k_max}


# A well-behaved smile
GOOD = SVIParams(a=0.02, b=0.1, rho=-0.3, m=0.0, sigma=0.2)

# A famous BAD example (Axel Vogt, in Gatheral & Jacquier 2014): it fits like a
# normal smile but implies a negative probability around k ≈ 0.9
BAD = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)


# ---------------------------------------------------------------------------
# 1. Butterfly
# ---------------------------------------------------------------------------
def test_good_smile_passes_butterfly():
    k = np.linspace(-1, 1, 201)
    assert butterfly_g(k, GOOD).min() > 0


def test_vogt_smile_fails_butterfly():
    k = np.linspace(-1.5, 1.5, 601)
    assert butterfly_g(k, BAD).min() < 0


def test_check_butterfly_flags_the_bad_one():
    fits = pd.DataFrame([fit_row(GOOD, 1.0, "2027-01-01"),
                         fit_row(BAD, 1.0, "2027-02-01", k_min=-1.5, k_max=1.5)])
    result = check_butterfly(fits)
    assert list(result["ok"]) == [True, False]


# ---------------------------------------------------------------------------
# 2. Calendar
# ---------------------------------------------------------------------------
def test_more_variance_later_passes_calendar():
    later = SVIParams(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.2)     # same shape, higher
    fits = pd.DataFrame([fit_row(GOOD, 0.5, "2027-01-01"), fit_row(later, 1.0, "2027-07-01")])
    assert check_calendar(fits)["ok"].all()


def test_less_variance_later_fails_calendar():
    lower = SVIParams(a=0.005, b=0.1, rho=-0.3, m=0.0, sigma=0.2)    # LONGER expiry but LOWER
    fits = pd.DataFrame([fit_row(GOOD, 0.5, "2027-01-01"), fit_row(lower, 1.0, "2027-07-01")])
    assert not check_calendar(fits)["ok"].all()


# ---------------------------------------------------------------------------
# 3. Real Deribit data should be arbitrage-free
# ---------------------------------------------------------------------------
def test_real_snapshot_has_no_arbitrage():
    with open(FIXTURE) as f:
        raw = json.load(f)
    chain = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    clean, _ = clean_chain(chain)
    fits = fit_svi_by_expiry(add_implied_vols(otm_only(clean)))

    assert check_butterfly(fits)["ok"].all()
    assert check_calendar(fits)["ok"].all()

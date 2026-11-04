"""Tests for the SVI fit (volatility/svi.py)."""

import json
from pathlib import Path

import numpy as np
import pytest

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.volatility.svi import (
    SVIParams,
    fit_svi,
    fit_svi_by_expiry,
    svi_implied_vol,
    svi_total_variance,
)

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


# ---------------------------------------------------------------------------
# 1. The formula itself
# ---------------------------------------------------------------------------
def test_formula_at_the_bottom():
    # With m = 0 and rho = 0, at k = 0: w = a + b × σ
    p = SVIParams(a=0.01, b=0.1, rho=0.0, m=0.0, sigma=0.2)
    assert svi_total_variance(0.0, p) == pytest.approx(0.01 + 0.1 * 0.2)


def test_negative_rho_lifts_the_left_side():
    # rho < 0 should make low strikes (k < 0) have higher vol than high strikes (k > 0)
    p = SVIParams(a=0.01, b=0.1, rho=-0.5, m=0.0, sigma=0.2)
    assert svi_implied_vol(-0.3, 0.5, p) > svi_implied_vol(0.3, 0.5, p)


# ---------------------------------------------------------------------------
# 2. Round trip: make a smile from known knobs, fit it, get the same smile back
# ---------------------------------------------------------------------------
def test_fit_recovers_a_known_smile():
    true = SVIParams(a=0.02, b=0.15, rho=-0.3, m=0.02, sigma=0.25)
    T = 0.25
    k = np.linspace(-0.5, 0.4, 15)
    vols = svi_implied_vol(k, T, true)

    fitted, rmse = fit_svi(k, vols, T)

    assert rmse < 0.01                                               # under 0.01 vol points
    assert svi_implied_vol(k, T, fitted) == pytest.approx(vols, abs=1e-5)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_svi([0.0, 0.1, 0.2], [0.4, 0.4, 0.4], T=0.5)


# ---------------------------------------------------------------------------
# 3. Real Deribit data
# ---------------------------------------------------------------------------
@pytest.fixture
def otm_chain():
    with open(FIXTURE) as f:
        raw = json.load(f)
    chain = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    clean, _ = clean_chain(chain)
    return add_implied_vols(otm_only(clean))


def test_fits_real_smiles_closely(otm_chain):
    fits = fit_svi_by_expiry(otm_chain)
    assert len(fits) >= 1
    assert (fits["rmse_pts"] < 1.0).all()           # curve within 1 vol point of the dots on average


def test_expiries_with_too_few_dots_are_skipped(otm_chain):
    fits = fit_svi_by_expiry(otm_chain)
    dots_per_expiry = otm_chain.dropna(subset=["iv_mid"]).groupby("expiry").size()
    fitted_expiries = set(fits["expiry"])
    for expiry, count in dots_per_expiry.items():
        assert (expiry in fitted_expiries) == (count >= 5)

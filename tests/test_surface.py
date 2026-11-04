"""Tests for the volatility surface (volatility/surface.py)."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # no pop-up windows during tests

import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402
import pytest                    # noqa: E402

from options_pricer.data.chain import add_implied_vols, build_chain        # noqa: E402
from options_pricer.data.cleaning import clean_chain, otm_only             # noqa: E402
from options_pricer.plotting import plot_surface                           # noqa: E402
from options_pricer.volatility.surface import build_surface               # noqa: E402
from options_pricer.volatility.svi import params_from_row, svi_implied_vol  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


@pytest.fixture
def otm_chain():
    with open(FIXTURE) as f:
        raw = json.load(f)
    chain = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    clean, _ = clean_chain(chain)
    return add_implied_vols(otm_only(clean))


@pytest.fixture
def surface(otm_chain):
    return build_surface(otm_chain)


def test_on_a_listed_expiry_it_equals_that_svi_curve(surface):
    fit = surface.fits.iloc[0]
    expected = svi_implied_vol(np.log(80000 / fit["forward"]), fit["T"], params_from_row(fit))
    assert surface.vol_at_T(80000, fit["T"]) == pytest.approx(float(expected))


def test_matches_market_vols(surface, otm_chain):
    # For every option used in the fit, the surface should be close to the market vol
    used = otm_chain[otm_chain["expiry"].isin(surface.fits["expiry"])].dropna(subset=["iv_mid"])
    for _, row in used.iterrows():
        assert surface.vol_at_T(row["strike"], row["T"]) == pytest.approx(row["iv_mid"], abs=0.01)  # 1 vol pt


def test_between_expiries_total_variance_is_in_between(surface):
    # Halfway between two listed expiries, total variance (vol² × T) must lie between theirs
    if len(surface.fits) < 2:
        pytest.skip("need two expiries")
    T1, T2 = surface.T_list[0], surface.T_list[1]
    T_mid = (T1 + T2) / 2
    strike = surface.forward(T_mid)                                   # at the money
    w1 = surface.vol_at_T(strike, T1) ** 2 * T1
    w2 = surface.vol_at_T(strike, T2) ** 2 * T2
    w_mid = surface.vol_at_T(strike, T_mid) ** 2 * T_mid
    assert min(w1, w2) - 1e-4 <= w_mid <= max(w1, w2) + 1e-4


def test_date_input_works(surface):
    one_month_later = surface.snapshot_time + pd.Timedelta(days=30)
    vol = surface.vol(80000, one_month_later)
    assert 0.1 < vol < 2.0                                            # a sensible vol


def test_date_string_without_time_zone_works(surface):
    assert 0.1 < surface.vol(80000, "1 Dec 2026") < 2.0


def test_past_date_raises(surface):
    with pytest.raises(ValueError):
        surface.vol(80000, "1 Jan 2020")


def test_several_strikes_at_once(surface):
    vols = surface.vol(np.array([70000, 80000, 90000]), "1 Dec 2026")
    assert len(vols) == 3


def test_surface_plot_draws(surface):
    fig = plot_surface(surface)
    assert len(fig.axes) == 1

"""Tests for the Deribit data pipeline (data/deribit.py, chain.py, cleaning.py).

They use a SAVED snapshot of real Deribit data (tests/fixtures/deribit_btc_options.json),
so they never need the internet and always give the same answer.
"""

import json
from pathlib import Path         # handy way to build file paths

import numpy as np
import pandas as pd
import pytest

from options_pricer.data import deribit
from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


# A "fixture" in pytest = something prepared once and handed to any test that asks for it
@pytest.fixture
def raw():
    """The saved Deribit API responses."""
    with open(FIXTURE) as f:
        return json.load(f)


@pytest.fixture
def chain(raw):
    """The saved data turned into a chain table, with our implied vols added."""
    table = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    return add_implied_vols(table)


# ---------------------------------------------------------------------------
# 1. Building the table
# ---------------------------------------------------------------------------
def test_one_row_per_option(raw, chain):
    assert len(chain) == len(raw["get_book_summary_by_currency"]["result"])


def test_coin_to_usd_conversion(chain):
    # price_usd = price_coin × forward, for every row
    # .to_numpy() turns a pandas column into a plain array, which pytest.approx understands
    expected = (chain["mark_coin"] * chain["forward"]).to_numpy()
    assert chain["mark_usd"].to_numpy() == pytest.approx(expected)


def test_time_to_expiry(chain):
    # Snapshot taken 3 Nov 2026; the 13 Nov expiry is about 10 days away
    row = chain[chain["instrument"] == "BTC-13NOV26-79000-C"].iloc[0]
    assert row["T"] * 365 == pytest.approx(9.7, abs=0.2)


def test_mark_iv_is_decimal(chain):
    # Deribit sends 38.54 (percent); we store 0.3854 (decimal) like the rest of the package
    assert chain["mark_iv"].between(0.05, 3.0).all()


# ---------------------------------------------------------------------------
# 2. Our implied vols vs Deribit's mark_iv
# ---------------------------------------------------------------------------
def test_our_iv_matches_deribit_for_longer_expiries(chain):
    # For OTM options with more than a month left, the match is almost exact (< 0.05 vol points)
    longer = otm_only(chain[chain["T"] > 30 / 365])
    assert len(longer) > 5
    assert longer["iv_diff_pts"].abs().max() < 0.05


def test_our_iv_close_to_deribit_for_all_otm(chain):
    # Short expiries are noisier (the forward moves while Deribit takes its snapshot),
    # but OTM options should still be within half a vol point
    assert otm_only(chain)["iv_diff_pts"].abs().max() < 0.5


def test_missing_bid_gives_no_mid_iv(chain):
    # This option has no bid, so there's no mid price and no mid implied vol
    row = chain[chain["instrument"] == "BTC-25DEC26-220000-C"].iloc[0]
    assert np.isnan(row["iv_mid"])


# ---------------------------------------------------------------------------
# 3. Cleaning
# ---------------------------------------------------------------------------
def test_cleaning_report_adds_up(chain):
    kept, report = clean_chain(chain)
    assert report["rows"].sum() == len(chain)                       # every row accounted for
    assert report.set_index("reason").loc["kept", "rows"] == len(kept)


def test_cleaning_drops_known_bad_rows(chain):
    kept, _ = clean_chain(chain)
    dropped = set(chain["instrument"]) - set(kept["instrument"])
    assert "BTC-25DEC26-220000-C" in dropped      # no bid
    assert "BTC-13NOV26-65000-C" in dropped       # zero open interest


def test_cleaning_catches_crossed_quote(chain):
    broken = chain.copy()
    broken.loc[0, "bid_coin"] = broken.loc[0, "ask_coin"] + 0.01    # make bid > ask on purpose
    broken["mid_coin"] = (broken["bid_coin"] + broken["ask_coin"]) / 2
    _, report = clean_chain(broken)
    assert "crossed_quote" in set(report["reason"])


def test_cleaning_catches_near_expiry(chain):
    nearly_done = chain.copy()
    nearly_done.loc[0, "T"] = 0.5 / 365                               # half a day left
    _, report = clean_chain(nearly_done)
    assert "near_expiry" in set(report["reason"])


def test_otm_only(chain):
    otm = otm_only(chain)
    calls, puts = otm[otm["option_type"] == "call"], otm[otm["option_type"] == "put"]
    assert (calls["strike"] >= calls["forward"]).all()
    assert (puts["strike"] < puts["forward"]).all()


# ---------------------------------------------------------------------------
# 4. Fetching (without the internet, using a fake)
# ---------------------------------------------------------------------------
class FakeResponse:
    """Pretends to be what requests.get() returns."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass                                       # pretend the request worked

    def json(self):
        return self.payload


def test_fetch_uses_result_field(monkeypatch, raw):
    # monkeypatch = temporarily replace a function. Here: replace requests.get with a fake,
    # so fetch_book_summary() gets the saved data instead of going online.
    def fake_get(url, params, timeout):
        assert "get_book_summary_by_currency" in url
        assert params["currency"] == "BTC"
        return FakeResponse(raw["get_book_summary_by_currency"])

    monkeypatch.setattr(deribit.requests, "get", fake_get)
    result = deribit.fetch_book_summary("BTC")
    assert len(result) == len(raw["get_book_summary_by_currency"]["result"])


def test_fetch_raises_on_deribit_error(monkeypatch):
    monkeypatch.setattr(deribit.requests, "get",
                        lambda url, params, timeout: FakeResponse({"error": {"message": "bad request"}}))
    monkeypatch.setattr(deribit.time, "sleep", lambda seconds: None)   # don't actually wait in tests

    with pytest.raises(RuntimeError):
        deribit.fetch_instruments("BTC")

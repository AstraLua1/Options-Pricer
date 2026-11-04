"""Tests for saving and loading daily snapshots (data/storage.py).

tmp_path is a pytest feature: a fresh, empty temporary folder for each test,
so the tests never touch your real data/history/ folder.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.storage import load_svi_history, save_snapshot, saved_days, snapshot_paths
from options_pricer.volatility.svi import fit_svi_by_expiry

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


def one_day():
    """The saved real snapshot -> clean OTM chain + SVI fits, like the collector makes."""
    with open(FIXTURE) as f:
        raw = json.load(f)
    chain = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    clean, _ = clean_chain(chain)
    otm = add_implied_vols(otm_only(clean))
    return otm, fit_svi_by_expiry(otm)


def test_file_names_use_the_date():
    chain_path, svi_path = snapshot_paths("BTC", "2026-11-05 14:30", folder="x")
    assert chain_path.name == "BTC_2026-11-05_chain.csv"
    assert svi_path.name == "BTC_2026-11-05_svi.csv"


def test_save_creates_both_files(tmp_path):
    otm, fits = one_day()
    chain_path, svi_path = save_snapshot("BTC", otm, fits, folder=tmp_path)
    assert chain_path.exists() and svi_path.exists()
    assert svi_path.name == "BTC_2026-11-03_svi.csv"          # the snapshot was taken on 3 Nov 2026


def test_saved_fits_come_back_the_same(tmp_path):
    otm, fits = one_day()
    save_snapshot("BTC", otm, fits, folder=tmp_path)
    history = load_svi_history("BTC", folder=tmp_path)

    assert len(history) == len(fits)
    assert history["rho"].to_numpy() == pytest.approx(fits["rho"].to_numpy())


def test_same_day_twice_keeps_one_snapshot(tmp_path):
    otm, fits = one_day()
    save_snapshot("BTC", otm, fits, folder=tmp_path)
    save_snapshot("BTC", otm, fits, folder=tmp_path)          # run again on the same day
    assert saved_days("BTC", folder=tmp_path) == ["2026-11-03"]


def test_history_stacks_several_days(tmp_path):
    otm, fits = one_day()
    save_snapshot("BTC", otm, fits, folder=tmp_path)

    # Pretend it's the next day: same data, snapshot time moved forward by one day
    next_day = otm.copy()
    next_day["snapshot_time"] = next_day["snapshot_time"] + pd.Timedelta(days=1)
    save_snapshot("BTC", next_day, fits, folder=tmp_path)

    assert saved_days("BTC", folder=tmp_path) == ["2026-11-03", "2026-11-04"]
    assert len(load_svi_history("BTC", folder=tmp_path)) == 2 * len(fits)


def test_empty_history_is_empty_table(tmp_path):
    assert load_svi_history("ETH", folder=tmp_path).empty

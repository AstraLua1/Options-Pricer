"""Tests for the charts (plotting.py), using the saved Deribit snapshot."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")            # draw charts in memory only (no pop-up windows during tests)

from options_pricer.data.chain import add_implied_vols, build_chain   # noqa: E402
from options_pricer.data.cleaning import clean_chain, otm_only        # noqa: E402
from options_pricer.plotting import plot_smiles                       # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "deribit_btc_options.json"


def load_clean_otm_chain():
    with open(FIXTURE) as f:
        raw = json.load(f)
    chain = build_chain(raw["get_book_summary_by_currency"]["result"], raw["get_instruments"]["result"])
    clean, _ = clean_chain(chain)
    return add_implied_vols(otm_only(clean))


def test_one_panel_per_expiry():
    chain = load_clean_otm_chain()
    fig = plot_smiles(chain)

    visible_panels = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_panels) == chain["expiry"].nunique()      # 3 expiries in the snapshot


def test_each_panel_has_dots():
    fig = plot_smiles(load_clean_otm_chain())
    for ax in fig.axes:
        if ax.get_visible():
            assert len(ax.lines) >= 1          # at least the dots (plus the ATM line)

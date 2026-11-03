"""Milestone 3: pull live BTC and ETH options from Deribit and compare our implied vols with Deribit's.

Run from the project folder (PyCharm terminal):
    python scripts/compare_iv.py

What it does, for each currency:
1. Download the option list and prices from Deribit's public API
2. Build a table with USD prices (coin price × forward)
3. Clean it (drop bad quotes) and print WHY rows were dropped
4. Work out our own implied vols and compare them with Deribit's mark_iv
5. Save the full table as a CSV in the data/ folder (not uploaded to GitHub)
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments

DATA_DIR = Path("data")          # top-level data/ folder (ignored by Git)


def summarise_by_expiry(otm):
    """One line per expiry: how many options, the ATM vol, and how close we are to Deribit."""
    rows = []
    for expiry, group in otm.groupby("expiry"):
        # "ATM" option = the one whose strike is closest to the forward
        atm = group.iloc[(group["moneyness"] - 1).abs().argmin()]
        rows.append({
            "expiry": expiry.strftime("%d %b %Y"),
            "days": round(group["T"].iloc[0] * 365, 1),
            "options": len(group),
            "forward": round(group["forward"].iloc[0]),
            "atm_vol_%": round(atm["mark_iv"] * 100, 1),
            "median_diff_pts": round(group["iv_diff_pts"].abs().median(), 3),
            "max_diff_pts": round(group["iv_diff_pts"].abs().max(), 3),
        })
    return pd.DataFrame(rows)


def run(currency):
    print(f"\n{'=' * 70}\n{currency} options\n{'=' * 70}")

    # 1. Download
    summary = fetch_book_summary(currency)
    instruments = fetch_instruments(currency)
    print(f"Downloaded {len(summary)} options")

    # 2. Build the table + our implied vols
    chain = add_implied_vols(build_chain(summary, instruments))

    # 3. Clean
    clean, report = clean_chain(chain)
    print("\nCleaning report (why rows were dropped):")
    print(report.to_string(index=False))

    # 4. Compare on out-of-the-money options only
    otm = otm_only(clean)
    print(f"\nOur implied vol vs Deribit's mark_iv ({len(otm)} clean OTM options):")
    print(summarise_by_expiry(otm).to_string(index=False))

    # 5. Save everything (all rows, so you can see the dropped ones too)
    DATA_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    path = DATA_DIR / f"deribit_{currency}_{stamp}.csv"
    chain.to_csv(path, index=False)
    print(f"\nSaved full table to {path}")


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        run(currency)

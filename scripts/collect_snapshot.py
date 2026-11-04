"""Milestone 5: the snapshot collector.

Run it whenever you want to save a snapshot of the market (one per day is kept):
    python scripts/collect_snapshot.py

For BTC and ETH it:
1. downloads the live options from Deribit
2. cleans them and works out implied vols
3. fits SVI to every expiry (the surface) and runs the arbitrage checks
4. saves the clean chain + the SVI fits to data/history/ with today's date

If one currency fails (e.g. the internet drops), it still tries the other,
and finishes with an error code so you can tell something went wrong.
"""

import sys
from datetime import datetime, timezone

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments
from options_pricer.data.storage import save_snapshot, saved_days
from options_pricer.volatility.arbitrage import check_butterfly, check_calendar
from options_pricer.volatility.svi import fit_svi_by_expiry


def collect(currency):
    """Download, clean, fit and save one currency. Returns a one-line summary."""
    chain = build_chain(fetch_book_summary(currency), fetch_instruments(currency))
    clean, report = clean_chain(chain)
    otm = add_implied_vols(otm_only(clean))
    fits = fit_svi_by_expiry(otm)

    # Record the arbitrage result next to each expiry, so the history shows it too
    fits["butterfly_ok"] = check_butterfly(fits)["ok"].to_numpy()
    calendar = check_calendar(fits)
    arbitrage_ok = bool(fits["butterfly_ok"].all() and calendar["ok"].all())

    chain_path, svi_path = save_snapshot(currency, otm, fits)

    return (f"{currency}: {len(chain)} options downloaded, {len(otm)} kept, "
            f"{len(fits)} expiries fitted, arbitrage {'OK' if arbitrage_ok else 'WARNING'}, "
            f"saved {svi_path.name}. Days collected so far: {len(saved_days(currency))}")


if __name__ == "__main__":
    started = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"--- Snapshot run {started} ---")

    failures = 0
    for currency in ["BTC", "ETH"]:
        try:
            print(collect(currency))
        except Exception as error:                     # keep going even if one currency fails
            failures += 1
            print(f"{currency}: FAILED - {error}")

    sys.exit(1 if failures else 0)                     # 1 = something went wrong (Task Scheduler shows this)

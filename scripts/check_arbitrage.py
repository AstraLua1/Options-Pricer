"""Milestone 4, Step 3: check the fitted SVI curves for arbitrage ("free money").

Run from the project folder (PyCharm terminal):
    python scripts/check_arbitrage.py

For each currency: download live options, clean, fit SVI per expiry, then
1. Butterfly check: does any smile imply a negative probability?
2. Calendar check: is any longer expiry "cheaper" than a shorter one?
"""

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments
from options_pricer.volatility.arbitrage import check_butterfly, check_calendar
from options_pricer.volatility.svi import fit_svi_by_expiry


def pretty(table, date_columns):
    """Format dates as '25 Dec 2026' and round numbers, for printing."""
    table = table.copy()
    for column in date_columns:
        table[column] = table[column].dt.strftime("%d %b %Y")
    return table.round(5).to_string(index=False)


def run(currency):
    print(f"\n{'=' * 70}\n{currency}\n{'=' * 70}")

    # Same as before: download -> clean -> OTM -> implied vols -> SVI fit
    chain = build_chain(fetch_book_summary(currency), fetch_instruments(currency))
    clean, _ = clean_chain(chain)
    fits = fit_svi_by_expiry(add_implied_vols(otm_only(clean)))

    # 1. Butterfly check (each expiry on its own)
    butterfly = check_butterfly(fits)
    print("\nButterfly check (min_g must be >= 0):")
    print(pretty(butterfly, ["expiry"]))

    # 2. Calendar check (each expiry vs the next one)
    calendar = check_calendar(fits)
    print("\nCalendar check (min_growth must be >= 0):")
    print(pretty(calendar, ["shorter", "longer"]))

    # Plain-English verdict
    problems = (~butterfly["ok"]).sum() + (~calendar["ok"]).sum()
    if problems == 0:
        print(f"\nRESULT {currency}: OK - no arbitrage found in any expiry.")
    else:
        print(f"\nRESULT {currency}: WARNING - {problems} problem(s) found, see the rows with ok = False.")


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        run(currency)

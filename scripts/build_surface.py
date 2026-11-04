"""Milestone 4, Step 4: build the full volatility surface for BTC and ETH.

Run from the project folder (PyCharm terminal):
    python scripts/build_surface.py

For each currency: download live options, clean, fit SVI per expiry,
check for arbitrage, join everything into a surface, then:
- print a table of vols for common strikes and dates
- show an example: the vol for a strike and date Deribit doesn't list
- draw the 3D surface (saved in data/ and shown on screen)
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments
from options_pricer.plotting import plot_surface
from options_pricer.volatility.arbitrage import check_butterfly, check_calendar
from options_pricer.volatility.surface import build_surface

DATA_DIR = Path("data")


def run(currency):
    print(f"\n{'=' * 70}\n{currency}\n{'=' * 70}")

    # Download -> clean -> OTM -> implied vols -> surface
    chain = build_chain(fetch_book_summary(currency), fetch_instruments(currency))
    clean, _ = clean_chain(chain)
    surface = build_surface(add_implied_vols(otm_only(clean)))

    # Safety check first: are the curves arbitrage-free?
    ok = check_butterfly(surface.fits)["ok"].all() and check_calendar(surface.fits)["ok"].all()
    print(f"Arbitrage checks: {'OK' if ok else 'WARNING - run check_arbitrage.py for details'}")

    # A table of vols: rows = days to expiry, columns = moneyness
    moneyness = [0.8, 0.9, 1.0, 1.1, 1.2]
    rows = {}
    for days in [7, 30, 90, 180, 365]:
        T = days / 365
        strikes = [m * surface.forward(T) for m in moneyness]
        rows[f"{days} days"] = [round(float(v) * 100, 1) for v in surface.vol_at_T(strikes, T)]
    table = pd.DataFrame(rows, index=[f"{m:.0%} of fwd" for m in moneyness]).T
    print("\nImplied vol (%) by days to expiry and strike:")
    print(table.to_string())

    # Example: a strike and date Deribit doesn't list (about 2 months out, 2% above the forward)
    example_date = surface.snapshot_time + pd.Timedelta(days=60)
    example_strike = round(surface.forward(60 / 365) * 1.02, -3)          # rounded to nearest 1,000
    example_vol = surface.vol(example_strike, example_date)
    print(f"\nExample: strike {example_strike:,.0f} expiring {example_date:%d %b %Y} -> vol {example_vol:.1%}")

    # 3D picture
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    fig = plot_surface(surface, title=f"{currency} volatility surface (Deribit, {stamp} UTC)")
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"surface_{currency}_{stamp}.png"
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        run(currency)
    plt.show()

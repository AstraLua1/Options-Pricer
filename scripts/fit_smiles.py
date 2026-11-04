"""Milestone 4, Step 2: fit an SVI curve through every BTC and ETH smile.

Run from the project folder (PyCharm terminal):
    python scripts/fit_smiles.py

For each currency it downloads live options from Deribit, cleans them,
fits SVI to each expiry, prints the 5 knobs per expiry, and draws the
smiles with the fitted curve on top. Charts are saved in data/ and shown on screen.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments
from options_pricer.plotting import plot_smiles
from options_pricer.volatility.svi import fit_svi_by_expiry

DATA_DIR = Path("data")


def run(currency):
    print(f"\n{'=' * 70}\n{currency}\n{'=' * 70}")

    # Same steps as before: download -> table -> clean -> keep OTM -> implied vols
    chain = build_chain(fetch_book_summary(currency), fetch_instruments(currency))
    clean, _ = clean_chain(chain)
    otm = add_implied_vols(otm_only(clean))

    # NEW: fit SVI to each expiry
    fits = fit_svi_by_expiry(otm)

    # Print the knobs in a readable table
    table = fits.copy()
    table["expiry"] = table["expiry"].dt.strftime("%d %b %Y")
    table["days"] = (table["T"] * 365).round(1)
    columns = ["expiry", "days", "a", "b", "rho", "m", "sigma", "rmse_pts", "points"]
    print(table[columns].round(4).to_string(index=False))

    # Draw the smiles with the fitted curves
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    fig = plot_smiles(otm, title=f"{currency} smiles + SVI fit (Deribit, {stamp} UTC)", fits=fits)

    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"svi_{currency}_{stamp}.png"
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        run(currency)
    plt.show()

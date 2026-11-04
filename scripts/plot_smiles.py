"""Milestone 4, Step 1: plot the real volatility smile for every BTC and ETH expiry.

Run from the project folder (PyCharm terminal):
    python scripts/plot_smiles.py

For each currency it downloads live options from Deribit, cleans them,
keeps the out-of-the-money ones, and draws one small chart per expiry.
The charts are saved as PNG files in the data/ folder and also pop up on screen.
"""

from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

from options_pricer.data.chain import add_implied_vols, build_chain
from options_pricer.data.cleaning import clean_chain, otm_only
from options_pricer.data.deribit import fetch_book_summary, fetch_instruments
from options_pricer.plotting import plot_smiles

DATA_DIR = Path("data")


def run(currency):
    # Same steps as Milestone 3: download -> table -> clean -> keep OTM options
    chain = build_chain(fetch_book_summary(currency), fetch_instruments(currency))
    clean, _ = clean_chain(chain)
    otm = add_implied_vols(otm_only(clean))

    # Draw the smiles
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    fig = plot_smiles(otm, title=f"{currency} volatility smiles (Deribit, {stamp} UTC)")

    # Save as a picture
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / f"smiles_{currency}_{stamp}.png"
    fig.savefig(path, dpi=120)
    print(f"Saved {path}")


if __name__ == "__main__":
    for currency in ["BTC", "ETH"]:
        run(currency)
    plt.show()          # open the charts on screen (close the windows to finish)

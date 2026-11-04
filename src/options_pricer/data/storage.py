"""Save each day's market snapshot to disk, and load the history back.

Folder layout (one pair of files per currency per day):

    data/history/
        BTC_2026-11-05_chain.csv    every clean option: strike, expiry, prices, implied vols
        BTC_2026-11-05_svi.csv      the fitted SVI curve for each expiry (the surface "slices")
        ETH_2026-11-05_chain.csv
        ETH_2026-11-05_svi.csv
        ...

Why keep both?
- The SVI file is small and is enough to rebuild the whole vol surface for that day.
- The chain file is the raw evidence, so a fit can always be re-checked or redone later.

CSV files open in Excel, which makes them easy to check by eye. (Big firms usually
use a faster format such as Parquet, or a database, but the idea is the same.)
"""

from pathlib import Path

import pandas as pd

HISTORY_DIR = Path("data") / "history"


def snapshot_paths(currency, date, folder=HISTORY_DIR):
    """The two file paths for one currency on one date, e.g. BTC on 2026-11-05."""
    day = pd.Timestamp(date).strftime("%Y-%m-%d")
    folder = Path(folder)
    return folder / f"{currency}_{day}_chain.csv", folder / f"{currency}_{day}_svi.csv"


def save_snapshot(currency, chain, fits, folder=HISTORY_DIR):
    """Save one day's clean chain and SVI fits.

    The date in the file name is the day the market data was taken (UTC).
    Running it twice on the same day simply overwrites that day's files,
    so each day keeps exactly one snapshot (the latest one).

    Returns the two paths that were written.
    """
    Path(folder).mkdir(parents=True, exist_ok=True)       # create data/history/ if it's missing
    snapshot_time = chain["snapshot_time"].max()
    chain_path, svi_path = snapshot_paths(currency, snapshot_time, folder)

    chain.to_csv(chain_path, index=False)

    fits = fits.copy()
    fits.insert(0, "snapshot_time", snapshot_time)        # remember when the surface was built
    fits.insert(0, "currency", currency)
    fits.to_csv(svi_path, index=False)

    return chain_path, svi_path


def load_svi_history(currency, folder=HISTORY_DIR):
    """Load every saved day of SVI fits for a currency into one table.

    One row per (day, expiry). Empty table if nothing has been saved yet.
    Project 3 (the risk engine) will use this to see how vols moved day to day.
    """
    files = sorted(Path(folder).glob(f"{currency}_*_svi.csv"))     # all days, oldest first
    if not files:
        return pd.DataFrame()
    days = [pd.read_csv(f, parse_dates=["snapshot_time", "expiry"]) for f in files]
    return pd.concat(days, ignore_index=True)                      # stack them into one table


def saved_days(currency, folder=HISTORY_DIR):
    """List the dates (as 'YYYY-MM-DD' text) that have a saved snapshot."""
    files = sorted(Path(folder).glob(f"{currency}_*_svi.csv"))
    return [f.name.split("_")[1] for f in files]                   # "BTC_2026-11-05_svi.csv" -> "2026-11-05"

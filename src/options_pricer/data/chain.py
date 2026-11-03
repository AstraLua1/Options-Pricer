"""Turn raw Deribit data into a clean option-chain table, and add our own implied vols.

Key fact about Deribit ("inverse" options):
- Prices are quoted in the COIN, not in USD. A BTC call at 0.0450 costs 0.045 BTC.
- To get USD: multiply by the forward price for that expiry ("underlying_price").
      price_usd = price_btc × forward
- Deribit's rate is 0 (the forward already includes the interest), so
  the discount factor is 1. Then the normal Black-76 formula applies in USD.
"""

import numpy as np
import pandas as pd              # tables ("DataFrames"), like Excel in Python

from options_pricer.volatility.implied import black76_implied_vol

MS_PER_YEAR = 365 * 24 * 60 * 60 * 1000     # Deribit timestamps are in milliseconds


def build_chain(book_summary, instruments):
    """Combine the two Deribit API results into one table, with USD prices.

    book_summary: result of fetch_book_summary()   (prices)
    instruments:  result of fetch_instruments()    (strike, expiry, call/put)

    Returns a DataFrame: one row per option.
    """
    prices = pd.DataFrame(book_summary)
    menu = pd.DataFrame(instruments)[["instrument_name", "strike", "option_type", "expiration_timestamp"]]

    # Join the two tables on the option's name, like VLOOKUP in Excel
    df = prices.merge(menu, on="instrument_name", how="inner")

    chain = pd.DataFrame({
        "instrument": df["instrument_name"],
        "currency": df["instrument_name"].str.split("-").str[0],          # "BTC-13NOV26-..." -> "BTC"
        "expiry": pd.to_datetime(df["expiration_timestamp"], unit="ms", utc=True),
        "snapshot_time": pd.to_datetime(df["creation_timestamp"], unit="ms", utc=True),
        "strike": df["strike"].astype(float),
        "option_type": df["option_type"],                                  # "call" or "put"
        # Time to expiry in years, measured from when Deribit made this snapshot
        "T": (df["expiration_timestamp"] - df["creation_timestamp"]) / MS_PER_YEAR,
        "forward": df["underlying_price"].astype(float),                   # futures price for this expiry
        "rate": df["interest_rate"].astype(float),                         # 0 on Deribit
        # Prices in the coin (BTC/ETH). Missing bids come through as NaN.
        "bid_coin": df["bid_price"].astype(float),
        "ask_coin": df["ask_price"].astype(float),
        "mark_coin": df["mark_price"].astype(float),
        "mark_iv": df["mark_iv"].astype(float) / 100,                      # Deribit: 45.2 -> we use 0.452
        "open_interest": df["open_interest"].astype(float),
        "volume": df["volume"].astype(float),
    })

    # Mid price = halfway between bid and ask (NaN if either side is missing)
    chain["mid_coin"] = (chain["bid_coin"] + chain["ask_coin"]) / 2

    # Convert coin prices to USD: price_usd = price_coin × forward
    for name in ["bid", "ask", "mid", "mark"]:
        chain[f"{name}_usd"] = chain[f"{name}_coin"] * chain["forward"]

    # Moneyness: how far the strike is from the forward (1.0 = at the money)
    chain["moneyness"] = chain["strike"] / chain["forward"]

    # Sort nicely: by expiry, then strike, then calls before puts
    return chain.sort_values(["expiry", "strike", "option_type"]).reset_index(drop=True)


def add_implied_vols(chain):
    """Add our own implied vols (from mid and mark prices) and compare with Deribit's mark_iv.

    New columns:
    - iv_mid:       implied vol from the mid USD price   (NaN if no mid or no solution)
    - iv_mark:      implied vol from Deribit's mark price
    - iv_diff_pts:  iv_mark − Deribit's mark_iv, in vol points (0.5 = half a vol point)
    """
    chain = chain.copy()                          # don't change the caller's table

    def solve(price, row):
        """Implied vol for one price on one row (NaN if the price is missing)."""
        if np.isnan(price):
            return np.nan
        df = np.exp(-row["rate"] * row["T"])      # discount factor (= 1 on Deribit)
        return black76_implied_vol(price, row["forward"], row["strike"], row["T"], df, row["option_type"])

    # .apply(..., axis=1) runs the function once per row (fine for a few hundred options)
    chain["iv_mid"] = chain.apply(lambda row: solve(row["mid_usd"], row), axis=1)
    chain["iv_mark"] = chain.apply(lambda row: solve(row["mark_usd"], row), axis=1)
    chain["iv_diff_pts"] = (chain["iv_mark"] - chain["mark_iv"]) * 100

    return chain

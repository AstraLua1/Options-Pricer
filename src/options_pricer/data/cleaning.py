"""Clean an option chain: throw away quotes we can't trust.

Real market data is messy. Before building a vol surface we drop:
- one-sided quotes      (no bid or no ask: we can't get a fair mid price)
- crossed quotes        (bid > ask: impossible, so the data is broken)
- wide spreads          (bid and ask far apart: nobody agrees on the price)
- nearly expired        (less than a day left: vols go haywire)
- illiquid / stale      (nobody holds it, or it's priced at the minimum tick)

Every dropped row gets a REASON, so you can always see why it was removed.
(Risk and valuation teams need this audit trail.)
"""

import pandas as pd


def clean_chain(chain, max_rel_spread=0.25, min_days=1.0, min_open_interest=0.0, min_price_coin=0.0002):
    """Remove untrustworthy quotes.

    chain:             table from build_chain()
    max_rel_spread:    drop if (ask − bid) / mid is above this (0.25 = 25%)
    min_days:          drop if fewer days than this to expiry
    min_open_interest: drop if open interest is not ABOVE this (0 -> drop "nobody holds it")
    min_price_coin:    drop if the mid price is below this in coin (0.0002 BTC = 2 ticks)

    Returns (clean_table, report):
    - clean_table: only the rows we keep
    - report:      table counting how many rows were dropped for each reason
    """
    chain = chain.copy()
    chain["drop_reason"] = ""                     # empty = keep

    rel_spread = (chain["ask_coin"] - chain["bid_coin"]) / chain["mid_coin"]

    # Each rule: (reason, True where the row FAILS the rule).
    # Checked in this order; a row keeps the FIRST reason it fails.
    rules = [
        ("no_bid_or_ask", chain["bid_coin"].isna() | chain["ask_coin"].isna()
                          | (chain["bid_coin"] <= 0) | (chain["ask_coin"] <= 0)),
        ("crossed_quote", chain["bid_coin"] > chain["ask_coin"]),
        ("near_expiry", chain["T"] * 365 < min_days),
        ("too_cheap", chain["mid_coin"] < min_price_coin),
        ("wide_spread", rel_spread > max_rel_spread),
        ("no_open_interest", chain["open_interest"] <= min_open_interest),
    ]

    for reason, fails in rules:
        not_yet_dropped = chain["drop_reason"] == ""
        chain.loc[fails & not_yet_dropped, "drop_reason"] = reason

    kept = chain[chain["drop_reason"] == ""].drop(columns="drop_reason").reset_index(drop=True)

    # Count rows per reason, e.g.  wide_spread: 41, no_bid_or_ask: 22, kept: 600
    report = chain["drop_reason"].replace("", "kept").value_counts().rename_axis("reason").reset_index(name="rows")

    return kept, report


def otm_only(chain):
    """Keep only out-of-the-money options: puts below the forward, calls above it.

    Industry standard for building a smile: OTM options are pure "vol value"
    (no payoff part), so their implied vols are the most reliable.
    """
    is_otm_call = (chain["option_type"] == "call") & (chain["strike"] >= chain["forward"])
    is_otm_put = (chain["option_type"] == "put") & (chain["strike"] < chain["forward"])
    return chain[is_otm_call | is_otm_put].reset_index(drop=True)

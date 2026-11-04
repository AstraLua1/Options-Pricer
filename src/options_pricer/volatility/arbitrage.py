"""Arbitrage checks: make sure the fitted vol curves don't allow "free money".

A curve can fit the dots nicely and still be impossible. Two checks:

1. BUTTERFLY check (one expiry at a time)
   The smile secretly says how likely each BTC price at expiry is.
   Probabilities can never be negative. If the curve bends the wrong way,
   it implies a NEGATIVE probability somewhere -> someone could buy a
   "butterfly" (a cheap 3-option combo) for less than zero = free money.
   Test: Gatheral's g(k) must be >= 0 everywhere.

2. CALENDAR check (comparing expiries)
   More time = more uncertainty, so for the same moneyness, a LONGER option
   can never be cheaper than a SHORTER one.
   Test: total variance w = vol² × T must not go DOWN as expiry gets later.

Both checks only look inside the strike range where we actually had market
data (k_min to k_max): outside it, the curve is just a guess anyway.
"""

import numpy as np
import pandas as pd

from options_pricer.volatility.svi import params_from_row, svi_total_variance

TOLERANCE = 1e-6     # tiny negatives from rounding don't count as arbitrage


def svi_slopes(k, p):
    """First and second derivatives of SVI total variance: w'(k) and w''(k).

    w'  = how steep the curve is at k
    w'' = how much it bends at k
    """
    k = np.asarray(k, dtype=float)
    root = np.sqrt((k - p.m) ** 2 + p.sigma ** 2)
    w1 = p.b * (p.rho + (k - p.m) / root)              # w'(k)
    w2 = p.b * p.sigma ** 2 / root ** 3                # w''(k)
    return w1, w2


def butterfly_g(k, p):
    """Gatheral's g(k). Negative g = negative probability = butterfly arbitrage.

    g(k) = (1 − k·w'/(2w))²  −  (w'²/4)·(1/w + 1/4)  +  w''/2
    """
    w = svi_total_variance(k, p)
    w1, w2 = svi_slopes(k, p)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


def check_butterfly(fits, points=401):
    """Butterfly check for every expiry in a fit_svi_by_expiry() table.

    Returns one row per expiry: the lowest g found, where it was, and ok = True/False.
    """
    rows = []
    for _, fit in fits.iterrows():                       # one fitted expiry at a time
        p = params_from_row(fit)
        k = np.linspace(fit["k_min"], fit["k_max"], points)   # strikes we had data for
        w = svi_total_variance(k, p)
        g = butterfly_g(k, p)
        worst = int(np.argmin(g))
        rows.append({
            "expiry": fit["expiry"],
            "min_total_variance": w.min(),                # must be > 0 too (vol² can't be negative)
            "min_g": g[worst],
            "at_moneyness": np.exp(k[worst]),             # where the worst point is (1.0 = ATM)
            "ok": bool(g.min() >= -TOLERANCE and w.min() > 0),
        })
    return pd.DataFrame(rows)


def check_calendar(fits, points=401):
    """Calendar check between each pair of neighbouring expiries.

    For the same log-moneyness k, total variance must not fall from one expiry
    to the next. Checked over the k range BOTH expiries had data for.
    Returns one row per pair: the biggest drop found (negative = arbitrage) and ok.
    """
    fits = fits.sort_values("T").reset_index(drop=True)
    rows = []
    for i in range(len(fits) - 1):
        short, long = fits.iloc[i], fits.iloc[i + 1]
        low = max(short["k_min"], long["k_min"])          # overlap of the two data ranges
        high = min(short["k_max"], long["k_max"])
        if low >= high:
            continue                                       # no overlap: nothing to compare
        k = np.linspace(low, high, points)
        # how much total variance grows from the shorter to the longer expiry
        growth = svi_total_variance(k, params_from_row(long)) - svi_total_variance(k, params_from_row(short))
        worst = int(np.argmin(growth))
        rows.append({
            "shorter": short["expiry"],
            "longer": long["expiry"],
            "min_growth": growth[worst],                   # negative = longer option "cheaper"
            "at_moneyness": np.exp(k[worst]),
            "ok": bool(growth.min() >= -TOLERANCE),
        })
    return pd.DataFrame(rows)

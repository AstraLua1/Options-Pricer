"""SVI: fit a smooth curve through the volatility smile of one expiry.

The idea in one line: SVI is a curve with 5 "knobs". We turn the knobs until
the curve passes as close as possible to the market's implied vols.

SVI works with two slightly different variables (both standard in industry):
- k = ln(K / F)          "log-moneyness": 0 = at the money, negative = below, positive = above
- w = vol² × T           "total variance": vol squared times years to expiry

The raw SVI formula (Jim Gatheral, 2004):

    w(k) = a + b × ( ρ × (k − m) + √((k − m)² + σ²) )

The 5 knobs:
    a  overall height (vertical shift)
    b  how steep the two sides are
    ρ  tilt: negative = left side higher (crash fear), positive = right side higher
    m  where the bottom of the smile is (left/right shift)
    σ  how rounded the bottom is (small = sharp V, large = round U)

To get back to a normal implied vol:  vol(k) = √( w(k) / T )
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares     # scipy's "turn the knobs to fit the dots" tool

MIN_POINTS = 5        # 5 knobs need at least 5 dots, otherwise the fit is meaningless


@dataclass
class SVIParams:
    """A labelled box for the 5 SVI numbers, e.g. params.rho instead of params[2]."""
    a: float
    b: float
    rho: float
    m: float
    sigma: float


def svi_total_variance(k, p):
    """w(k) = a + b × ( ρ(k − m) + √((k − m)² + σ²) )"""
    k = np.asarray(k, dtype=float)
    return p.a + p.b * (p.rho * (k - p.m) + np.sqrt((k - p.m) ** 2 + p.sigma ** 2))


def svi_implied_vol(k, T, p):
    """Turn SVI total variance back into an ordinary implied vol: √(w / T)."""
    w = svi_total_variance(k, p)
    return np.sqrt(np.maximum(w, 0.0) / T)        # max(..., 0) guards against tiny negatives


def fit_svi(k, vols, T):
    """Find the 5 SVI knobs that best fit one expiry's smile.

    k:    log-moneyness of each option, ln(K/F)
    vols: market implied vols (decimals) for those options
    T:    years to expiry (one number: the same for the whole expiry)

    Returns (params, rmse_vol_points):
    - params: the fitted SVIParams
    - rmse:   typical distance between curve and dots, in vol points (0.3 = 0.3%)
    """
    k = np.asarray(k, dtype=float)
    vols = np.asarray(vols, dtype=float)
    if len(k) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} points to fit SVI, got {len(k)}")

    w_market = vols ** 2 * T                       # market total variance at each strike

    def errors(x):
        """How far the curve is from each dot, in vol terms (what least_squares minimises)."""
        p = SVIParams(*x)
        return svi_implied_vol(k, T, p) - vols

    # Allowed range for each knob:    a            b     rho      m                    sigma
    lower = [-w_market.max(),         0.0,  -0.999,  k.min() - 0.5,       1e-4]
    upper = [ w_market.max(),         10.0,  0.999,  k.max() + 0.5,       2.0]

    # The fit can get stuck depending on where it starts, so try a few starting
    # points and keep the best one. (Common practice for SVI.)
    best = None
    for m0 in [0.0, k.min() / 2, k.max() / 2]:
        for sigma0 in [0.05, 0.2]:
            start = [w_market.min() / 2, 0.1, -0.3, m0, sigma0]
            start = np.clip(start, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
            result = least_squares(errors, start, bounds=(lower, upper))
            if best is None or result.cost < best.cost:
                best = result

    params = SVIParams(*best.x)
    rmse_pts = float(np.sqrt(np.mean(errors(best.x) ** 2)) * 100)
    return params, rmse_pts


def fit_svi_by_expiry(chain, vol_column="iv_mid"):
    """Fit SVI separately for every expiry in a clean, out-of-the-money chain.

    chain: table from clean_chain() + otm_only() + add_implied_vols()
    Returns one row per expiry: the 5 knobs, how good the fit is, and how many dots were used.
    Expiries with fewer than 5 usable dots are skipped.
    """
    rows = []
    for expiry, group in chain.dropna(subset=[vol_column]).groupby("expiry"):
        if len(group) < MIN_POINTS:
            continue                                # not enough dots for 5 knobs
        T = group["T"].iloc[0]
        forward = group["forward"].median()         # one forward per expiry (they differ by a few $)
        k = np.log(group["strike"] / forward)       # log-moneyness
        params, rmse = fit_svi(k, group[vol_column], T)
        rows.append({"expiry": expiry, "T": T, "forward": forward,
                     "a": params.a, "b": params.b, "rho": params.rho,
                     "m": params.m, "sigma": params.sigma,
                     "rmse_pts": rmse, "points": len(group),
                     # the range of strikes (as k) the curve was fitted on:
                     # outside this range the curve is a guess (extrapolation)
                     "k_min": k.min(), "k_max": k.max()})
    return pd.DataFrame(rows)


def params_from_row(row):
    """Rebuild SVIParams from one row of fit_svi_by_expiry()'s table."""
    return SVIParams(row["a"], row["b"], row["rho"], row["m"], row["sigma"])

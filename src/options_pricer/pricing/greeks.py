"""Analytic Black-Scholes Greeks for European options.

All Greeks are returned in RAW units (per 1.00 change in the input):
- delta: change in price per $1 move in spot
- gamma: change in delta per $1 move in spot
- vega:  change in price per 1.00 change in vol (0.20 -> 1.20)
- theta: change in price per 1 YEAR passing (negative = losing value)
- rho:   change in price per 1.00 change in rate (0.05 -> 1.05)

Desks usually quote vega per 1 vol point (vega / 100), theta per day
(theta / 365) and rho per 1% (rho / 100). See to_desk_units() below.
"""

import numpy as np               # maths on single numbers or whole arrays at once
from scipy.stats import norm     # the bell curve: norm.cdf = N(x), norm.pdf = n(x)

from options_pricer.pricing.black_scholes import _d1_d2   # reuse the same d1/d2 as the price


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Return delta, gamma, vega, theta and rho as a dictionary.

    Same inputs as bs_price:
    S: spot, K: strike, T: years to expiry, r: rate,
    sigma: vol, q: dividend yield, option_type: "call" or "put"
    """
    # Convert inputs to numpy float arrays (numbers, lists or arrays all work)
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))

    # Rebuild the same forward and discount factor as bs_price,
    # so d1 and d2 are exactly the ones used for the price
    F = S * np.exp((r - q) * T)          # F = S·e^((r−q)T)
    d1, d2 = _d1_d2(F, K, T, sigma)

    # Building blocks used by several Greeks
    sqrt_T = np.sqrt(T)                  # √T
    df_r = np.exp(-r * T)                # e^(−rT): discounting for the strike (cash) leg
    df_q = np.exp(-q * T)                # e^(−qT): dividend adjustment for the spot leg
    pdf_d1 = norm.pdf(d1)                # n(d1): HEIGHT of the bell curve at d1

    # ---- Same for calls and puts --------------------------------------
    # Gamma = e^(−qT)·n(d1) / (S·σ·√T)
    #   How fast delta changes. Biggest at-the-money, near expiry.
    gamma = df_q * pdf_d1 / (S * sigma * sqrt_T)

    # Vega = S·e^(−qT)·n(d1)·√T
    #   Price change for a 1.00 change in vol. Biggest at-the-money, long expiry.
    vega = S * df_q * pdf_d1 * sqrt_T

    # First part of theta (shared): the "time decay" of the option's vol value
    #   −S·e^(−qT)·n(d1)·σ / (2√T)
    theta_decay = -S * df_q * pdf_d1 * sigma / (2 * sqrt_T)

    # ---- Different for calls and puts ---------------------------------
    if option_type == "call":
        # Delta = e^(−qT)·N(d1)               -> between 0 and 1
        delta = df_q * norm.cdf(d1)

        # Theta = decay − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)
        #   − r·K... : the strike you'll pay gets "less discounted" as time passes
        #   + q·S... : you miss fewer dividends as time passes
        theta = theta_decay - r * K * df_r * norm.cdf(d2) + q * S * df_q * norm.cdf(d1)

        # Rho = K·T·e^(−rT)·N(d2)             -> calls gain when rates rise
        rho = K * T * df_r * norm.cdf(d2)

    elif option_type == "put":
        # Delta = −e^(−qT)·N(−d1)             -> between −1 and 0
        delta = -df_q * norm.cdf(-d1)

        # Theta = decay + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)
        #   (mirror image of the call: the put RECEIVES the strike)
        theta = theta_decay + r * K * df_r * norm.cdf(-d2) - q * S * df_q * norm.cdf(-d1)

        # Rho = −K·T·e^(−rT)·N(−d2)           -> puts lose when rates rise
        rho = -K * T * df_r * norm.cdf(-d2)

    else:
        # Fail loudly on typos instead of returning wrong numbers
        raise ValueError("option_type must be either 'call' or 'put'")

    # A dictionary lets you look Greeks up by name: greeks["delta"]
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def to_desk_units(greeks):
    """Convert raw Greeks into the units traders and risk reports use.

    vega  -> per 1 vol point (20% -> 21%)
    theta -> per calendar day
    rho   -> per 1% move in rates
    Delta and gamma are unchanged.
    """
    return {
        "delta": greeks["delta"],
        "gamma": greeks["gamma"],
        "vega": greeks["vega"] / 100,     # 1 vol point = 0.01
        "theta": greeks["theta"] / 365,   # 1 day = 1/365 of a year
        "rho": greeks["rho"] / 100,       # 1% = 0.01
    }

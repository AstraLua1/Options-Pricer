"""Analytic Black-Scholes Greeks for European options.

All Greeks are returned in RAW units (per 1.00 change in the input):
- delta: change in price per $1 move in spot
- gamma: change in delta per $1 move in spot
- vega:  change in price per 1.00 change in vol (0.20 -> 1.20)
- theta: change in price per 1 YEAR passing (negative = losing value)
- rho:   change in price per 1.00 change in rate (0.05 -> 1.05)

Desks usually quote vega per 1 vol point (vega / 100), theta per day
(theta / 365) and rho per 1% (rho / 100). See to_desk_units() below.

Edge case: with no time value left (T = 0 or sigma = 0) the bell curve
becomes a "cliff": N(d1) and N(d2) become 1 if in the money and 0 if not,
and n(d1) becomes 0. So delta is 0 or 1, and gamma and vega are 0.
"""

import numpy as np               # maths on single numbers or whole arrays at once
from scipy.stats import norm     # the bell curve: norm.cdf = N(x), norm.pdf = n(x)

# Reuse the SAME helpers as the pricer, so price and Greeks can never disagree
from options_pricer.pricing.black_scholes import MIN_TOTAL_VOL, _check_inputs, _d1_d2


def bs_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Return delta, gamma, vega, theta and rho as a dictionary.

    Same inputs as bs_price:
    S: spot, K: strike, T: years to expiry, r: rate,
    sigma: vol, q: dividend yield, option_type: "call" or "put"
    """
    # Convert inputs to numpy float arrays (numbers, lists or arrays all work)
    S, K, T, r, sigma, q = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma, q))

    # Rebuild the same forward and discount factors as bs_price
    F = S * np.exp((r - q) * T)          # F = S·e^((r−q)T)
    df_r = np.exp(-r * T)                # e^(−rT): discounting for the strike (cash) leg
    df_q = np.exp(-q * T)                # e^(−qT): dividend adjustment for the spot leg

    # Same input checks as the pricer (BEFORE √T, which can't handle negative T)
    _check_inputs(F, K, T, sigma, df_r)

    sqrt_T = np.sqrt(T)                  # √T

    # Which options have no time value left? (True/False for each option)
    no_time_value = sigma * sqrt_T < MIN_TOTAL_VOL

    # 1.0 where the call is in the money (F > K), else 0.0
    call_itm = (F > K).astype(float)

    # Silence divide-by-zero warnings: those values get replaced below anyway
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = _d1_d2(F, K, T, sigma)

        # The building blocks. With no time value, use the "cliff" versions.
        N_d1 = np.where(no_time_value, call_itm, norm.cdf(d1))     # N(d1)
        N_d2 = np.where(no_time_value, call_itm, norm.cdf(d2))     # N(d2)
        pdf_d1 = np.where(no_time_value, 0.0, norm.pdf(d1))        # n(d1): HEIGHT of the bell curve

        # ---- Same for calls and puts ----------------------------------
        # Gamma = e^(−qT)·n(d1) / (S·σ·√T)
        #   How fast delta changes. Biggest at-the-money, near expiry.
        gamma = np.where(no_time_value, 0.0, df_q * pdf_d1 / (S * sigma * sqrt_T))

        # Vega = S·e^(−qT)·n(d1)·√T
        #   Price change for a 1.00 change in vol. Biggest at-the-money, long expiry.
        vega = S * df_q * pdf_d1 * sqrt_T

        # Shared part of theta: the "time decay" of the option's vol value
        #   −S·e^(−qT)·n(d1)·σ / (2√T)
        theta_decay = np.where(no_time_value, 0.0, -S * df_q * pdf_d1 * sigma / (2 * sqrt_T))

    # N(−x) = 1 − N(x): the put versions of the building blocks
    N_minus_d1 = 1.0 - N_d1
    N_minus_d2 = 1.0 - N_d2

    # ---- Different for calls and puts -----------------------------------
    if option_type == "call":
        # Delta = e^(−qT)·N(d1)               -> between 0 and 1
        delta = df_q * N_d1

        # Theta = decay − r·K·e^(−rT)·N(d2) + q·S·e^(−qT)·N(d1)
        #   − r·K... : the strike you'll pay gets "less discounted" as time passes
        #   + q·S... : you miss fewer dividends as time passes
        theta = theta_decay - r * K * df_r * N_d2 + q * S * df_q * N_d1

        # Rho = K·T·e^(−rT)·N(d2)             -> calls gain when rates rise
        rho = K * T * df_r * N_d2

    elif option_type == "put":
        # Delta = −e^(−qT)·N(−d1)             -> between −1 and 0
        delta = -df_q * N_minus_d1

        # Theta = decay + r·K·e^(−rT)·N(−d2) − q·S·e^(−qT)·N(−d1)
        #   (mirror image of the call: the put RECEIVES the strike)
        theta = theta_decay + r * K * df_r * N_minus_d2 - q * S * df_q * N_minus_d1

        # Rho = −K·T·e^(−rT)·N(−d2)           -> puts lose when rates rise
        rho = -K * T * df_r * N_minus_d2

    else:
        # Fail loudly on typos instead of returning wrong numbers
        raise ValueError("option_type must be either 'call' or 'put'")

    # A dictionary lets you look Greeks up by name: greeks["delta"]
    # [()] turns single-number arrays back into plain numbers (arrays stay arrays)
    return {
        "delta": delta[()],
        "gamma": gamma[()],
        "vega": vega[()],
        "theta": theta[()],
        "rho": rho[()],
    }


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

"""Black-Scholes and Black-76 option pricing.

Conventions used throughout this package:
- sigma: annualised volatility as a decimal (0.20 means 20%)
- T: time to expiry in years (30 days = 30/365)
- r, q: continuously compounded annual rates as decimals
- Prices come out in the same currency as F, S and K.
"""

import numpy as np               # maths on single numbers or whole arrays at once
from scipy.stats import norm     # the bell curve; norm.cdf(x) = N(x) in the formula


def _d1_d2(F, K, T, sigma):
    """Return d1 and d2 from the Black-76 formula.

    The leading underscore means "internal helper": the price and
    every Greek need d1 and d2, so we write the formula only once.
    """
    # σ√T = total uncertainty between now and expiry
    # (vol is per year; uncertainty grows with the square root of time)
    total_vol = sigma * np.sqrt(T)

    # d1 = [ln(F/K) + ½σ²T] / (σ√T)
    #   np.log(F / K)         -> how far in/out of the money, in % terms
    #   0.5 * total_vol**2    -> ½σ²T, because (σ√T)² = σ²T
    #   / total_vol           -> turns it into "number of standard deviations"
    d1 = (np.log(F / K) + 0.5 * total_vol**2) / total_vol

    # d2 = d1 − σ√T
    d2 = d1 - total_vol

    return d1, d2                # send both back; caller writes: d1, d2 = _d1_d2(...)


def black76_price(F, K, T, sigma, df, option_type="call"):
    """Price a European option from the forward price.

    F: forward price of the underlying at expiry
    K: strike
    T: time to expiry in years
    sigma: volatility (decimal)
    df: discount factor from expiry back to today
    option_type: "call" or "put"
    """
    # Turn every input into a numpy array of floats, so you can pass a
    # single number (100), a list ([90, 100, 110]) or an array and the maths
    # works the same. Same as writing F = np.asarray(F, dtype=float) etc. 5 times.
    # Note: this does NOT fix units. sigma=20 stays 20.0, not 0.20.
    F, K, T, sigma, df = (np.asarray(x, dtype=float) for x in (F, K, T, sigma, df))

    # Get d1 and d2 from the functionbs above
    d1, d2 = _d1_d2(F, K, T, sigma)

    if option_type == "call":
        # Call = DF × [ F·N(d1) − K·N(d2) ]
        #   K * norm.cdf(d2)  -> average strike you PAY (N(d2) = prob. call ends in the money)
        #   F * norm.cdf(d1)  -> average value of the asset you RECEIVE when you exercise
        #   df * (...)        -> payoff happens at expiry, so discount it back to today
        return df * (F * norm.cdf(d1) - K * norm.cdf(d2))

    elif option_type == "put":
        # Put = DF × [ K·N(−d2) − F·N(−d1) ]   (mirror image of the call)
        #   K * norm.cdf(-d2) -> average strike you RECEIVE (N(−d2) = prob. put ends in the money)
        #   F * norm.cdf(-d1) -> average value of the asset you GIVE UP
        #   N(−x) = 1 − N(x) because the bell curve is symmetric
        return df * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

    else:
        # Fail loudly on typos like "Call" or "c" instead of returning a wrong price
        raise ValueError("option_type must be either 'call' or 'put'")


def bs_price(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Price a European option from the spot price (classic Black-Scholes).

    S: spot price today
    K: strike
    T: time to expiry in years
    r: risk-free rate (decimal)
    sigma: volatility (decimal)
    q: dividend yield (decimal, 0 if none)
    option_type: "call" or "put"

    Converts spot inputs into a forward and a discount factor,
    then uses black76_price so the formula lives in one place.
    """
    # Same conversion as above: numbers, lists or arrays -> numpy float arrays
    S, T, r, q = (np.asarray(x, dtype=float) for x in (S, T, r, q))

    # Forward price: F = S·e^((r−q)T)
    # Holding the asset costs interest (r) but pays dividends (q),
    # so the fair price for delivery at expiry grows at (r − q)
    F = S * np.exp((r - q) * T)

    # Discount factor: df = e^(−rT)
    # What $1 received at expiry is worth today (e.g. r=5%, T=1 -> 0.951)
    df = np.exp(-r * T)

    # Hand off to the core formula. In Milestone 3 (Deribit) we'll call
    # black76_price directly with the market's F and skip this wrapper.
    return black76_price(F, K, T, sigma, df, option_type)

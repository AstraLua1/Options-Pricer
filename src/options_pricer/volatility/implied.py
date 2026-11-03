"""Implied volatility: find the vol that makes Black-Scholes match a market price.

Pricing goes vol -> price. Implied vol goes the other way: price -> vol.
There's no formula for that, so we SEARCH:

1. Newton's method (fast): guess a vol, see how far the price is off,
   use vega to jump to a better guess. Usually done in about 5 steps.
2. Brent's method (safe backup): if Newton misbehaves, trap the answer
   between a low vol and a high vol and keep narrowing the gap.
   Slower, but it can't fail if an answer exists.

Edge cases handled:
- Prices outside what ANY vol could produce -> nan ("bad quote, drop it")
- Price equal to the payoff value -> vol 0
- Deep in-the-money options -> solved on the matching out-of-the-money
  option instead (same vol, but much easier to solve accurately)
"""

import numpy as np                    # maths
from scipy.optimize import brentq     # Brent's method, ready-made in scipy
from scipy.stats import norm          # bell curve: norm.pdf = n(x)

from options_pricer.pricing.black_scholes import _d1_d2, black76_price

VOL_LOW = 1e-6         # lowest vol we search (0.0001%)
VOL_HIGH = 10.0        # highest vol we search (1000%; crypto can be wild!)
PRICE_TOL = 1e-12      # prices this close to the bounds count as "on" the bound (per $1 of forward)
VOL_TOL = 1e-10        # stop Newton once a step changes the vol by less than this


def _black76_vega(F, K, T, sigma, df):
    """Vega for the forward-based formula: DF·F·n(d1)·√T.

    Newton needs this: it tells us how much the price moves per 1.00 of vol.
    """
    d1, _ = _d1_d2(F, K, T, sigma)            # "_" = we don't need d2 here
    return df * F * norm.pdf(d1) * np.sqrt(T) # formula for vega refer to above


def black76_implied_vol(price, F, K, T, df, option_type="call",
                        newton_max_iter=50, return_info=False):
    """Implied vol from a market price, using the forward-based formula.

    price: market price of the option (same currency as F and K)
    F, K, T, df: forward, strike, years to expiry, discount factor
    option_type: "call" or "put"
    newton_max_iter: max Newton steps before switching to Brent
    return_info: if True, also return a dict saying which method was used

    Returns the vol as a decimal (0.20 = 20%), or nan if no vol fits the price.
    """
    # ---- 1. Check the inputs (error handling) ----
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be either 'call' or 'put'")
    if T <= 0:
        raise ValueError("Implied vol needs time to expiry > 0")
    if F <= 0 or K <= 0 or df <= 0:
        raise ValueError("F, K and df must be positive")

    info = {"method": None, "iterations": 0}

    def done(vol, method):
        """Small helper so every exit returns the same way."""
        info["method"] = method
        return (vol, info) if return_info else vol

    # ---- 2. Switch deep ITM options to their OTM twin -------------------
    # An ITM call is mostly payoff plus a little "vol value". Solving on that
    # little piece is fragile. Put-call parity gives the matching OTM put,
    # which is ALL vol value, and has the same implied vol:
    #   Call − Put = DF·(F − K)   ->   Put = Call − DF·(F − K)
    # ps this is because deep ITM options is difficult to solve accurately
    if option_type == "call" and F > K:
        price = price - df * (F - K)
        option_type = "put"
    elif option_type == "put" and F < K:
        price = price + df * (F - K)
        option_type = "call"

    # ---- 3. Is the price even possible? ----------------------------------
    # Now the option is OTM, so with zero vol it's worth 0,
    # and with infinite vol it's worth DF·F (call) or DF·K (put).
    lower = 0.0
    upper = df * F if option_type == "call" else df * K
    tol = PRICE_TOL * F                        # scale "close enough" to the price level

    if price < lower - tol or price > upper + tol:
        return done(np.nan, "no_solution")    # impossible quote: no vol can produce it
    if price <= lower + tol:
        return done(0.0, "intrinsic")         # worth only its payoff -> vol 0

    # The function we want to be zero: model price minus market price
    def price_error(sigma):
        return black76_price(F, K, T, sigma, df, option_type) - price

    # ---- 4. Newton's method ---------------------------------------------
    # Starting guess. Two candidates, take the larger:
    #   a) desk rule of thumb for ATM: price ≈ 0.4 × DF × F × σ × √T  ->  σ ≈ price / (0.4·DF·F·√T)
    #   b) σ = √(2·|ln(F/K)| / T): the "turning point" of the price curve. Starting
    #      here (or above) stops Newton from overshooting on OTM options.
    guess_atm = price / (0.4 * df * F * np.sqrt(T))
    guess_otm = np.sqrt(2 * abs(np.log(F / K)) / T)
    sigma = min(max(guess_atm, guess_otm, 0.05), VOL_HIGH / 2)   # keep it in a sensible range

    for i in range(newton_max_iter): #newton_max_iter is set to 50 btw
        info["iterations"] = i + 1

        error = price_error(sigma)                    # how far off is our price?
        vega = _black76_vega(F, K, T, sigma, df)      # how fast does price move with vol?
        if vega < 1e-12:                              # flat spot: Newton would jump wildly
            break

        # Newton step: new guess = old guess − (how far off) / (how fast price moves)
        step = error / vega
        sigma = sigma - step

        if not (VOL_LOW < sigma < VOL_HIGH):          # jumped outside the sensible range
            break
        if abs(step) < VOL_TOL:                       # guesses stopped changing: done!
            return done(float(sigma), "newton")

    # ---- 5. Brent's method (backup) --------------------------------------
    # Brent needs the answer trapped: error < 0 at the low vol, > 0 at the high vol
    if price_error(VOL_HIGH) < 0:
        return done(np.nan, "no_solution")     # even 1000% vol isn't enough
    vol = brentq(price_error, VOL_LOW, VOL_HIGH, xtol=1e-12)
    return done(float(vol), "brent")


def bs_implied_vol(price, S, K, T, r, q=0.0, option_type="call",
                   newton_max_iter=50, return_info=False):
    """Implied vol from a market price, using spot inputs (classic Black-Scholes).

    Converts spot to a forward and a discount factor, then calls black76_implied_vol.
    """
    F = S * np.exp((r - q) * T)      # F = S·e^((r−q)T)
    df = np.exp(-r * T)              # df = e^(−rT)
    return black76_implied_vol(price, F, K, T, df, option_type,
                               newton_max_iter=newton_max_iter, return_info=return_info)

"""Numerical ("bump-and-reprice") Greeks.

Idea: nudge one input a tiny bit up and down, re-price the option each time,
and measure how much the price changed. No Greek formulas needed - only the
pricer. This is how risk systems get Greeks for products with no neat formula,
and it's how we check that our analytic formulas in greeks.py are right.

Uses CENTRAL differences (bump up AND down), which are far more accurate
than bumping in one direction only.
"""

from options_pricer.pricing.black_scholes import bs_price   # the only thing we need: a pricer


def numerical_greeks(S, K, T, r, sigma, q=0.0, option_type="call"):
    """Return delta, gamma, vega, theta and rho by bump-and-reprice.

    Same inputs and same RAW units as bs_greeks, so the two can be compared.
    """

    # Small helper: price the option, changing only the inputs we pass in.
    # Everything we don't mention keeps its original value.
    #   e.g. price(S=101) -> same option but with spot 101
    def price(S=S, T=T, r=r, sigma=sigma):
        return bs_price(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)

    # ---- Bump sizes ----------------------------------------------------
    # Too big  -> the straight-line approximation is poor (truncation error)
    # Too small -> computer rounding noise dominates (round-off error)
    # These are sensible middle-ground values.
    h_S = S * 1e-4        # spot bump for delta: 0.01% of spot
    h_S2 = S * 1e-3       # spot bump for gamma: bigger, because gamma divides by h²
    h_sigma = 1e-4        # vol bump: 0.01 vol points
    h_T = 1e-4            # time bump: about 0.04 days
    h_r = 1e-4            # rate bump: 1 basis point

    # ---- Delta: slope of price vs spot ------------------------------------
    # (P(S+h) − P(S−h)) / 2h   = "rise over run" across a tiny window around S
    delta = (price(S=S + h_S) - price(S=S - h_S)) / (2 * h_S)

    # ---- Gamma: curvature of price vs spot -------------------------------
    # (P(S+h) − 2·P(S) + P(S−h)) / h²   = how much the slope itself changes
    gamma = (price(S=S + h_S2) - 2 * price() + price(S=S - h_S2)) / h_S2**2

    # ---- Vega: slope of price vs vol ------------------------------------
    vega = (price(sigma=sigma + h_sigma) - price(sigma=sigma - h_sigma)) / (2 * h_sigma)

    # ---- Theta: change in price as time PASSES ---------------------------
    # Time passing means T gets SMALLER, hence the minus sign in front
    theta = -(price(T=T + h_T) - price(T=T - h_T)) / (2 * h_T)

    # ---- Rho: slope of price vs interest rate ----------------------------
    rho = (price(r=r + h_r) - price(r=r - h_r)) / (2 * h_r)

    # Same dictionary shape as bs_greeks, so tests can compare them key by key
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

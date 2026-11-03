# Options Pricer

Options Pricer: Black-Scholes, implied vol and crypto vol surfaces from public Deribit data.

A learning project in Python, built step by step to understand how banks and trading firms price options and measure their risk. Uses public data only.

## Roadmap

| Milestone | What | Status |
|---|---|---|
| 1 | Black-Scholes pricer + Greeks (delta, gamma, vega, theta, rho), with tests | ✅ Done |
| 2 | Implied volatility solver (Newton's method, Brent's method as a fallback) | ✅ Done |
| 3 | Pull BTC and ETH option data from Deribit's public API, clean it, compute implied vols | ⏳ Next |
| 4 | Plot the vol smile, fit SVI, build a vol surface, check for arbitrage | |
| 5 | Daily snapshot collector that saves the surface to disk | |

## Project structure

```
src/options_pricer/
├── pricing/
│   ├── black_scholes.py   # option prices (forward-based core + spot wrapper)
│   ├── greeks.py          # analytic Greeks + conversion to desk units
│   └── numerical.py       # "bump-and-reprice" Greeks, used to check the formulas
├── volatility/
│   └── implied.py         # implied vol solver: Newton first, Brent as a backup
└── data/                  # (Milestone 3)
tests/                     # pytest tests
practice/                  # interview practice: Black-Scholes, Greeks and Newton from memory
notebooks/                 # demos only
```

## Install

```
python -m venv .venv
pip install -e ".[dev]"
```

## Quick example

```python
from options_pricer.pricing.black_scholes import bs_price
from options_pricer.pricing.greeks import bs_greeks, to_desk_units

# Spot 100, strike 100, 1 year, 5% rate, 20% vol
bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2)                      # call ≈ 10.4506
bs_price(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type="put")   # put  ≈ 5.5735

greeks = to_desk_units(bs_greeks(S=100, K=100, T=1, r=0.05, sigma=0.2))
# delta ≈ 0.637, vega ≈ 0.375 per vol point, theta ≈ -0.0176 per day

# The other way round: market price -> implied vol
from options_pricer.volatility.implied import bs_implied_vol
bs_implied_vol(10.4506, S=100, K=100, T=1, r=0.05)                  # ≈ 0.20 (20% vol)
bs_implied_vol(150, S=100, K=100, T=1, r=0.05)                      # nan: impossible price
```

Inputs can be single numbers or arrays, so a whole option chain can be priced in one call.

Units: volatility and rates as decimals (20% = 0.20), time in years (30 days = 30/365).

## Tests

```
pytest
```

The tests check:

- known textbook prices
- put-call parity
- the Greek formulas against "bump-and-reprice" Greeks (two independent methods must agree)
- edge cases: expiry, zero volatility and impossible inputs
- implied vol "round trip": price an option at a known vol, solve back, get the same vol (160 scenarios, from deep in the money to deep out of the money, 1 week to 5 years, 5% to 200% vol)

## What I learned

### Milestone 1: Black-Scholes and the Greeks

**How the Greeks behave in the money, at the money and out of the money.**
- Delta is highest deep in the money (close to 1) because the option is almost sure to be exercised and behaves like the stock. At the money it sits around 0.5, and out of the money it's close to 0.
- Gamma (how fast delta changes) is highest at the money, because that's where a small move in the stock changes the odds of exercising the most. Deep in or out of the money, gamma is close to 0.
- Vega is also highest at the money, and it grows with time to expiry.
- I expected gamma and theta to shrink near expiry, but for at-the-money options they shoot up. Vega does the opposite and fades to zero.

**Error handling: why one `nan` matters.**
- With no time left (or zero volatility), the Black-Scholes formula divides by zero and returns `nan`.
- One `nan` makes the total of a whole portfolio `nan`, which would break a risk report.
- The fix: when there's no time or volatility left, the option is just worth its payoff (stock minus strike for a call, strike minus stock for a put). Impossible inputs, like negative time, raise a clear error instead of giving a wrong number.

### Milestone 2: Implied volatility

**What implied vol is.**
- Pricing goes vol → price. Implied vol goes the other way: given the market price (plus S, K, T and r), find the vol that makes Black-Scholes match it.
- A higher price always means a higher implied vol, because vega is always positive. So every valid price has exactly one implied vol.

**How the solver finds it.**
- There's no formula, so the computer guesses and improves.
- Newton's method is the default: guess a vol, see how far off the price is, and use vega to decide how big a correction to make. It usually finishes in about 5 steps.
- Brent's method is the failsafe: it traps the answer between a very low and a very high vol and keeps narrowing the gap. Slower, but it always works if an answer exists.
- Deep in-the-money options are hard to solve accurately, so the solver switches to the matching out-of-the-money option, which has the same implied vol.

**Impossible prices.**
- A call can never be worth more than the stock, and an option can never be worth less than its payoff. If a quote breaks these rules, no vol can produce it, so the solver returns `nan` as a "bad quote" flag. This will matter for cleaning real Deribit data.

**What implied vol is used for.**
- Traders quote and compare options in vol rather than dollars.
- Plotting it across strikes and expiries gives the vol smile and vol surface (Milestone 4).
- Banks use market implied vols to check traders' valuations (independent price verification).

## Disclaimer

For learning purposes only. 

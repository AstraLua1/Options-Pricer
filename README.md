# Options Pricer

Options Pricer: Black-Scholes, implied vol and crypto vol surfaces from public Deribit data.

A learning project in Python, built step by step to understand how banks and trading firms price options and measure their risk. Uses public data only.

## Roadmap

| Milestone | What | Status |
|---|---|---|
| 1 | Black-Scholes pricer + Greeks (delta, gamma, vega, theta, rho), with tests | ✅ Done |
| 2 | Implied volatility solver (Newton's method, Brent's method as a fallback) | ✅ Done |
| 3 | Pull BTC and ETH option data from Deribit's public API, clean it, compute implied vols | ✅ Done |
| 4 | Plot the vol smile, fit SVI, build a vol surface, check for arbitrage | ✅ Done |
| 5 | Snapshot collector that saves the surface to disk (run on demand) | ✅ Done |

## Project structure

```
src/options_pricer/
├── pricing/
│   ├── black_scholes.py   # option prices (forward-based core + spot wrapper)
│   ├── greeks.py          # analytic Greeks + conversion to desk units
│   └── numerical.py       # "bump-and-reprice" Greeks, used to check the formulas
├── volatility/
│   ├── implied.py         # implied vol solver: Newton first, Brent as a backup
│   ├── svi.py             # fits a smooth SVI curve through each expiry's smile
│   ├── arbitrage.py       # butterfly and calendar checks (no "free money" in the curves)
│   └── surface.py         # joins the curves into a surface: a vol for any strike and date
├── plotting.py            # smile charts and the 3D surface picture
└── data/
    ├── deribit.py         # downloads option data from Deribit's public API
    ├── chain.py           # builds one table per currency, converts BTC prices to USD, adds implied vols
    ├── cleaning.py        # drops untrustworthy quotes and records why
    └── storage.py         # saves each day's surface to data/history/ and loads the history back
scripts/
├── compare_iv.py          # live run: download, clean, compare our vols with Deribit's
├── plot_smiles.py         # live run: plot the smile for every expiry
├── fit_smiles.py          # live run: fit SVI and draw the curves on the smiles
├── check_arbitrage.py     # live run: butterfly and calendar checks
├── build_surface.py       # live run: build the surface, print a vol table, draw it in 3D
└── collect_snapshot.py    # live run: save today's BTC and ETH surfaces to data/history/
tests/                     # pytest tests
└── fixtures/              # a saved real Deribit snapshot, so tests don't need the internet
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

# The volatility surface (after building it from Deribit data, see scripts/build_surface.py)
surface.vol(79000, "10 Jan 2027")                                  # e.g. 0.383 (38.3%)
```

Inputs can be single numbers or arrays, so a whole option chain can be priced in one call.

Units: volatility and rates as decimals (20% = 0.20), time in years (30 days = 30/365).

## Live Deribit data

```
python scripts/compare_iv.py
```

Downloads every live BTC and ETH option, prints a cleaning report and compares our implied vols with Deribit's, expiry by expiry. The full table is saved as a CSV in `data/` (not uploaded to GitHub).

```
python scripts/build_surface.py
```

Builds the full BTC and ETH volatility surfaces: fits SVI to every expiry, checks for arbitrage, prints a table of vols by strike and days to expiry, and draws the surface in 3D.

```
python scripts/collect_snapshot.py
```

Saves a snapshot of today's BTC and ETH surfaces to `data/history/` (one per day; running it again the same day replaces it). Run it whenever you want to add a day to the history.

## Tests

```
pytest
```

The tests check:

- known textbook prices
- put-call parity
- the Greek formulas against "bump-and-reprice" Greeks (two independent methods must agree)
- edge cases: expiry, zero volatility and impossible inputs
- the Deribit pipeline, using a saved real snapshot: BTC-to-USD conversion, cleaning rules, and our vols matching Deribit's
- saving and loading snapshots (in a temporary folder, so real data is never touched)
- SVI fits, arbitrage checks (including a famous "bad" curve that must be caught) and the surface
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

### Milestone 3: Real market data from Deribit

**Pulling data from an API.**
- An API is a website for programs: you send a request to a web address with some parameters and get raw data (JSON) back.
- In Python that's `requests.get(url, params=...)`, then `.json()`, then take the `"result"` part.
- Two requests were needed: the "menu" (`get_instruments`: strike, expiry, call/put) and the "prices" (`get_book_summary_by_currency`: bid, ask, mark price, Deribit's implied vol, forward). They're joined into one table, one row per option.
- Real-world extras: a timeout so the program never hangs, and retries if the connection fails.

**Deribit prices are in BTC, not USD.**
- A price of 0.0354 means 0.0354 BTC. To get dollars, multiply by the BTC futures price for that expiry: 0.0354 × $77,834 ≈ $2,759.

**Cleaning real data.**
- Real quotes are messy: some options have no bid, some spreads are very wide, some nobody holds.
- These are dropped before using the data, and every dropped row gets a reason, so there's an audit trail.

**Validating my implied vol solver on real prices.**
- Deribit gives every input (strike, expiry, forward, option price) and also its own implied vol.
- I fed the inputs into my own solver and compared. Example: the 25 Dec 2026 $82,000 call priced at $2,759 gives 37.88%, the same as Deribit's.
- For longer expiries the match was within about 0.002 vol points. Very short expiries and in-the-money options matched less closely, so out-of-the-money options are the most reliable for building a vol surface.

### Milestone 4: The volatility smile and surface

**Vol isn't one number.**
- In Black-Scholes, vol is a single number. In the real market every strike has a different implied vol. Plotted, it makes a "smile".
- For BTC (Nov 2026) the left side is higher: people pay more for crash protection than for upside bets. (I guessed the opposite. In crypto it can flip during strong rallies.)
- Short expiries have much more curved smiles than long ones. (I guessed the opposite here too.) Over a week, a big move is a sudden "jump", and the market charges a lot extra for it.

**SVI: a best-fit curve through the smile.**
- The market only gives vols at certain strikes (dots). To get a vol at any strike, you need a smooth curve. Like weather stations (dots) and a temperature map (the curve).
- SVI ("Stochastic Volatility Inspired", Jim Gatheral) is a smile-shaped curve with 5 knobs: height, steepness, tilt, where the bottom is, and how rounded it is. The computer turns the knobs until the curve fits the dots.
- On real Deribit data, every expiry fitted to within 0.05 to 0.4 vol points.

**Checking for arbitrage ("free money").**
- Butterfly check (across strikes): for three neighbouring strikes, the middle option can't cost more than the average of the two either side. Example: $15 / $10 / $6 is fine, but $15 / $10 / $4 is impossible, because selling 2 middle options and buying 1 of each neighbour pays you $1 upfront for a bet that can never lose.
- That's the same as saying probabilities can't be negative: a butterfly only pays if the price ends near the middle strike, so its price is roughly payout × probability. A negative price would mean a negative probability.
- Calendar check (across dates): a longer expiry can't be cheaper than a shorter one, because more time means more uncertainty.
- All BTC and ETH expiries passed both checks.

**The surface.**
- All the SVI curves lined up by date, like slices of a loaf of bread. For a date between two Deribit expiries, the surface blends the two curves, leaning towards the closer date.
- Example: the 79,000 strike was 38.1% on 25 Dec and 38.4% on 29 Jan; 10 Jan (not listed) comes out at 38.3%, in between.
- Days to expiry matter more than the calendar date, because an option's value depends on how much time is left.

### Milestone 5: Saving snapshots

**Why keep history.**
- A surface built today is a one-off photo of the market. Risk teams need many days of photos to answer questions like "how much do vols usually move in a day?" or "what could we lose on a bad day?" (Value at Risk).
- The next project, a risk engine, will use this history.

**What gets saved.**
- For each currency and day, two CSV files in `data/history/`: the SVI fits (small, and enough to rebuild that day's whole surface) and the clean option chain (the raw evidence, so a fit can always be re-checked).
- One snapshot per day: running it twice on the same day replaces that day's files.
- The data stays on my computer (the `data/` folder is ignored by Git).

**First run (4 Nov 2026).**
- BTC: 898 options downloaded, 333 kept after cleaning, 11 expiries fitted, no arbitrage.
- ETH: 738 options downloaded, 279 kept, 11 expiries fitted, no arbitrage.
- The forward price rises with expiry (BTC: about 78,500 for 2 days out up to about 82,400 for September 2027).

**Design choice.** I run it on demand rather than every day automatically. If the risk engine needs a longer history, public sources such as Deribit's historical volatility index (DVOL) can fill the gap.

## Disclaimer

For learning purposes only. 

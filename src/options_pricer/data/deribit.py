"""Download option data from Deribit's PUBLIC API (no account or API key needed).

Two endpoints:
- get_instruments:                the "menu": every option's strike, expiry and type
- get_book_summary_by_currency:   the "prices": bid, ask, mark price, mark_iv, forward

This file ONLY talks to the internet. Turning the raw data into a clean
table happens in chain.py, so that part can be tested without the internet.
"""

import time                      # for waiting between retries

import requests                  # the standard Python library for web requests

BASE_URL = "https://www.deribit.com/api/v2/public/"


def _get(method, params, retries=3, timeout=10):
    """Call one Deribit public API method and return its "result" part.

    method:  e.g. "get_instruments"
    params:  e.g. {"currency": "BTC", "kind": "option"}
    retries: how many times to try before giving up (the internet is flaky!)
    timeout: seconds to wait for an answer before giving up on that try
    """
    url = BASE_URL + method

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()          # turn HTTP errors (404, 500...) into Python errors
            data = response.json()               # the answer is JSON -> Python dict

            if "result" not in data:             # Deribit puts problems in an "error" field
                raise RuntimeError(f"Deribit error: {data.get('error')}")
            return data["result"]

        except (requests.RequestException, RuntimeError) as error:
            if attempt == retries:               # last try failed: give up loudly
                raise
            wait = 2 ** attempt                  # wait 2s, then 4s... ("exponential backoff")
            print(f"Deribit request failed ({error}); retrying in {wait}s...")
            time.sleep(wait)


def fetch_instruments(currency="BTC"):
    """List every live option for a currency: strike, expiry, call/put.

    Returns a list of dicts, one per option.
    """
    return _get("get_instruments", {"currency": currency, "kind": "option", "expired": "false"})


def fetch_book_summary(currency="BTC"):
    """Current prices for every live option of a currency.

    Returns a list of dicts with bid_price, ask_price, mark_price, mark_iv,
    underlying_price (the forward), open_interest, volume...
    NOTE: prices are in the COIN (BTC or ETH), not in USD!
    """
    return _get("get_book_summary_by_currency", {"currency": currency, "kind": "option"})

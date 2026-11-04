"""The volatility surface: a vol for ANY strike and ANY expiry date.

We have one SVI curve per Deribit expiry (e.g. 25 Dec, 29 Jan, 26 Mar).
The surface joins them up, so you can ask for e.g. strike 79,000 on 15 Jan,
a date Deribit doesn't list.

How it works, in plain English:
- On a listed expiry: just read the SVI curve for that expiry.
- BETWEEN two expiries: blend the two curves either side, weighted by how
  close the date is to each one. (We blend "total variance" = vol² × T,
  which is the standard way: it keeps the calendar check happy.)
- BEFORE the first expiry / AFTER the last: keep the nearest curve's vol.
"""

import numpy as np
import pandas as pd

from options_pricer.volatility.svi import fit_svi_by_expiry, params_from_row, svi_total_variance

YEAR = pd.Timedelta(days=365)


class VolSurface:
    """A volatility surface built from one SVI fit per expiry.

    Use it like this:
        surface = build_surface(otm_chain)
        surface.vol(strike=79000, expiry="15 Jan 2027")      # -> e.g. 0.384 (38.4%)
    """

    def __init__(self, fits, snapshot_time):
        """fits: table from fit_svi_by_expiry(); snapshot_time: when the market data was taken."""
        self.fits = fits.sort_values("T").reset_index(drop=True)   # shortest expiry first
        self.snapshot_time = pd.Timestamp(snapshot_time)
        self.T_list = self.fits["T"].to_numpy()                    # years to each listed expiry

    # ---- helpers ------------------------------------------------------------
    def years_to(self, expiry):
        """Turn a date (e.g. "15 Jan 2027") into years from the snapshot."""
        expiry = pd.Timestamp(expiry)
        if expiry.tzinfo is None:                     # dates typed by hand have no time zone
            expiry = expiry.tz_localize("UTC")        # Deribit times are UTC
        return (expiry - self.snapshot_time) / YEAR

    def forward(self, T):
        """Forward price at time T: straight-line blend between the listed expiries' forwards."""
        return float(np.interp(T, self.T_list, self.fits["forward"]))

    def _total_variance_on_expiry(self, i, k):
        """w = vol² × T from the SVI curve of listed expiry number i."""
        return svi_total_variance(k, params_from_row(self.fits.iloc[i]))

    # ---- the main function ------------------------------------------------
    def vol_at_T(self, strike, T):
        """Implied vol for a strike at T years from now."""
        if T <= 0:
            raise ValueError("Expiry must be after the snapshot time")

        k = np.log(np.asarray(strike, dtype=float) / self.forward(T))   # log-moneyness at this date
        T_first, T_last = self.T_list[0], self.T_list[-1]

        if T <= T_first:
            # Before the first expiry: same vol as the first curve -> scale variance with time
            w = self._total_variance_on_expiry(0, k) * T / T_first
        elif T >= T_last:
            # After the last expiry: same vol as the last curve
            w = self._total_variance_on_expiry(len(self.T_list) - 1, k) * T / T_last
        else:
            # Between two expiries: find the neighbours and blend their total variance
            right = int(np.searchsorted(self.T_list, T))        # first listed expiry after T
            left = right - 1
            T1, T2 = self.T_list[left], self.T_list[right]
            weight = (T - T1) / (T2 - T1)                        # 0 at T1, 1 at T2
            w1 = self._total_variance_on_expiry(left, k)
            w2 = self._total_variance_on_expiry(right, k)
            w = (1 - weight) * w1 + weight * w2

        vol = np.sqrt(np.maximum(w, 0.0) / T)                    # back from total variance to vol
        return vol[()]                                           # plain number if one strike given

    def vol(self, strike, expiry):
        """Implied vol for a strike on an expiry DATE, e.g. vol(79000, "15 Jan 2027")."""
        return self.vol_at_T(strike, self.years_to(expiry))


def build_surface(otm_chain):
    """Fit SVI to every expiry of a clean OTM chain and join them into a VolSurface.

    otm_chain: table from clean_chain() + otm_only() + add_implied_vols()
    """
    fits = fit_svi_by_expiry(otm_chain)
    snapshot_time = otm_chain["snapshot_time"].max()
    return VolSurface(fits, snapshot_time)

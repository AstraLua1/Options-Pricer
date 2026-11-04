"""Charts for the project. Notebooks and scripts call these; the maths lives elsewhere.

plot_smiles(): one small chart per expiry, showing implied vol against strike,
               optionally with the fitted SVI curve drawn through the dots.
"""

import math

import matplotlib.pyplot as plt
import numpy as np

from options_pricer.volatility.svi import params_from_row, svi_implied_vol

# Colours (a calm, readable palette)
POINT_COLOUR = "#2a78d6"     # blue: market implied vols
CURVE_COLOUR = "#eb6834"     # orange: fitted SVI curve
ATM_COLOUR = "#8a8984"       # grey: the "at the money" line
GRID_COLOUR = "#e6e5e1"


def plot_smiles(chain, title="Volatility smiles", columns=4, fits=None):
    """One small chart per expiry: implied vol (y) against moneyness (x).

    chain:   a CLEAN, out-of-the-money table (clean_chain() then otm_only())
             with columns expiry, T, moneyness, iv_mid
    title:   title for the whole figure
    columns: how many charts per row
    fits:    OPTIONAL table from fit_svi_by_expiry(). If given, the fitted
             SVI curve is drawn through the dots for each expiry.

    Moneyness = strike / forward, so 1.0 = at the money, 0.8 = 20% below, 1.2 = 20% above.
    Using moneyness (not raw strikes) puts every expiry on the same scale.

    Returns the matplotlib figure (so a script can save it).
    """
    chain = chain.dropna(subset=["iv_mid"])             # skip options with no mid vol
    expiries = sorted(chain["expiry"].unique())         # earliest expiry first
    rows = math.ceil(len(expiries) / columns)           # e.g. 12 expiries / 4 columns = 3 rows

    # sharex/sharey = every small chart uses the SAME axes, so you can compare them fairly
    fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 3 * rows),
                             sharex=True, sharey=True, squeeze=False)

    for ax, expiry in zip(axes.flat, expiries):
        one_expiry = chain[chain["expiry"] == expiry].sort_values("moneyness")
        days = one_expiry["T"].iloc[0] * 365
        label = f"{expiry:%d %b %Y}  ({days:.0f} days)"

        # The market's implied vols: one dot per option
        ax.plot(one_expiry["moneyness"], one_expiry["iv_mid"] * 100,
                "o", markersize=4, color=POINT_COLOUR, label="Market")

        # The fitted SVI curve (only if fits were given and this expiry was fitted)
        if fits is not None:
            fit_rows = fits[fits["expiry"] == expiry]
            if len(fit_rows) == 1:
                fit = fit_rows.iloc[0]
                # 200 evenly spaced points across this expiry's strikes, as log-moneyness k
                k_dots = np.log(one_expiry["moneyness"])
                k_line = np.linspace(k_dots.min(), k_dots.max(), 200)
                vols = svi_implied_vol(k_line, fit["T"], params_from_row(fit))
                ax.plot(np.exp(k_line), vols * 100, color=CURVE_COLOUR, linewidth=2, label="SVI fit")
                label += f"  error {fit['rmse_pts']:.2f} pts"

        # Dashed line where strike = forward (at the money)
        ax.axvline(1.0, color=ATM_COLOUR, linestyle="--", linewidth=1)

        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(True, color=GRID_COLOUR)

    # Hide any empty panels (e.g. 10 expiries in a 3×4 grid leaves 2 empty)
    for ax in list(axes.flat)[len(expiries):]:
        ax.set_visible(False)

    # Axis labels on the outer edge only, to keep it tidy
    for ax in axes[-1, :]:
        ax.set_xlabel("Moneyness (strike ÷ forward)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied vol (%)")

    # One legend for the whole figure (top right), only if a curve was drawn
    if fits is not None:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, ncol=2)

    fig.suptitle(title, fontsize=14, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_surface(surface, moneyness=(0.7, 1.4), days=(7, None), title="Volatility surface"):
    """3D picture of the whole surface: vol (height) by moneyness and days to expiry.

    surface:   a VolSurface (from build_surface())
    moneyness: range of strike ÷ forward to show (0.7 = 30% below, 1.4 = 40% above)
    days:      range of days to expiry to show; None = up to the last listed expiry
    """
    first_day = days[0]
    last_day = days[1] if days[1] is not None else surface.T_list[-1] * 365

    # A grid of points: 40 moneyness values × 40 expiry dates
    m_values = np.linspace(moneyness[0], moneyness[1], 40)
    d_values = np.linspace(first_day, last_day, 40)
    M, D = np.meshgrid(m_values, d_values)            # every combination of the two

    # Ask the surface for the vol at every point of the grid
    vols = np.zeros_like(M)
    for i, day in enumerate(d_values):
        T = day / 365
        strikes = m_values * surface.forward(T)       # moneyness -> actual strike at that date
        vols[i, :] = surface.vol_at_T(strikes, T) * 100

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")              # a 3D chart
    ax.plot_surface(M, D, vols, cmap="Blues", edgecolor="none", alpha=0.95)

    ax.set_xlabel("Moneyness (strike ÷ forward)")
    ax.set_ylabel("Days to expiry")
    ax.set_zlabel("Implied vol (%)")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.view_init(elev=25, azim=-60)                    # camera angle
    fig.tight_layout()
    return fig

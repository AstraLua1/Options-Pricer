"""Charts for the project. Notebooks and scripts call these; the maths lives elsewhere.

plot_smiles(): one small chart per expiry, showing implied vol against strike.
"""

import math

import matplotlib.pyplot as plt

# Colours (a calm, readable palette)
POINT_COLOUR = "#2a78d6"     # blue: market implied vols
ATM_COLOUR = "#8a8984"       # grey: the "at the money" line
GRID_COLOUR = "#e6e5e1"


def plot_smiles(chain, title="Volatility smiles", columns=4):
    """One small chart per expiry: implied vol (y) against moneyness (x).

    chain:   a CLEAN, out-of-the-money table (clean_chain() then otm_only())
             with columns expiry, T, moneyness, iv_mid
    title:   title for the whole figure
    columns: how many charts per row

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

        # The market's implied vols: one dot per option
        ax.plot(one_expiry["moneyness"], one_expiry["iv_mid"] * 100,
                "o", markersize=4, color=POINT_COLOUR)

        # Dashed line where strike = forward (at the money)
        ax.axvline(1.0, color=ATM_COLOUR, linestyle="--", linewidth=1)

        ax.set_title(f"{expiry:%d %b %Y}  ({days:.0f} days)", fontsize=10, loc="left")
        ax.grid(True, color=GRID_COLOUR)

    # Hide any empty panels (e.g. 10 expiries in a 3×4 grid leaves 2 empty)
    for ax in list(axes.flat)[len(expiries):]:
        ax.set_visible(False)

    # Axis labels on the outer edge only, to keep it tidy
    for ax in axes[-1, :]:
        ax.set_xlabel("Moneyness (strike ÷ forward)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Implied vol (%)")

    fig.suptitle(title, fontsize=14, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig

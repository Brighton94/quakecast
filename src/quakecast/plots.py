"""Shared matplotlib style and the plots the EDA notebooks reuse."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical slots in fixed order; use at most three on one chart.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = (BLUE, ORANGE, AQUA)


def set_style() -> None:
    """Thin marks, recessive grid and axes, no top or right spines."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "figure.figsize": (10, 4),
            "axes.edgecolor": INK_MUTED,
            "axes.labelcolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.prop_cycle": plt.cycler(color=SERIES),
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "lines.linewidth": 1.5,
            "legend.frameon": False,
            "font.size": 10,
        }
    )


def run_sequence_plot(series: pd.Series, title: str, ylabel: str, ax=None, **kw):
    """Plot a series against its index, as in a run-sequence plot."""
    ax = ax or plt.gca()
    ax.plot(series.index, series.to_numpy(), **({"lw": 0.8, "color": BLUE} | kw))
    ax.set(title=title, ylabel=ylabel)
    return ax


def fmd_plot(fmd: pd.DataFrame, mc: float, b: float, ax=None, label: str | None = None):
    """Frequency-magnitude distribution with the Gutenberg-Richter fit above Mc."""
    ax = ax or plt.gca()
    ax.semilogy(fmd["mag"], fmd["cum_count"], "o", ms=4, color=BLUE, label="N(≥M)")
    ax.semilogy(
        fmd["mag"], fmd["count"].replace(0, np.nan), "s", ms=3, color=ORANGE, label="N per 0.1 bin"
    )
    n_mc = fmd.loc[np.isclose(fmd["mag"], mc), "cum_count"]
    if len(n_mc):
        m = fmd["mag"][fmd["mag"] >= mc]
        ax.semilogy(
            m, n_mc.iloc[0] * 10 ** (-b * (m - mc)), color=INK, lw=1, label=f"G-R fit, b = {b:.2f}"
        )
    ax.axvline(mc, color=INK_MUTED, ls="--", lw=1)
    ax.annotate(
        label or f"Mc = {mc:.1f}",
        (mc, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        color=INK_MUTED,
    )
    ax.set(xlabel="Magnitude", ylabel="Number of events", title="Frequency-magnitude distribution")
    ax.legend()
    return ax


def seasonal_bars(profiles: dict[str, pd.DataFrame], by: str, ax=None):
    """Observed / expected ratio per period for up to three subsets, with chi-square p-values."""
    ax = ax or plt.gca()
    n = len(profiles)
    width = 0.8 / n
    for i, (name, prof) in enumerate(profiles.items()):
        x = np.arange(len(prof)) + (i - (n - 1) / 2) * width
        ax.bar(
            x,
            prof["ratio"],
            width=width * 0.9,
            color=SERIES[i],
            label=f"{name} (p = {prof.attrs['p_value']:.2g})",
        )
    ax.axhline(1, color=INK, lw=1)
    names = {
        "dayofweek": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "month": [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ],
    }
    ax.set_xticks(np.arange(len(prof)), names.get(by, prof.index))
    ax.set(xlabel=by, ylabel="Observed / expected", title=f"Events by {by}")
    ax.legend()
    return ax

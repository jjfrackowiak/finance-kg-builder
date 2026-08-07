import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8.5,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.22,
    "grid.linewidth":     0.5,
    "axes.axisbelow":     True,
})

TICKERS = ["MSFT", "TSLA"]
TC = {"MSFT": "#2ca02c", "TSLA": "#c0392b"}
OOS = "#e07b20"

# Identical windows for both tickers:
# 200-day in-sample window 2022-05-02 -> 2022-11-18
# 100-day OOS hold-out immediately following, 2022-11-19 -> 2023-02-26
_W = dict(ts="2022-05-02", te="2022-11-18", os="2022-11-19", oe="2023-02-26")
WINDOWS = {"MSFT": _W, "TSLA": _W}

DATA_PATH = ("/Users/mac/Desktop/Research and Study/Doktorat/"
             "downstream-finance-graph/finance-kg-builder/"
             "data/fnspid_sample_nasdaq_long_text.csv")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["date"] = df["timestamp"].dt.normalize()

print("Downloading prices…")
prices = yf.download(TICKERS, start="2021-07-01", end="2023-04-15",
                     auto_adjust=True, progress=False)["Close"]
prices.index = pd.to_datetime(prices.index)

FULL_START = pd.Timestamp("2021-07-01")
FULL_END   = pd.Timestamp("2023-04-15")

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 5.6),
    gridspec_kw={"hspace": 0.62, "wspace": 0.28,
                 "width_ratios": [2.6, 1]},
)


def clean_xaxis(ax):
    """Year-only ticks, minor ticks at quarters."""
    ax.set_xlim(FULL_START, FULL_END)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    ax.tick_params(axis="x", which="major", labelsize=8, pad=3)
    ax.tick_params(axis="x", which="minor", length=3, width=0.5)


def shade_windows(ax, ticker):
    w = WINDOWS[ticker]
    c = TC[ticker]
    ts, te = pd.Timestamp(w["ts"]), pd.Timestamp(w["te"])
    os_, oe = pd.Timestamp(w["os"]), pd.Timestamp(w["oe"])
    ax.axvspan(ts, te,  color=c,   alpha=0.14, linewidth=0, zorder=1)
    ax.axvspan(os_, oe, color=OOS, alpha=0.18, linewidth=0, zorder=1)
    # boundary lines
    for x, col in [(ts, c), (te, c), (oe, OOS)]:
        ax.axvline(x, color=col, alpha=0.70, linewidth=1.0,
                   linestyle="--", zorder=3)


def label_boundaries(ax, ticker):
    """Small rotated date labels just above each boundary line."""
    w = WINDOWS[ticker]
    c = TC[ticker]
    ts, te = pd.Timestamp(w["ts"]), pd.Timestamp(w["te"])
    oe = pd.Timestamp(w["oe"])

    ymin, ymax = ax.get_ylim()
    y_top = ymax - (ymax - ymin) * 0.01   # just inside top

    for x, col, label in [
        (ts,  c,   ts.strftime("%-d %b %Y")),
        (te,  c,   te.strftime("%-d %b %Y")),
        (oe,  OOS, oe.strftime("%-d %b %Y")),
    ]:
        ax.text(x, y_top, label,
                rotation=90, va="top", ha="right",
                fontsize=6.5, color=col, alpha=0.85,
                transform=ax.get_xaxis_transform()
                    if False else ax.transData,
                clip_on=True)


for row, ticker in enumerate(TICKERS):
    ax_p = axes[row, 0]
    ax_v = axes[row, 1]
    c = TC[ticker]

    # ── Price ─────────────────────────────────────────────────────────────
    clean_xaxis(ax_p)
    shade_windows(ax_p, ticker)

    px = prices[ticker].dropna()
    ax_p.plot(px.index, px.values, color=c, linewidth=1.1, zorder=4)
    ax_p.set_ylabel(f"{ticker}\nAdj. close (USD)",
                    fontweight="bold", color=c, labelpad=6)
    ax_p.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

    # boundary date labels AFTER y-limits are set by plot()
    ax_p.figure.canvas.draw()   # force layout so get_ylim() is accurate
    label_boundaries(ax_p, ticker)

    # ── Volume ────────────────────────────────────────────────────────────
    clean_xaxis(ax_v)
    shade_windows(ax_v, ticker)

    daily = (
        df[df["ticker"] == ticker]
        .groupby("date").size()
        .reindex(pd.date_range(FULL_START, FULL_END), fill_value=0)
    )
    rolling = daily.rolling(14, center=True, min_periods=1).mean()

    ymax_vol = max(rolling.quantile(0.97) * 1.7, 15)
    ax_v.bar(daily.index, daily.values.clip(0, ymax_vol * 1.05),
             width=1, color=c, alpha=0.18, linewidth=0, zorder=2)
    ax_v.plot(rolling.index, rolling.values.clip(0, ymax_vol),
              color=c, linewidth=1.3, zorder=3)
    ax_v.set_ylim(0, ymax_vol)
    ax_v.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4, prune="upper"))
    ax_v.set_ylabel("Articles / day", labelpad=4)

    covered = daily[(daily.index >= pd.Timestamp(_W["ts"])) &
                    (daily.index <= pd.Timestamp(_W["oe"])) &
                    (daily > 0)]
    p25 = covered.quantile(0.25)
    ax_v.axhline(p25, color=c, linewidth=0.9, linestyle=":", alpha=0.85,
                 label=f"p25 = {p25:.0f} art./day")
    ax_v.legend(loc="upper left", handlelength=1.2,
                handletextpad=0.4, framealpha=0, fontsize=7.5)

# ── Column headers ───────────────────────────────────────────────────────
axes[0, 0].set_title("Price history", fontsize=10, pad=6)
axes[0, 1].set_title("News volume",   fontsize=10, pad=6)

# ── Legend ───────────────────────────────────────────────────────────────
train_patch = mpatches.Patch(facecolor="#555577", alpha=0.40,
                              label="Training window  2 May 2022 → 18 Nov 2022  (200 days, 144 bdays)")
oos_patch   = mpatches.Patch(facecolor=OOS, alpha=0.60,
                              label="Out-of-sample hold-out  19 Nov 2022 → 26 Feb 2023  (100 days, 70 bdays)")
fig.legend(
    handles=[train_patch, oos_patch],
    loc="lower center", ncol=2,
    bbox_to_anchor=(0.5, -0.03),
    framealpha=0, fontsize=8.5,
    handlelength=1.4, handletextpad=0.5, columnspacing=2.0,
)

fig.suptitle(
    "FNSPID: price history and news coverage with experiment windows",
    fontsize=11, y=1.012,
)

OUT = ("/Users/mac/Desktop/Research and Study/Doktorat/"
       "downstream-finance-graph/finance-kg-builder/"
       "manuscript/figures/experiment_windows")
fig.savefig(OUT + ".pdf", bbox_inches="tight", format="pdf")
fig.savefig(OUT + ".png", bbox_inches="tight", dpi=300)
print(f"Saved {OUT}.pdf  and  {OUT}.png")

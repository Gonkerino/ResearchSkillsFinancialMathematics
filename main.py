# =============================================================================
# Compound Hawkes Processes for Limit Order Book Events
# MSc Mini-Project Notebook
#
# This notebook is self-contained. It walks you through:
#   1. What a Limit Order Book (LOB) is and how to read LOBSTER data
#   2. Visualising the LOB for 5 stocks: AMZN, AAPL, GOOG, MSFT, TSLA
#   3. Computing stylised facts (inter-arrivals, autocorrelation)
#   4. Fitting a Hawkes process model to order-flow event timestamps
#
# Required packages:
#   pip install numpy pandas matplotlib scipy statsmodels
#
# LOBSTER data format:
#   https://lobsterdata.com/info/DataStructure.php
#   Each stock-day has two files:
#     <TICKER>_<DATE>_34200000_57600000_message_10.csv
#     <TICKER>_<DATE>_34200000_57600000_orderbook_10.csv
#
# Set DATA_PATH below to the folder containing all your LOBSTER CSV files.
# =============================================================================

import os
import time
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize
from scipy.stats import ks_1samp, expon
from statsmodels.tsa.api import VAR
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        MofNCompleteColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
    HAVE_RICH = True
except ModuleNotFoundError:
    HAVE_RICH = False

# =============================================================================
# Global display settings
# =============================================================================
plt.rcParams.update({
    "figure.figsize"  : (12, 5),
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True, 
    "grid.alpha"      : 0.3,
    "font.size"       : 11,
})

STOCKS   = ["AMZN", "AAPL", "GOOG", "MSFT", "INTC"]
COLORS   = {"AMZN": "#FF9900", "AAPL": "#555555", "GOOG": "#4285F4",
            "MSFT": "#00A4EF", "INTC": "#CC0000"}

# =============================================================================
# 0.  CONFIGURE PATHS  (edit these two lines)
# =============================================================================
DATA_PATH  = "data/"          # folder containing all LOBSTER CSV files from https://data.lobsterdata.com/info/DataSamples.php
START_DATE = "2012-06-21"     # first date to load (YYYY-MM-DD)
END_DATE   = "2012-06-21"     # last  date to load (YYYY-MM-DD)

# Save all figures to 'plots/main'
PLOTS_DIR  = os.path.join("plots", "main")
os.makedirs(PLOTS_DIR, exist_ok=True)

console = Console() if HAVE_RICH else None
_QUIET_LOGS = False


def set_quiet_logs(flag):
    """Enable/disable non-critical logging (used during live progress rendering)."""
    global _QUIET_LOGS
    _QUIET_LOGS = bool(flag)


def _log(msg, force=False):
    """Unified logger: rich if available, print otherwise."""
    if _QUIET_LOGS and not force:
        return
    if HAVE_RICH and console is not None:
        console.print(msg)
    else:
        print(msg)


# =============================================================================
# SECTION 1 — LOBSTER DATA LOADER
# =============================================================================
# 1.1  Background: what does a Limit Order Book look like?
# =============================================================================
"""
A Limit Order Book (LOB) records every buy and sell order that has been
submitted but not yet filled.  Think of it as two sorted queues:

   BID SIDE (buyers)            ASK SIDE (sellers)
   ─────────────────            ─────────────────
   Best bid  ← $ 99.98  ...  $ 100.02 →  Best ask
             ← $ 99.96  ...  $ 100.04 →
             ← $ 99.94  ...  $ 100.06 →

   • Spread  = Best ask − Best bid  (the cost of immediately trading)
   • Mid-price = (Best ask + Best bid) / 2

LOBSTER records *every* event that changes this book:
   Type 1 → New limit order arrives         (adds liquidity)
   Type 2 → Partial cancellation            (removes some volume)
   Type 3 → Full deletion                   (order withdrawn)
   Type 4 → Visible limit order executed    (a market order hit it)
   Type 5 → Hidden limit order executed
   Type 7 → Trading halt
"""

class Loader:
    """
    Loads and pre-processes LOBSTER message + order-book files.

    Parameters
    ----------
    ric       : str   Ticker symbol, e.g. "AMZN" or "AMZN.O"
    sDate     : str   Start date "YYYY-MM-DD"
    eDate     : str   End  date  "YYYY-MM-DD"
    nlevels   : int   Number of price levels to retain (default 10)
    dataPath  : str   Folder containing the LOBSTER CSV files
    """

    # ── Event-type labels (for plots and printed output) ──────────────────
    EVENT_LABELS = {
        1: "New Limit Order",
        2: "Partial Cancel",
        3: "Full Delete",
        4: "Market Order (visible)",
        5: "Market Order (hidden)",
        7: "Trading Halt",
    }

    def __init__(self, ric, sDate, eDate, **kwargs):
        self.ric      = ric.split(".")[0]   # strip exchange suffix if present
        self.sDate    = sDate
        self.eDate    = eDate
        self.nlevels  = kwargs.get("nlevels",  10)
        self.dataPath = kwargs.get("dataPath", DATA_PATH)

    # ── Internal helpers ──────────────────────────────────────────────────
    def _col_names(self):
        """Return column names for the order-book file (up to nlevels)."""
        sides   = ["Ask Price", "Ask Size", "Bid Price", "Bid Size"]
        all_names, keep = [], []
        for lvl in range(1, 11):
            for s in sides:
                col = f"{s} {lvl}"
                all_names.append(col)
                if lvl <= self.nlevels:
                    keep.append(col)
        return all_names, keep

    def _find_file(self, date_str, kind):
        """
        Locate a LOBSTER file.  Handles the two most common naming schemes:
          <TICKER>_<DATE>_34200000_57600000_<kind>_10.csv
          <TICKER>_<DATE>_34200000_57600000_<kind>_5.csv
        """
        for levels_tag in ["10", "5"]:
            fname = f"{self.ric}_{date_str}_34200000_57600000_{kind}_{levels_tag}.csv"
            if os.path.exists(os.path.join(self.dataPath, fname)):
                return os.path.join(self.dataPath, fname)
        return None

    # ── Public API ────────────────────────────────────────────────────────
    def load(self):
        """
        Load and clean all available trading days.

        Returns
        -------
        list of pd.DataFrame
            One DataFrame per trading day.  Each DataFrame contains the
            LOBSTER message columns plus the order-book columns, restricted
            to the continuous-trading session.
        """
        data = []
        all_names, keep_cols = self._col_names()

        for d in pd.date_range(self.sDate, self.eDate, freq="B"):  # business days
            date_str = d.strftime("%Y-%m-%d")
            msg_path = self._find_file(date_str, "message")
            ob_path  = self._find_file(date_str, "orderbook")

            if msg_path is None or ob_path is None:
                continue

            _log(f"  Loading {self.ric}  {date_str} ...")

            # ── Message book ──────────────────────────────────────────────
            msg = pd.read_csv(
                msg_path,
                names=["Time", "Type", "OrderID", "Size", "Price", "TradeDirection", "tmp"],
            )

            # Restrict to 09:30 – 16:00 (in seconds after midnight)
            t_open  = 9.5 * 3600   # 34200 s
            t_close = 16.0 * 3600  # 57600 s
            msg = msg[(msg.Time >= t_open) & (msg.Time <= t_close)].copy()

            # Identify the opening auction (first Type==6 event) and any
            # closing auction (second Type==6 event, or end-of-day).
            type6 = msg[msg.Type == 6]
            if type6.empty:
                # No auction markers → use full session
                t_start, t_end = t_open, t_close
            else:
                t_start = type6.iloc[0].Time
                t_end   = type6.iloc[1].Time if len(type6) > 1 else t_close

            msg = msg[(msg.Time >= t_start) & (msg.Time <= t_end)].copy()

            # ── Order book ────────────────────────────────────────────────
            ob = pd.read_csv(ob_path, names=all_names)[keep_cols]

            # Align to the same rows as the filtered message book
            row_idx = ob.index[
                (msg.index[0] <= ob.index) & (ob.index <= msg.index[-1])
                ]
            # Safer: use positional iloc based on original message index
            ob_idx = msg.index  # same row numbers as the original file
            ob_aligned = ob.loc[ob_idx]

            # Convert prices from integer ticks (×10 000) to USD
            price_cols = [c for c in ob_aligned.columns if "Price" in c]
            ob_aligned = ob_aligned.copy()
            ob_aligned[price_cols] = ob_aligned[price_cols] / 10_000.0
            msg["Price"] = msg["Price"] / 10_000.0

            # ── Combine & tag ─────────────────────────────────────────────
            combined = pd.concat([msg.reset_index(drop=True),
                                  ob_aligned.reset_index(drop=True)], axis=1)
            combined["Date"] = date_str
            combined["Ticker"] = self.ric
            # Zero-based time (seconds from open)
            combined["TimeSinceOpen"] = combined["Time"] - t_start

            data.append(combined)

        if not data:
            _log(f"  [yellow]⚠[/yellow] No data found for {self.ric} between {self.sDate} and {self.eDate}.", force=True)
            _log(f"     Expected files in: {os.path.abspath(self.dataPath)}", force=True)

        return data

    def load12DTimestamps(self):
        """
        Return event-time arrays for the 12-dimensional Hawkes model:
          [lo_deep_Bid, co_deep_Bid, lo_top_Bid, co_top_Bid, mo_Bid,
           lo_inspread_Bid,
           lo_inspread_Ask, mo_Ask, co_top_Ask, lo_top_Ask,
           co_deep_Ask, lo_deep_Ask]

        (Bid-side list is reversed so that the ordering is symmetric.)

        Returns
        -------
        dict  {date_string : list of 12 np.ndarray of timestamps}
        """
        data = self.load()
        if not data:
            return {}

        offset = 9.5 * 3600
        order_types = {"limit": [1], "cancel": [2, 3], "market": [4]}
        res = {}

        for df in data:
            df = df.copy()
            df["Time"] -= offset
            df["BidDiff"]  = df["Bid Price 1"].diff()
            df["AskDiff"]  = df["Ask Price 1"].diff()
            df["BidDiff2"] = df["Bid Price 2"].diff()
            df["AskDiff2"] = df["Ask Price 2"].diff()

            arr, df_res_l = [], []

            for s in [1, -1]:
                side = "Bid" if s == 1 else "Ask"
                lo = df[(df.Type.isin(order_types["limit"]))  & (df.TradeDirection == s)]
                co = df[(df.Type.isin(order_types["cancel"])) & (df.TradeDirection == s)]
                mo = df[(df.Type.isin(order_types["market"])) & (df.TradeDirection == s)]

                at_top = lambda x: (
                        (x["Price"] <= x["Ask Price 1"] + 1e-3) and
                        (x["Price"] >= x["Bid Price 1"] - 1e-3)
                )
                at_lvl2 = lambda x, sd: np.isclose(x.Price, x[f"{sd} Price 2"])

                lo_deep     = lo[lo.apply(at_lvl2, sd=side, axis=1)].copy()
                lo_deep["event"] = f"lo_deep_{side}"

                co_deep     = co[co.apply(at_lvl2, sd=side, axis=1) |
                                 (((co["BidDiff2"] < 0) & (co["BidDiff"] == 0)) |
                                  ((co["AskDiff2"] > 0) & (co["AskDiff"] == 0)))].copy()
                co_deep["event"] = f"co_deep_{side}"

                lo_inspread = lo[((lo["BidDiff"] > 0) | (lo["AskDiff"] < 0))].copy()
                lo_inspread["event"] = f"lo_inspread_{side}"

                lo_top = lo[lo.apply(at_top, axis=1)].copy()
                lo_top = lo_top[lo_top[f"{side}Diff"] == 0].copy()
                lo_top["event"] = f"lo_top_{side}"

                co_top = co[co.apply(at_top, axis=1)].copy()
                co_top["event"] = f"co_top_{side}"

                mo["event"] = f"mo_{side}"

                df_res_l.append(pd.concat([lo_deep, co_deep, lo_top, co_top, mo, lo_inspread]))

                l = [lo_deep.Time.values, co_deep.Time.values, lo_top.Time.values,
                     co_top.Time.values, mo.Time.values, lo_inspread.Time.values]
                if s == 1:
                    l.reverse()
                arr += l

            res[df.Date.iloc[0]] = arr

        return res

    def load8DTimestamps_Bacry(self):
        """
        Return event-time arrays for the 8-dimensional Hawkes model of
        Bacry et al. (2016):
          [P_Bid, mo_Bid, lo_top_Bid, co_top_Bid,
           co_top_Ask, lo_top_Ask, mo_Ask, P_Ask]
        where P_* = price-change events on that side.

        Returns
        -------
        dict  {date_string : list of 8 np.ndarray of timestamps}
        """
        data = self.load()
        if not data:
            return {}

        offset = 9.5 * 3600
        order_types = {"limit": [1], "cancel": [2, 3], "market": [4]}
        res = {}

        for df in data:
            df = df.copy()
            df["Time"] -= offset
            df["BidDiff"] = df["Bid Price 1"].diff()
            df["AskDiff"] = df["Ask Price 1"].diff()

            arr, df_res_l = [], []

            for s in [1, -1]:
                side = "Bid" if s == 1 else "Ask"
                at_top = lambda x: (
                        (x["Price"] <= x["Ask Price 1"] + 1e-3) and
                        (x["Price"] >= x["Bid Price 1"] - 1e-3)
                )

                P  = df[df[f"{side}Diff"] != 0].copy()
                P["event"] = f"pc_{side}"

                mo = df[(df.Type.isin(order_types["market"])) &
                        (df.TradeDirection == s) &
                        (df[f"{side}Diff"] == 0)].copy()
                mo["event"] = f"mo_{side}"

                lo = df[(df.Type.isin(order_types["limit"]))  &
                        (df.TradeDirection == s) & (df[f"{side}Diff"] == 0)]
                co = df[(df.Type.isin(order_types["cancel"])) &
                        (df.TradeDirection == s) & (df[f"{side}Diff"] == 0)]

                lo_top = lo[lo.apply(at_top, axis=1)].copy()
                lo_top["event"] = f"lo_top_{side}"

                co_top = co[co.apply(at_top, axis=1)].copy()
                co_top["event"] = f"co_top_{side}"

                df_res_l.append(pd.concat([P, mo, lo_top, co_top]))

                l = [P.Time.values, mo.Time.values, lo_top.Time.values, co_top.Time.values]
                if s == 1:
                    l.reverse()
                arr += l

            res[df.Date.iloc[0]] = arr

        return res

    def loadBinned(self, binLength=1, filterTop=False):
        """
        Bin events into fixed-length intervals of `binLength` seconds.

        Returns
        -------
        dict  {date_string : {event_key : DataFrame with columns count, Size}}
        """
        data = self.load()
        order_types = {"limit": [1], "cancel": [2, 3], "market": [4]}
        binnedData = {}

        for df in data:
            binnedL = {}
            for k, v in order_types.items():
                for s in [1, -1]:
                    side = "bid" if s == 1 else "ask"
                    l = df[(df.Type.isin(v)) & (df.TradeDirection == s)].copy()
                    if filterTop:
                        l = l[l.apply(
                            lambda x: (x["Price"] <= x["Ask Price 1"] + 1e-3) and
                                      (x["Price"] >= x["Bid Price 1"] - 1e-3), axis=1
                        )]
                    l["count"] = 1
                    bins   = np.arange(df.Time.min() - 1e-3, df.Time.max(), binLength)
                    labels = np.arange(0, len(bins) - 1)
                    l["binIndex"] = pd.cut(l["Time"], bins=bins, labels=labels)
                    binL = l.groupby("binIndex").sum()[["count", "Size"]]
                    binL.reset_index(inplace=True)
                    binnedL[f"{k}_{side}"] = binL
            binnedData[df.Date.iloc[0]] = binnedL

        return binnedData


# =============================================================================
# SECTION 2 — LOB VISUALISATION HELPERS
# =============================================================================

def plot_lob_diagram():
    """
    Draw a static annotated diagram explaining the LOB structure.
    No real data needed — just a teaching illustration.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Limit Order Book — Structure Overview", fontsize=14, fontweight="bold", pad=12)

    # ── Bid side ──────────────────────────────────────────────────────────
    bid_prices  = [99.96, 99.97, 99.98]
    bid_volumes = [300,   200,   500]
    bid_color   = "#4CAF50"

    for i, (p, v) in enumerate(zip(bid_prices, bid_volumes)):
        bar_len = v / 60
        ax.barh(7 - i, bar_len, left=4.8 - bar_len, color=bid_color, alpha=0.7 + 0.1*i)
        ax.text(4.75, 7 - i, f"${p:.2f}  [{v}]", ha="right", va="center",
                fontsize=10, color="darkgreen", fontweight="bold")

    # ── Ask side ──────────────────────────────────────────────────────────
    ask_prices  = [100.00, 100.01, 100.02]
    ask_volumes = [400,    250,    600]
    ask_color   = "#F44336"

    for i, (p, v) in enumerate(zip(ask_prices, ask_volumes)):
        bar_len = v / 60
        ax.barh(7 - i, bar_len, left=5.2, color=ask_color, alpha=0.7 + 0.1*i)
        ax.text(5.25, 7 - i, f"${p:.2f}  [{v}]", ha="left", va="center",
                fontsize=10, color="darkred", fontweight="bold")

    # ── Labels ────────────────────────────────────────────────────────────
    ax.text(3.2, 8.5, "BID SIDE\n(buyers)", ha="center", fontsize=11,
            color="darkgreen", fontweight="bold")
    ax.text(7.5, 8.5, "ASK SIDE\n(sellers)", ha="center", fontsize=11,
            color="darkred", fontweight="bold")
    ax.text(5.0, 8.8, "SPREAD", ha="center", fontsize=9, color="purple")
    ax.annotate("", xy=(5.15, 8.45), xytext=(4.85, 8.45),
                arrowprops=dict(arrowstyle="<->", color="purple", lw=1.5))

    ax.text(5.0, 4.3, "Price", ha="center", fontsize=9, color="grey")
    ax.text(5.0, 4.0, "Level 1  ← Best bid | Best ask →", ha="center",
            fontsize=9, color="grey")

    # ── Legend ────────────────────────────────────────────────────────────
    ax.text(0.5, 1.8, "Event types:", fontsize=10, fontweight="bold")
    events = [
        ("Type 1", "New limit order"),
        ("Type 2/3", "Cancel / Delete"),
        ("Type 4/5", "Market order (fills limit)"),
    ]
    for j, (t, desc) in enumerate(events):
        ax.text(0.5, 1.3 - 0.5*j, f"  {t}: {desc}", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "lob_diagram.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


def plot_lob_snapshot(df, ticker, n_levels=5, title_extra=""):
    """
    Bar chart of one LOB snapshot: bid and ask depth across price levels.

    Parameters
    ----------
    df          : pd.DataFrame   One row of a loaded LOBSTER DataFrame
    ticker      : str
    n_levels    : int            How many price levels to show
    title_extra : str            Extra info to show in the title
    """
    row = df.iloc[len(df) // 2]   # snapshot from the middle of the day

    bid_prices  = [row[f"Bid Price {i}"] for i in range(1, n_levels + 1)]
    bid_vols    = [row[f"Bid Size {i}"]  for i in range(1, n_levels + 1)]
    ask_prices  = [row[f"Ask Price {i}"] for i in range(1, n_levels + 1)]
    ask_vols    = [row[f"Ask Size {i}"]  for i in range(1, n_levels + 1)]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.barh(range(n_levels), bid_vols,  color="#4CAF50", alpha=0.75, label="Bid")
    ax.barh(range(n_levels), [-v for v in ask_vols], color="#F44336", alpha=0.75, label="Ask")

    labels = [f"${bp:.2f} | ${ap:.2f}" for bp, ap in zip(bid_prices, ask_prices)]
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"Level {i+1}: {lbl}" for i, lbl in enumerate(labels)])
    ax.axvline(0, color="black", lw=1.2)
    ax.set_xlabel("Volume (shares)  ←  Bid  |  Ask  →")
    ax.set_title(f"{ticker} — LOB Snapshot  {title_extra}", fontweight="bold")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{abs(int(x))}"))

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"lob_snapshot_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


def plot_midprice_and_spread(df, ticker):
    """
    Two-panel plot:
      Top   — mid-price through the trading day
      Bottom — bid-ask spread (in cents)
    """
    df = df.copy()
    df["mid"]    = (df["Ask Price 1"] + df["Bid Price 1"]) / 2
    df["spread"] = (df["Ask Price 1"] - df["Bid Price 1"]) * 100  # in cents

    # Convert time to HH:MM for the x-axis
    df["TimeHM"] = pd.to_datetime(df["Time"], unit="s", origin="1970-01-01") \
        .dt.strftime("%H:%M")

    # Thin out for readability (every 500th event)
    thin = max(1, len(df) // 500)
    df_thin = df.iloc[::thin]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(df_thin.index, df_thin["mid"], color=COLORS.get(ticker, "steelblue"), lw=1.2)
    ax1.set_ylabel("Mid-price ($)")
    ax1.set_title(f"{ticker} — Mid-price & Spread  ({df.Date.iloc[0]})", fontweight="bold")

    ax2.fill_between(df_thin.index, df_thin["spread"], alpha=0.4, color="purple")
    ax2.plot(df_thin.index, df_thin["spread"], color="purple", lw=0.8)
    ax2.set_ylabel("Bid-ask spread (¢)")
    ax2.set_xlabel("Event index")

    # Annotate average spread
    avg_spread = df["spread"].mean()
    ax2.axhline(avg_spread, color="red", ls="--", lw=1, label=f"Mean = {avg_spread:.2f}¢")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"midprice_spread_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")

def plot_midprice_all(summaries):
    """
    Combined mid-price and spread plot for all tickers on a shared time axis.
    Two rows of subplots: top row = mid-price, bottom row = bid-ask spread.
    """
    tickers = list(summaries.keys())
    n = len(tickers)
    if n == 0:
        return

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 6), sharex=False)
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle(
        "Mid-price and Quoted Spread — All Tickers (2012-06-21, 10:30–15:00 ET)",
        fontweight="bold", fontsize=12,
    )

    for j, ticker in enumerate(tickers):
        df = summaries[ticker].copy()
        df["mid"]    = (df["Ask Price 1"] + df["Bid Price 1"]) / 2
        df["spread"] = (df["Ask Price 1"] - df["Bid Price 1"]) * 100

        thin = max(1, len(df) // 500)
        df_thin = df.iloc[::thin]

        color = COLORS.get(ticker, "steelblue")

        ax_top = axes[0, j]
        ax_top.plot(df_thin["TimeSinceOpen"] / 3600, df_thin["mid"],
                    color=color, lw=1.0)
        ax_top.set_title(ticker, fontweight="bold", color=color)
        if j == 0:
            ax_top.set_ylabel("Mid-price ($)")
        ax_top.tick_params(labelbottom=False)

        ax_bot = axes[1, j]
        ax_bot.fill_between(df_thin["TimeSinceOpen"] / 3600,
                            df_thin["spread"], alpha=0.35, color=color)
        ax_bot.plot(df_thin["TimeSinceOpen"] / 3600, df_thin["spread"],
                    color=color, lw=0.8)
        avg = df["spread"].mean()
        ax_bot.axhline(avg, color="red", ls="--", lw=0.8,
                       label=f"Mean={avg:.2f}¢")
        ax_bot.legend(fontsize=7)
        ax_bot.set_xlabel("Hours from open")
        if j == 0:
            ax_bot.set_ylabel("Spread (¢)")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "midprice_all.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}", force=True)

def plot_event_breakdown(df, ticker):
    """
    Stacked bar chart showing the mix of order types across 30-min buckets.
    """
    df = df.copy()
    # Bucket into 30-min windows
    bucket_size  = 1800   # seconds
    df["bucket"] = ((df["Time"] - df["Time"].iloc[0]) // bucket_size).astype(int)

    type_labels = {1: "Limit", 2: "Part-Cancel", 3: "Delete", 4: "Market", 5: "Hidden"}
    grouped = df.groupby(["bucket", "Type"]).size().unstack(fill_value=0)
    grouped.rename(columns=type_labels, inplace=True)
    grouped = grouped.reindex(columns=[v for v in type_labels.values() if v in grouped.columns])

    ax = grouped.plot(kind="bar", stacked=True, figsize=(11, 4),
                      color=["#4CAF50", "#FF9800", "#F44336", "#2196F3", "#9C27B0"])
    ax.set_title(f"{ticker} — Order-type Mix per 30-min Bucket  ({df.Date.iloc[0]})",
                 fontweight="bold")
    ax.set_xlabel("30-min bucket (0 = 09:30)")
    ax.set_ylabel("Number of events")
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"event_breakdown_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


def plot_depth_heatmap(df, ticker, n_levels=5, n_bins=50):
    """
    Heatmap of bid and ask depth over time.
    Rows = price levels 1..n_levels; columns = time bins.
    """
    df = df.copy()
    time_bins = np.linspace(df["Time"].min(), df["Time"].max(), n_bins + 1)
    bin_idx   = np.digitize(df["Time"], time_bins) - 1
    bin_idx   = np.clip(bin_idx, 0, n_bins - 1)
    df["bin"] = bin_idx

    bid_map = np.zeros((n_levels, n_bins))
    ask_map = np.zeros((n_levels, n_bins))

    for lvl in range(1, n_levels + 1):
        bid_agg = df.groupby("bin")[f"Bid Size {lvl}"].mean()
        ask_agg = df.groupby("bin")[f"Ask Size {lvl}"].mean()
        bid_map[lvl - 1, bid_agg.index] = bid_agg.values
        ask_map[lvl - 1, ask_agg.index] = ask_agg.values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    im1 = ax1.imshow(bid_map, aspect="auto", cmap="Greens", origin="lower")
    ax1.set_title(f"{ticker} Bid Depth over Time", fontweight="bold")
    ax1.set_xlabel("Time bin")
    ax1.set_ylabel("Price level (1 = best)")
    ax1.set_yticks(range(n_levels))
    ax1.set_yticklabels([f"L{i+1}" for i in range(n_levels)])
    plt.colorbar(im1, ax=ax1, label="Avg volume")

    im2 = ax2.imshow(ask_map, aspect="auto", cmap="Reds", origin="lower")
    ax2.set_title(f"{ticker} Ask Depth over Time", fontweight="bold")
    ax2.set_xlabel("Time bin")
    ax2.set_ylabel("Price level (1 = best)")
    ax2.set_yticks(range(n_levels))
    ax2.set_yticklabels([f"L{i+1}" for i in range(n_levels)])
    plt.colorbar(im2, ax=ax2, label="Avg volume")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"depth_heatmap_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


def plot_cross_stock_summary(summaries):
    """
    Four-panel comparison across all five stocks:
      (a) Average bid-ask spread
      (b) Total event count
      (c) Average level-1 bid volume
      (d) Average level-1 ask volume

    Parameters
    ----------
    summaries : dict  {ticker : pd.DataFrame}   one df per stock
    """
    tickers = list(summaries.keys())
    colors  = [COLORS.get(t, "grey") for t in tickers]

    def stat(key):
        return [summaries[t][key].mean() for t in tickers]

    spread_cents  = [((summaries[t]["Ask Price 1"] - summaries[t]["Bid Price 1"]) * 100).mean()
                     for t in tickers]
    n_events      = [len(summaries[t]) for t in tickers]
    avg_bid_vol   = [summaries[t]["Bid Size 1"].mean() for t in tickers]
    avg_ask_vol   = [summaries[t]["Ask Size 1"].mean() for t in tickers]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    fig.suptitle("Cross-stock LOB Summary", fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], spread_cents,  "Avg Bid-Ask Spread (¢)",    "purple"),
        (axes[0, 1], n_events,      "Total Events",               "steelblue"),
        (axes[1, 0], avg_bid_vol,   "Avg Best-Bid Volume (shares)","#4CAF50"),
        (axes[1, 1], avg_ask_vol,   "Avg Best-Ask Volume (shares)","#F44336"),
    ]

    for ax, vals, ylabel, clr in panels:
        bars = ax.bar(tickers, vals, color=colors, edgecolor="white", lw=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{v:,.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "cross_stock_summary.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


# =============================================================================
# SECTION 3 — STYLISED FACTS
# =============================================================================

def compute_stylised_facts(df, ticker, quiet=False):
    """
    Plot inter-arrival time distribution and signed-move autocorrelation
    for market-order events in `df`.
    """
    mo = df[df["Type"] == 4].copy()
    if len(mo) < 10:
        _log(f"  [yellow]⚠[/yellow] Not enough market orders for {ticker} to compute stylised facts.", force=not quiet)
        return

    T = mo["Time"].values
    inter_arrival = np.diff(T)

    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(f"{ticker} — Stylised Facts  ({df.Date.iloc[0]})",
                 fontsize=13, fontweight="bold")

    # ── Inter-arrival times ───────────────────────────────────────────────
    log_bins = np.logspace(np.log10(inter_arrival[inter_arrival > 0].min()),
                           np.log10(inter_arrival.max()), 60)
    ax1.hist(inter_arrival, bins=log_bins, density=True, color=COLORS.get(ticker, "steelblue"),
         alpha=0.7, edgecolor="white")
    # Overlay exponential fit (memoryless baseline)
    lam  = 1 / inter_arrival.mean()
    xs   = np.linspace(0, np.percentile(inter_arrival, 97), 200)
    ax1.plot(xs, lam * np.exp(-lam * xs), "r--", lw=2, label=f"Exp(λ={lam:.2f})")
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Inter-arrival time (s) - log")
    ax1.set_ylabel("Density")
    ax1.set_title("Market-order Inter-arrival Times")
    ax1.legend()
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-5, top=1e4)

    # ── Signed-move autocorrelation ───────────────────────────────────────
    max_lag = min(30, len(X) - 2)
    lags    = np.arange(1, max_lag + 1)
    if max_lag >= 1:
        xc = X - X.mean()
        var = float(np.dot(xc, xc))
        if var == 0.0:
            acf = np.zeros(max_lag)
        else:
            full = np.correlate(xc, xc, mode="full")
            mid = len(full) // 2
            acf = full[mid + 1: mid + 1 + max_lag] / var
    else:
        acf = np.array([])
    ax2.bar(lags, acf, color=COLORS.get(ticker, "steelblue"), alpha=0.8)
    ax2.axhline(0, color="black", lw=0.8)
    # 95 % confidence band (i.i.d. benchmark)
    ci = 1.96 / np.sqrt(len(X))
    ax2.axhline( ci, color="red", ls="--", lw=1, label="95% CI (i.i.d.)")
    ax2.axhline(-ci, color="red", ls="--", lw=1)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Signed Trade-size Autocorrelation")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"stylised_facts_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")

    # ── Save individual panels ────────────────────────────────────────────
    # Panel 1: IAT density only
    fig1, ax1s = plt.subplots(1, 1, figsize=(16, 5))
    fig1.suptitle(f"{ticker} — Market-order Inter-arrival Times  ({df.Date.iloc[0]})",
                  fontsize=13, fontweight="bold")
    log_bins = np.logspace(np.log10(inter_arrival[inter_arrival > 0].min()),
                           np.log10(inter_arrival.max()), 60)
    ax1s.hist(inter_arrival, bins=log_bins, density=True,
              color=COLORS.get(ticker, "steelblue"), alpha=0.7, edgecolor="white")
    lam  = 1 / inter_arrival.mean()
    xs   = np.linspace(0, np.percentile(inter_arrival, 97), 200)
    ax1s.plot(xs, lam * np.exp(-lam * xs), "r--", lw=2, label=f"Exp(λ={lam:.2f})")
    ax1s.set_xscale('log'); ax1s.set_yscale('log')
    ax1s.set_xlabel("Inter-arrival time (s) - log"); ax1s.set_ylabel("Density")
    ax1s.legend()
    ax1s.set_xlim(left=1e-6, right=5e2); ax1s.set_ylim(bottom=1e-5, top=1e4)
    plt.tight_layout()
    fname_iat = os.path.join(PLOTS_DIR, f"iat_density_{ticker}.png")
    plt.savefig(fname_iat, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname_iat}")

    # Panel 2: ACF only
    fig2, ax2s = plt.subplots(1, 1, figsize=(16, 4))
    fig2.suptitle(f"{ticker} — Signed Trade-size Autocorrelation  ({df.Date.iloc[0]})",
                  fontsize=13, fontweight="bold")
    ax2s.bar(lags, acf, color=COLORS.get(ticker, "steelblue"), alpha=0.8)
    ax2s.axhline(0, color="black", lw=0.8)
    ci = 1.96 / np.sqrt(len(X))
    ax2s.axhline( ci, color="red", ls="--", lw=1, label="95% CI (i.i.d.)")
    ax2s.axhline(-ci, color="red", ls="--", lw=1)
    ax2s.set_xlabel("Lag"); ax2s.set_ylabel("Autocorrelation")
    ax2s.legend(fontsize=9)
    plt.tight_layout()
    fname_acf = os.path.join(PLOTS_DIR, f"acf_{ticker}.png")
    plt.savefig(fname_acf, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname_acf}")


# =============================================================================
# SECTION 4 — VAR MEMORY TEST
# =============================================================================

def var_memory_test(df, ticker, bin_length=1.0, max_lags=10, quiet=False):
    """
    Bin market-order events into `bin_length`-second windows and fit a
    Vector Auto-Regression to test for temporal dependence.

    Returns the fitted VAR result object.
    """
    mo = df[df["Type"] == 4]
    T = mo["Time"].values
    if len(T) < 50:
        _log(f"  [yellow]⚠[/yellow] Too few market orders for VAR test ({ticker}).", force=not quiet)
        return None

    bins = np.arange(T.min(), T.max(), bin_length)
    counts, _ = np.histogram(T, bins=bins)

    count_df = pd.DataFrame({"N": counts})
    # VAR requires >=2 variables - add lagged columns as a second series
    count_df["N_lag1"] = count_df["N"].shift(1).fillna(0)
    model = VAR(count_df)
    try:
        res = model.fit(maxlags=max_lags, ic="aic")
    except Exception as e:
        _log(f"  [yellow]⚠[/yellow] VAR fitting failed for {ticker}: {e}", force=not quiet)
        return None

    if not quiet:
        _log(f"\n{'='*60}", force=True)
        _log(f"  VAR Memory Test - {ticker}  (bin = {bin_length}s)", force=True)
        _log(f"  Selected lag order : {res.k_ar}", force=True)
        _log(f"  AIC                : {res.aic:.2f}", force=True)
        _log("  If lag > 0 -> market-order arrivals have memory (consistent with Hawkes).", force=True)
        _log(f"{'='*60}\n", force=True)

    return res

# =============================================================================
# SECTION 5 — 1-D HAWKES PROCESS (exponential kernel)
# =============================================================================
"""
The Hawkes process is a *self-exciting* point process.  Each event raises
the future intensity (rate of arrivals), which then decays exponentially.

Intensity:
  λ(t) = μ  +  Σ_{tᵢ < t}  α · exp(−β (t − tᵢ))
            ↑               ↑
      baseline rate      self-excitation

Parameters:
  μ (mu)    — background (unconditional) intensity  [events/sec]
  α (alpha) — jump in intensity after each event
  β (beta)  — decay rate of the excitation

Branching ratio:  n = α/β
  n < 1  →  process is stationary (does not explode)
  n ≈ 0  →  close to a Poisson process (no memory)
  n → 1  →  near-critical, heavy clustering

Log-likelihood:
  ℓ(μ,α,β) = Σᵢ log λ(tᵢ)  −  ∫₀ᵀ λ(t) dt
"""

def hawkes_intensity(t, T_history, mu, alpha, beta):
    """Evaluate λ(t) given past event times T_history."""
    past = T_history[T_history < t]
    return mu + np.sum(alpha * np.exp(-beta * (t - past)))


def hawkes_loglik(params, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return np.inf

    n   = len(T)
    ll  = 0.0
    R   = 0.0   # recursive kernel accumulator:  R_i = sum_{k<i} exp(-beta*(t_i - t_k))
    G   = 0.0   # tracks sum_{k} (1 - exp(-beta*(T[-1] - t_k))) for compensator

    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            R  = np.exp(-beta * dt) * (R + 1.0)   # update recursion
        lam = mu + alpha * R
        if lam <= 0:
            return np.inf
        ll += np.log(lam)

    # Compensator: mu*(T[-1]-T[0])  +  (alpha/beta)*sum_i(1 - exp(-beta*(T[-1]-t_i)))
    # Computed in one pass to avoid re-scanning T
    G = np.sum(1.0 - np.exp(-beta * (T[-1] - T)))
    compensator = mu * (T[-1] - T[0]) + (alpha / beta) * G

    return -(ll - compensator)

def hawkes_loglik_grad(params, T):
    """Returns (negative log-likelihood, gradient) as a tuple for scipy."""
    mu, alpha, beta = params
    if mu <= 0 or alpha < 0 or beta <= 0:
        return np.inf, np.zeros(3)

    n   = len(T)
    R   = 0.0    # kernel accumulator
    S   = 0.0    # for grad_beta:  sum_{k<i} (t_i-t_k)*exp(-beta*(t_i-t_k))

    ll       = 0.0
    d_mu     = 0.0
    d_alpha  = 0.0
    d_beta   = 0.0

    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            e  = np.exp(-beta * dt)
            S  = e * (S + (T[i-1] - T[0]))   # accumulate weighted decay for beta grad
            R  = e * (R + 1.0)
        lam = mu + alpha * R
        if lam <= 0:
            return np.inf, np.zeros(3)
        ll       += np.log(lam)
        d_mu     += 1.0 / lam
        d_alpha  += R   / lam
        d_beta   -= alpha * (R * (T[i] - T[0]) - S) / lam  # chain rule through R

    # Compensator gradients
    exp_terms = np.exp(-beta * (T[-1] - T))
    G         = np.sum(1.0 - exp_terms)
    d_mu     -= (T[-1] - T[0])
    d_alpha  -= G / beta
    d_beta   -= (alpha / beta) * (
            np.sum((T[-1] - T) * exp_terms) / 1.0
            - G / beta
    )

    return -(ll - (mu * (T[-1] - T[0]) + (alpha / beta) * G)), \
        -np.array([d_mu, d_alpha, d_beta])

def _make_inits(T, n_starts=8):
    """
    Empirical moment matching gives a good starting region:
      mu_0    ~ mean rate = n / (T[-1] - T[0])
      beta_0  ~ 1 / mean_inter_arrival  (characteristic decay scale)
      alpha_0 ~ 0.5 * beta_0            (branching ratio ~0.5 as neutral start)
    """
    mean_rate = len(T) / (T[-1] - T[0])
    mean_ia   = np.mean(np.diff(T))
    beta_0    = 1.0 / mean_ia if mean_ia > 0 else 1.0

    inits = []
    for scale in np.linspace(0.1, 0.9, n_starts):
        inits.append(np.array([
            mean_rate * scale,
            0.5 * beta_0 * scale,
            beta_0 * (0.5 + scale),
            ]))
    return inits


_HAWKES_WORKER_T = np.empty(0, dtype=float)


def _pool_initializer(T):
    """
    Seed a worker process with one ticker's event times.

    Shared pools can reuse this so each optimisation start only carries the
    parameter tuple, not the full timestamp array.
    """
    global _HAWKES_WORKER_T
    _HAWKES_WORKER_T = np.ascontiguousarray(T, dtype=float)


def _solve_hawkes_start(T, init, bounds):
    """Run one exponential Hawkes L-BFGS-B start and return a compact summary."""
    res = minimize(
        hawkes_loglik_grad, init, args=(T,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
    )
    return np.asarray(res.x, dtype=float), float(res.fun), bool(res.success)


def _run_one_hawkes_start(args):
    """Worker entrypoint for one exponential Hawkes start on shared worker data."""
    T = _HAWKES_WORKER_T
    if T.size == 0:
        raise RuntimeError("Exponential Hawkes worker pool was not initialised with T.")

    init, bounds = args
    return _solve_hawkes_start(T, init, bounds)

def fit_hawkes(T, label="", quiet=False, return_nll=False, _pool=None):
    """Fit exponential Hawkes model by MLE.

    Parameters
    ----------
    return_nll : bool
        If True, return ``(mu, alpha, beta, nll)`` instead of ``(mu, alpha, beta)``.
        Avoids a redundant ``hawkes_loglik`` call in callers that need the NLL.
    _pool : executor or None
        Optional shared worker pool already initialised with ``_pool_initializer``.
    """
    T = np.sort(np.asarray(T, dtype=float))
    T = T - T[0]                    # zero-index time (important for numerical stability)
    T = T[np.isfinite(T)]
    if len(T) < 20:
        _log(f"  [yellow]⚠[/yellow] Not enough events to fit Hawkes ({label}).", force=not quiet)
        return None

    best_res = None
    best_val = np.inf
    mean_ia = np.mean(np.diff(T))
    beta_max = 10.0 / mean_ia      # fastest meaningful decay ~ 10x the mean inter-arrival
    alpha_max = 0.99 * beta_max    # enforce branching ratio < 1 hard

    bounds = [
        (1e-6, None),        # mu
        (1e-6, alpha_max),   # alpha
        (1e-3, beta_max),    # beta
    ]
    starts = [np.asarray(init, dtype=float) for init in _make_inits(T)]
    all_res = []

    if _pool is not None:
        args_list = [(init, bounds) for init in starts]
        for x_opt, f_opt, success in _pool.map(_run_one_hawkes_start, args_list):
            all_res.append((x_opt, f_opt, success))
            if success and np.isfinite(f_opt) and f_opt < best_val:
                best_val = f_opt
                best_res = x_opt
    else:
        for init in starts:
            x_opt, f_opt, success = _solve_hawkes_start(T, init, bounds)
            all_res.append((x_opt, f_opt, success))
            if success and np.isfinite(f_opt) and f_opt < best_val:
                best_val = f_opt
                best_res = x_opt

    # Fallback: if no converged run, keep the best finite objective from same runs.
    if best_res is None:
        finite = [(x_opt, f_opt) for x_opt, f_opt, _ in all_res if np.isfinite(f_opt)]
        if not finite:
            _log(f"  [yellow]⚠[/yellow] Hawkes optimisation failed ({label}).", force=not quiet)
            return None
        best_res, best_val = min(finite, key=lambda item: item[1])

    mu, alpha, beta = map(float, best_res)
    br = alpha / beta

    if not quiet:
        _log(f"\n{'-' * 50}", force=True)
        _log(f"  Hawkes fit - {label}", force=True)
        _log(f"  mu (baseline)      = {mu:.5f}  events/sec", force=True)
        _log(f"  alpha (jump size)  = {alpha:.5f}", force=True)
        _log(f"  beta (decay rate)  = {beta:.5f}", force=True)
        _log(f"  Branching ratio    = {br:.4f}", force=True)
        if br >= 1:
            _log("  [yellow]⚠[/yellow] Branching ratio >= 1 -> non-stationary; check data quality.", force=True)
        else:
            _log(f"  -> ~{br * 100:.1f}% of events are triggered by previous events.", force=True)
        _log(f"{'-' * 50}\n", force=True)

    nll = float(best_val)
    if return_nll:
        return mu, alpha, beta, nll
    return mu, alpha, beta

def plot_hawkes_intensity(T, mu, alpha, beta, ticker, n_grid=2000, quiet=False):
    """
    Plot the fitted Hawkes intensity λ(t) against the raw event times.
    """
    T = np.sort(np.asarray(T, dtype=float))
    t_grid = np.linspace(T[0], T[-1], n_grid)

    # Evaluate intensity via a recursive accumulator in O(n_grid + n_events).
    lam = np.empty_like(t_grid)
    accum = 0.0
    prev_t = t_grid[0]
    j = 0
    n_events = len(T)

    for i, t in enumerate(t_grid):
        if i > 0:
            accum *= np.exp(-beta * (t - prev_t))

        while j < n_events and T[j] < t:
            accum += np.exp(-beta * (t - T[j]))
            j += 1

        lam[i] = mu + alpha * accum
        prev_t = t

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, lam, color=COLORS.get(ticker, "steelblue"), lw=1.2, label="λ(t)")
    ax.axhline(mu, color="red", ls="--", lw=1, label=f"Baseline μ = {mu:.4f}")

    # Rug plot of event times
    ax.plot(T, np.zeros_like(T) - 0.02 * lam.max(), "|",
            color="black", alpha=0.3, ms=6)

    ax.set_xlabel("Time (s from open)")
    ax.set_ylabel("Intensity λ(t)")
    ax.set_title(f"{ticker} — Fitted Hawkes Intensity  (α/β = {alpha/beta:.3f})",
                 fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"hawkes_intensity_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")


def plot_residual_qqplot(T, mu, alpha, beta, ticker, quiet=False):
    """
    Goodness-of-fit via the time-change theorem:
    The compensated times  Λ(tᵢ) = ∫₀^{tᵢ} λ(t) dt  should be
    i.i.d. Exponential(1) if the model is correct.

    Returns
    -------
    ks_stat : float   Kolmogorov–Smirnov statistic against Exp(1)
    """
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)

    # Compensator increments (corrected recursion)
    # Uses (1 + A) BEFORE the decay update, so that event t_{i-1}
    # contributes to the excitation integral over [t_{i-1}, t_i].
    residuals = np.empty(n - 1)
    A = 0.0
    for i in range(1, n):
        dt = T[i] - T[i - 1]
        e  = np.exp(-beta * dt)
        residuals[i - 1] = mu * dt + (alpha / beta) * (1 - e) * (1.0 + A)
        A = e * (A + 1.0)

    # KS test against Exp(1)
    res_pos = residuals[residuals > 0]
    ks_stat = float("nan")
    ks_pval = float("nan")
    if len(res_pos) >= 5:
        ks_stat, ks_pval = ks_1samp(res_pos, expon.cdf)

    # Q-Q plot against Exp(1)
    quantiles_emp = np.sort(residuals)
    quantiles_th  = -np.log(1 - np.linspace(0.01, 0.99, len(residuals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ks_label = (
        f"  (KS = {ks_stat:.3f}, p = {ks_pval:.3g})"
        if np.isfinite(ks_stat) and np.isfinite(ks_pval)
        else ""
    )
    fig.suptitle(f"{ticker} -- Hawkes Goodness-of-Fit{ks_label}", fontweight="bold")

    ax1.plot(quantiles_th, quantiles_emp, ".", alpha=0.4,
             color=COLORS.get(ticker, "steelblue"), ms=3)
    lim = max(quantiles_th.max(), quantiles_emp.max())
    ax1.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect fit")
    ax1.set_xlabel("Theoretical Exp(1) quantiles")
    ax1.set_ylabel("Empirical quantiles")
    ax1.set_title("Q-Q Plot (residual inter-arrivals)")
    ax1.legend(fontsize=9)

    ax2.hist(residuals, bins=40, density=True, color=COLORS.get(ticker, "steelblue"),
             alpha=0.7, edgecolor="white")
    xs = np.linspace(0, residuals.max(), 200)
    ax2.plot(xs, np.exp(-xs), "r--", lw=2, label="Exp(1)")
    ax2.set_xlabel("Residual inter-arrival")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution vs Exp(1)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"hawkes_qqplot_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")

    return ks_stat


# =============================================================================
# SECTION 6 — MAIN PIPELINE
# =============================================================================

def run_pipeline(tickers=None, start=START_DATE, end=END_DATE, data_path=DATA_PATH):
    """
    End-to-end pipeline: load -> visualise LOB -> stylised facts -> Hawkes fit.

    Parameters
    ----------
    tickers   : list of str   Ticker symbols (default: STOCKS = the 5 stocks)
    start     : str            "YYYY-MM-DD"
    end       : str            "YYYY-MM-DD"
    data_path : str            Folder with LOBSTER files
    """
    t0 = time.perf_counter()
    if tickers is None:
        tickers = STOCKS

    if HAVE_RICH and console is not None:
        console.print(Panel(
            f"[bold]Tickers[/bold] : {', '.join(tickers)}\n"
            f"[bold]Period[/bold]  : {start} -> {end}",
            title="[bold cyan]Main LOB/Hawkes Pipeline[/bold cyan]",
            border_style="cyan",
        ))
    else:
        _log("\n" + "=" * 65, force=True)
        _log("  STEP 0 - LOB Structure Diagram", force=True)
        _log("=" * 65, force=True)

    plot_lob_diagram()

    summaries = {}
    hawkes_params = {}
    hawkes_ks = {}

    def _run_one_ticker(ticker, progress=None, stage_task=None):
        stage_names = [
            "loaded", "LOB snapshot", "mid/spread", "event breakdown",
            "depth heatmap", "stylised facts", "VAR test", "Hawkes diagnostics",
        ]

        def _advance(ix):
            if progress is not None and stage_task is not None:
                progress.advance(stage_task)
                progress.update(stage_task, status=stage_names[ix])

        if progress is not None and stage_task is not None:
            progress.update(stage_task, status="loading data")

        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily = loader.load()

        if not daily:
            _log(f"  [yellow]⚠[/yellow] Skipping {ticker} (no data found).", force=True)
            if progress is not None and stage_task is not None:
                progress.update(stage_task, status="no data")
            return

        df = daily[0]
        t_open_buffer = df["Time"].min() + 3600
        t_close_buffer = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_open_buffer) & (df["Time"] <= t_close_buffer)].copy()
        summaries[ticker] = df
        _advance(0)

        plot_lob_snapshot(df, ticker, n_levels=5, title_extra=df.Date.iloc[0]); _advance(1)
        plot_midprice_and_spread(df, ticker); _advance(2)
        plot_event_breakdown(df, ticker); _advance(3)
        plot_depth_heatmap(df, ticker, n_levels=5); _advance(4)

        compute_stylised_facts(df, ticker, quiet=True)
        _advance(5)

        var_res = var_memory_test(df, ticker, quiet=True)
        _advance(6)

        mo = df[df["Type"] == 4]
        T = np.sort(mo["Time"].values.astype(float))
        T = T[np.isfinite(T)]

        if len(T) >= 20:
            params = fit_hawkes(T, label=f"{ticker} market orders", quiet=True)
            if params is not None:
                mu, alpha, beta = params
                hawkes_params[ticker] = params
                plot_hawkes_intensity(T, mu, alpha, beta, ticker, quiet=True)
                ks = plot_residual_qqplot(T, mu, alpha, beta, ticker, quiet=True)
                hawkes_ks[ticker] = ks
                if HAVE_RICH and console is not None:
                    ks_str = f"  KS={ks:.3f}" if np.isfinite(ks) else ""
                    _log(
                        f"  [bold]{ticker}[/bold]  VAR lag={var_res.k_ar if var_res is not None else 'n/a'}  "
                        f"|  Hawkes BR={alpha/beta:.3f}{ks_str}",
                        force=True,
                    )
        else:
            _log(f"  [yellow]⚠[/yellow] Not enough market orders for Hawkes fit ({ticker}).", force=True)

        _advance(7)
        if progress is not None and stage_task is not None:
            progress.update(stage_task, status="done")

    if HAVE_RICH and console is not None:
        set_quiet_logs(True)
        try:
            with Progress(
                SpinnerColumn(spinner_name="dots"),
                TextColumn("[bold blue]{task.description:<24}"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("[dim]·[/dim]"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[status]}"),
                console=console,
                refresh_per_second=10,
            ) as progress:
                outer = progress.add_task("tickers", total=len(tickers), status="")
                for ticker in tickers:
                    stage_task = progress.add_task(
                        f"[cyan]{ticker}[/cyan]",
                        total=8,
                        status="starting",
                    )
                    _run_one_ticker(ticker, progress=progress, stage_task=stage_task)
                    progress.advance(outer)
        finally:
            set_quiet_logs(False)
    else:
        for ticker in tickers:
            _log(f"\n{'=' * 65}", force=True)
            _log(f"  Loading {ticker} ...", force=True)
            _log(f"{'=' * 65}", force=True)
            _run_one_ticker(ticker)

    if len(summaries) > 1:
        _log("\n[bold cyan]Cross-stock comparison[/bold cyan]" if HAVE_RICH else "\nCross-stock comparison", force=True)
        plot_cross_stock_summary(summaries)
        plot_midprice_all(summaries)

    if hawkes_params:
        tks = list(hawkes_params.keys())
        mu_vals = [hawkes_params[t][0] for t in tks]
        br_vals = [hawkes_params[t][1] / hawkes_params[t][2] for t in tks]
        ks_vals = [hawkes_ks.get(t, float("nan")) for t in tks]
        has_ks  = any(np.isfinite(k) for k in ks_vals)

        n_panels = 3 if has_ks else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels + 1, 4))
        fig.suptitle("Hawkes Parameters — Cross-stock Comparison", fontweight="bold")

        ax1 = axes[0]
        ax1.bar(tks, mu_vals, color=[COLORS.get(t, "grey") for t in tks])
        ax1.set_title("Background Rate μ (events/sec)")
        ax1.set_ylabel("μ")

        ax2 = axes[1]
        ax2.bar(tks, br_vals, color=[COLORS.get(t, "grey") for t in tks])
        ax2.axhline(1, color="red", ls="--", lw=1, label="Stationarity boundary")
        ax2.set_title("Branching Ratio α/β")
        ax2.set_ylabel("α/β")
        ax2.legend(fontsize=9)

        annotate_axes = [ax1, ax2]

        if has_ks:
            ax3 = axes[2]
            ax3.bar(tks, ks_vals, color=[COLORS.get(t, "grey") for t in tks])
            ax3.set_title("KS Statistic vs Exp(1)")
            ax3.set_ylabel("KS stat (lower = better)")
            annotate_axes.append(ax3)

        for ax in annotate_axes:
            for bar in ax.patches:
                h = bar.get_height()
                if np.isfinite(h):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        h * 1.01,
                        f"{h:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        plt.tight_layout()
        fname = os.path.join(PLOTS_DIR, "hawkes_comparison.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        _log(f"Saved: {fname}", force=True)

    elapsed = time.perf_counter() - t0
    if HAVE_RICH and console is not None:
        console.print(Panel(
            f"[bold green]Done[/bold green] in [bold]{elapsed:.1f}s[/bold] · "
            f"plots -> [dim]{os.path.abspath(PLOTS_DIR)}[/dim]",
            border_style="green",
        ))
    else:
        _log(f"\nPipeline complete ({elapsed:.1f}s).", force=True)

    return summaries, hawkes_params, hawkes_ks
# =============================================================================
# SECTION 7 — STUDENT EXPERIMENTS
# =============================================================================
"""
Once you have run run_pipeline() above, try the following extensions:

─────────────────────────────────────────────────────────────────────────────
Experiment 1 — Bin-size sensitivity
─────────────────────────────────────────────────────────────────────────────
  for bin_length in [0.5, 1.0, 5.0, 30.0]:
      var_memory_test(df_AMZN, "AMZN", bin_length=bin_length)

  Does the selected VAR lag change?  What does this tell you about the
  time-scale of order-flow memory?

─────────────────────────────────────────────────────────────────────────────
Experiment 2 — Directional asymmetry
─────────────────────────────────────────────────────────────────────────────
  Fit separate Hawkes models to buy-initiated and sell-initiated trades:

  mo_buy  = df[(df.Type == 4) & (df.TradeDirection ==  1)].Time.values
  mo_sell = df[(df.Type == 4) & (df.TradeDirection == -1)].Time.values

  fit_hawkes(mo_buy,  "AMZN buy-side")
  fit_hawkes(mo_sell, "AMZN sell-side")

  Compare branching ratios.  Are buy and sell flows equally self-exciting?

─────────────────────────────────────────────────────────────────────────────
Experiment 3 — Power-law vs exponential kernel
─────────────────────────────────────────────────────────────────────────────
  Replace the exponential kernel  α·exp(−β·Δt)  with a power-law:
    h(Δt) = c / (1 + Δt/τ)^η

  Fit by adding a power_law_loglik() function and compare AIC values
  against the exponential model.

─────────────────────────────────────────────────────────────────────────────
Experiment 4 — De-drift the Hawkes  
─────────────────────────────────────────────────────────────────────────────
There seems to be a negative drift on this date for all the stocks, perhaps better
Hawkes would be  λ(t) = (μ_0 + μ_1 * t) + Σ α·exp(-β·(t - tᵢ))

Modify the code to add this new parameter and see if it improves your QQ plots.
"""

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # ── Quick-start ────────────────────────────────────────────────────────
    # Edit DATA_PATH, START_DATE, END_DATE at the top of the file, then run:
    #
    #   python main.py
    #
    # or, to run only a subset of stocks:
    #   summaries, params, ks = run_pipeline(
    #       tickers=["AMZN", "AAPL"],
    #       data_path="my_data/"
    #   )
    # ──────────────────────────────────────────────────────────────────────

    summaries, hawkes_params, hawkes_ks = run_pipeline()
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
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# =============================================================================
# AUTO-SAVE ALL MATPLOTLIB FIGURES
# =============================================================================
import os
from datetime import datetime

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# store original show
_original_show = plt.show

def _autosave_show(*args, **kwargs):
    """
    Automatically save every matplotlib figure before showing it.
    """
    figs = list(map(plt.figure, plt.get_fignums()))

    for i, fig in enumerate(figs):
        timestamp = datetime.now().strftime("%H%M%S_%f")
        filename = f"{PLOT_DIR}/plot_{timestamp}_{i}.png"
        fig.savefig(filename, dpi=140, bbox_inches="tight")

    _original_show(*args, **kwargs)

# monkey-patch matplotlib show()
plt.show = _autosave_show
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR

# ---------------------------------------------------------------------------
# Global display settings
# ---------------------------------------------------------------------------
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

# ─────────────────────────────────────────────────────────────────────────────
# 0.  CONFIGURE PATHS  (edit these two lines)
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH  = "data/"          # folder containing all LOBSTER CSV files from https://data.lobsterdata.com/info/DataSamples.php
START_DATE = "2012-06-21"     # first date to load (YYYY-MM-DD)
END_DATE   = "2012-06-21"     # last  date to load (YYYY-MM-DD)


# =============================================================================
# SECTION 1 — LOBSTER DATA LOADER
# =============================================================================
# ─────────────────────────────────────────────────────────────────────────────
# 1.1  Background: what does a Limit Order Book look like?
# ─────────────────────────────────────────────────────────────────────────────
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

            print(f"  Loading {self.ric}  {date_str} …")

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
            print(f"  ⚠  No data found for {self.ric} between {self.sDate} and {self.eDate}.")
            print(f"     Expected files in: {os.path.abspath(self.dataPath)}")

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
    plt.savefig("lob_diagram.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: lob_diagram.png")


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
    fname = f"lob_snapshot_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


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
    fname = f"midprice_spread_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


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
    fname = f"event_breakdown_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


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
    fname = f"depth_heatmap_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


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
    plt.savefig("cross_stock_summary.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved: cross_stock_summary.png")

# =============================================================================
# EXTRA PLOTS — Multivariate Hawkes diagnostics (Intensity + Branching heatmap)
# =============================================================================

def _mv_intensity_on_grid(times_list, mu, alpha, beta, t_grid):
    """
    Compute multivariate Hawkes intensity λ_i(t) on a time grid for all i.

    Model:
      λ_i(t) = μ_i + sum_j sum_{t_k^(j) < t} α_{j,i} exp(-β_{j,i}(t - t_k^(j)))

    alpha[source, target], beta[source, target]
    Returns: lam_grid shape (len(t_grid), d)
    """
    d = len(times_list)
    t_grid = np.asarray(t_grid, dtype=float)
    lam = np.zeros((len(t_grid), d), dtype=float)

    # Prepare per-dimension sorted times
    ts = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        a = np.sort(a)
        ts.append(a)

    # Merge events into one stream (time, source_dim)
    t_all, k_all = _mv_stack_events(ts)
    if len(t_all) == 0:
        # no events -> constant intensities
        lam[:] = mu.reshape(1, -1)
        return lam

    # Shift everything so grid starts at 0 for stability
    t0 = min(t_all[0], t_grid[0])
    t_all = t_all - t0
    t_grid0 = t_grid - t0

    # Recursion state R[source, target]
    R = np.zeros((d, d), dtype=float)
    last_t = 0.0
    e_idx = 0

    for g_idx, t in enumerate(t_grid0):
        # advance events up to this grid time
        while e_idx < len(t_all) and t_all[e_idx] <= t:
            te = t_all[e_idx]
            dt = te - last_t
            if dt < 0:
                dt = 0.0
            R *= np.exp(-beta * dt)
            src = k_all[e_idx]
            R[src, :] += 1.0
            last_t = te
            e_idx += 1

        # decay from last event time to current grid time
        dtg = t - last_t
        if dtg < 0:
            dtg = 0.0
        Rg = R * np.exp(-beta * dtg)

        # intensity for each target i
        # λ_i = μ_i + Σ_source α[source,i] * Rg[source,i]
        lam[g_idx, :] = mu + np.sum(alpha * Rg, axis=0)

    return lam


def plot_mv_hawkes_intensity(times_list, fit_result, label="", dim_names=None, n_grid=1500):
    """
    Plot λ_i(t) over time for each dimension i, plus event rugs.
    One figure per dimension (cleaner than overlaying all).
    """
    if fit_result is None:
        return
    mu = fit_result["mu"]
    alpha = fit_result["alpha"]
    beta = fit_result["beta"]

    d = len(times_list)
    if dim_names is None:
        dim_names = [f"dim{i}" for i in range(d)]

    # Build a plotting grid covering the observed window
    # Use global min/max across all dims
    all_times = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if len(a):
            all_times.append(a)
    if not all_times:
        return
    tmin = float(np.min([np.min(a) for a in all_times]))
    tmax = float(np.max([np.max(a) for a in all_times]))
    if tmax <= tmin:
        return

    t_grid = np.linspace(tmin, tmax, int(n_grid))
    lam_grid = _mv_intensity_on_grid(times_list, mu, alpha, beta, t_grid)

    # One figure per dimension
    for i in range(d):
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_grid, lam_grid[:, i])
        ax.set_title(f"Multivariate Hawkes intensity — {label} — {dim_names[i]}")
        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel(r"$\lambda_i(t)$")
        ax.grid(True, alpha=0.3)

        # Rug plot for events of this dimension
        ti = np.asarray(times_list[i], dtype=float)
        ti = ti[np.isfinite(ti)]
        if len(ti):
            # draw short vertical lines at y=0 up to a small fraction of max intensity
            y0 = 0.0
            y1 = 0.08 * float(np.nanmax(lam_grid[:, i]) if np.isfinite(np.nanmax(lam_grid[:, i])) else 1.0)
            ax.vlines(ti, y0, y1, linewidth=0.5)

        plt.tight_layout()
        plt.show()


def plot_branching_heatmap(A, dim_names=None, title="Branching matrix heatmap"):
    """
    Heatmap of branching matrix A where A[target, source] = alpha[source,target]/beta[source,target].
    """
    A = np.asarray(A, dtype=float)
    d = A.shape[0]
    if dim_names is None:
        dim_names = [f"dim{i}" for i in range(d)]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(A, aspect="equal")  # use default colormap
    ax.set_title(title)
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")

    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(dim_names, rotation=45, ha="right")
    ax.set_yticklabels(dim_names)

    # annotate numbers
    for i in range(d):
        for j in range(d):
            ax.text(j, i, f"{A[i, j]:.3f}", ha="center", va="center", fontsize=9)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    
# =============================================================================
# SECTION 3 — STYLISED FACTS
# =============================================================================

def compute_stylised_facts(df, ticker):
    """
    Plot inter-arrival time distribution and signed-move autocorrelation
    for market-order events in `df`.
    """
    mo = df[df["Type"] == 4].copy()
    if len(mo) < 10:
        print(f"  ⚠  Not enough market orders for {ticker} to compute stylised facts.")
        return

    T = mo["Time"].values
    inter_arrival = np.diff(T)

    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
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
    ax1.set_xlim(left=0)

    # ── Signed-move autocorrelation ───────────────────────────────────────
    max_lag = min(30, len(X) - 2)
    lags    = range(1, max_lag + 1)
    acf     = [np.corrcoef(X[:-k], X[k:])[0, 1] for k in lags]
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
    fname = f"stylised_facts_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# =============================================================================
# SECTION 4 — VAR MEMORY TEST
# =============================================================================

def var_memory_test(df, ticker, bin_length=1.0, max_lags=10):
    """
    Bin market-order events into `bin_length`-second windows and fit a
    Vector Auto-Regression to test for temporal dependence.

    Returns the fitted VAR result object.
    """
    mo   = df[df["Type"] == 4]
    T    = mo["Time"].values
    if len(T) < 50:
        print(f"  ⚠  Too few market orders for VAR test ({ticker}).")
        return None

    bins   = np.arange(T.min(), T.max(), bin_length)
    counts, _ = np.histogram(T, bins=bins)

    count_df = pd.DataFrame({"N": counts})
    # VAR requires ≥2 variables — add lagged columns as a second series
    count_df["N_lag1"] = count_df["N"].shift(1).fillna(0)
    model    = VAR(count_df)
    try:
        res = model.fit(maxlags=max_lags, ic="aic")
    except Exception as e:
        print(f"  ⚠  VAR fitting failed for {ticker}: {e}")
        return None

    print(f"\n{'='*60}")
    print(f"  VAR Memory Test — {ticker}  (bin = {bin_length}s)")
    print(f"  Selected lag order : {res.k_ar}")
    print(f"  AIC                : {res.aic:.2f}")
    print(f"  If lag > 0 → market-order arrivals have memory (consistent with Hawkes).")
    print(f"{'='*60}\n")
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

def fit_hawkes(T, label=""):
    T = np.sort(np.asarray(T, dtype=float))
    T = T - T[0]                    # zero-index time (important for numerical stability)
    T = T[np.isfinite(T)]
    if len(T) < 20:
        print(f"  ⚠  Not enough events to fit Hawkes ({label}).")
        return None

    best_res, best_val = None, np.inf
    mean_ia   = np.mean(np.diff(T))
    beta_max  = 10.0 / mean_ia      # fastest meaningful decay ~ 10x the mean inter-arrival
    alpha_max = 0.99 * beta_max     # enforce branching ratio < 1 hard

    bounds = [
        (1e-6, None),        # mu
        (1e-6, alpha_max),   # alpha
        (1e-3, beta_max),    # beta
    ]
    for init in _make_inits(T):
        res = minimize(
            hawkes_loglik_grad, init, args=(T,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
        )
        if res.fun < best_val and res.success:
            best_val = res.fun
            best_res = res

    # Fallback: accept best non-converged result if nothing succeeded
    if best_res is None:
        best_res = min(
            [minimize(hawkes_loglik_grad, init, args=(T,), method="L-BFGS-B",
                      jac=True,
                      bounds=[(1e-6, None), (1e-6, None), (1e-3, None)])
             for init in _make_inits(T)],
            key=lambda r: r.fun
        )

    mu, alpha, beta = best_res.x
    br = alpha / beta

    print(f"\n{'─'*50}")
    print(f"  Hawkes fit — {label}")
    print(f"  μ (baseline)      = {mu:.5f}  events/sec")
    print(f"  α (jump size)     = {alpha:.5f}")
    print(f"  β (decay rate)    = {beta:.5f}")
    print(f"  Branching ratio   = {br:.4f}")
    if br >= 1:
        print("Branching ratio ≥ 1 → non-stationary; check data quality.")
    else:
        print(f"  → ~{br*100:.1f}% of events are triggered by previous events.")
    print(f"{'─'*50}\n")
    return mu, alpha, beta


def plot_hawkes_intensity(T, mu, alpha, beta, ticker, n_grid=2000):
    """
    Plot the fitted Hawkes intensity λ(t) against the raw event times.
    """
    T = np.sort(np.asarray(T, dtype=float))
    t_grid = np.linspace(T[0], T[-1], n_grid)

    # Evaluate intensity at each grid point (vectorised for speed)
    lam = np.array([
        mu + alpha * np.sum(np.exp(-beta * (t - T[T < t])))
        for t in t_grid
    ])

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
    fname = f"hawkes_intensity_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


def plot_residual_qqplot(T, mu, alpha, beta, ticker):
    """
    Goodness-of-fit via the time-change theorem:
    The compensated times  Λ(tᵢ) = ∫₀^{tᵢ} λ(t) dt  should be
    i.i.d. Exponential(1) if the model is correct.
    """
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)

    # Compensator increments (recursive)
    A      = 0.0
    Lambda = np.zeros(n)
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            A  = A * np.exp(-beta * dt)
            Lambda[i] = Lambda[i - 1] + mu * dt + (alpha / beta) * (1 - np.exp(-beta * dt)) * A
        A += 1.0

    residuals = np.diff(Lambda)   # should be ~Exp(1)

    # Q-Q plot against Exp(1)
    quantiles_emp = np.sort(residuals)
    quantiles_th  = -np.log(1 - np.linspace(0.01, 0.99, len(residuals)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ticker} — Hawkes Goodness-of-Fit", fontweight="bold")

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
    fname = f"hawkes_qqplot_{ticker}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")

# =============================================================================
# SECTION 5B — MULTIVARIATE HAWKES (2D / 4D, exponential kernels)
# =============================================================================
"""
Multivariate Hawkes with exponential kernels:

For i in {0..d-1}:
  λ_i(t) = μ_i + Σ_j Σ_{t_k^(j) < t} α_{j,i} exp(-β_{j,i} (t - t_k^(j)))

Parameterization in code:
  alpha[source, target] = α_{source,target}
  beta[source, target]  = β_{source,target}

Branching matrix A:
  A[target, source] = α_{source,target} / β_{source,target}

Stability (stationarity): spectral radius ρ(A) < 1
"""

def _mv_stack_events(times_list):
    """Merge event times from each dimension into one sorted stream."""
    all_t = []
    all_k = []
    for k, arr in enumerate(times_list):
        if arr is None:
            continue
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        for t in a:
            all_t.append(float(t))
            all_k.append(int(k))
    if len(all_t) == 0:
        return np.array([]), np.array([], dtype=int)
    idx = np.argsort(all_t)
    return np.asarray(all_t)[idx], np.asarray(all_k, dtype=int)[idx]


def _branching_matrix(alpha, beta):
    """
    Return branching matrix A where A[target, source] = alpha[source,target]/beta[source,target]
    """
    d = alpha.shape[0]
    A = np.zeros((d, d), dtype=float)
    for source in range(d):
        for target in range(d):
            A[target, source] = alpha[source, target] / beta[source, target]
    return A


def _spectral_radius(M):
    vals = np.linalg.eigvals(M)
    return float(np.max(np.abs(vals)))


def mv_hawkes_negloglik(params, times_list, d, stability_rho_max=0.99):
    """
    Negative log-likelihood for d-dim Hawkes with exp kernels.

    Param layout:
      params = [mu_0..mu_{d-1},
                alpha_{0,0}..alpha_{d-1,d-1} (row-major: source-major),
                beta_{0,0}..beta_{d-1,d-1}  (row-major)]
      alpha[source, target], beta[source, target]
    """
    params = np.asarray(params, dtype=float)
    mu = params[:d]
    alpha_flat = params[d:d + d*d]
    beta_flat  = params[d + d*d:]
    alpha = alpha_flat.reshape((d, d))
    beta  = beta_flat.reshape((d, d))

    # constraints
    if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
        return np.inf

    # stability penalty
    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)
    if not np.isfinite(rho) or rho >= stability_rho_max:
        return np.inf

    # stack events and shift to start at 0
    t_all, k_all = _mv_stack_events(times_list)
    if len(t_all) < 30:
        return np.inf

    t0 = t_all[0]
    t_all = t_all - t0
    T_end = t_all[-1]

    # shift per-dimension times for compensator
    shifted = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        a = np.sort(a)
        shifted.append(a - t0 if len(a) else np.array([]))

    # recursion state R[source, target]
    R = np.zeros((d, d), dtype=float)
    last_t = 0.0
    ll = 0.0

    for t, k in zip(t_all, k_all):
        dt = t - last_t
        if dt < 0:
            return np.inf

        # decay all R by dt (pairwise beta)
        R *= np.exp(-beta * dt)

        # intensity for target dimension k
        lam_k = mu[k] + np.sum(alpha[:, k] * R[:, k])
        if lam_k <= 0 or not np.isfinite(lam_k):
            return np.inf
        ll += np.log(lam_k)

        # after event happens: source=k adds 1 to R[k, target] for all targets
        R[k, :] += 1.0
        last_t = t

    # compensator (closed form)
    # ∫0^T λ_i(t) dt = μ_i T + Σ_j (α_{j,i}/β_{j,i}) Σ_{t_k^(j)} [1 - exp(-β_{j,i}(T - t_k^(j)))]
    comp = float(np.sum(mu) * T_end)
    for source in range(d):
        tj = shifted[source]
        if len(tj) == 0:
            continue
        delta = (T_end - tj)
        for target in range(d):
            b = beta[source, target]
            comp += (alpha[source, target] / b) * np.sum(1.0 - np.exp(-b * delta))

    return -(ll - comp)


def mv_hawkes_negloglik_and_grad(params, times_list, d, stability_rho_max=0.99):
    """
    Negative log-likelihood + numerical gradient (finite differences).
    Robust and easy to maintain for 2D/4D.

    Returns (nll, grad)
    """
    params = np.asarray(params, dtype=float)
    base = mv_hawkes_negloglik(params, times_list, d, stability_rho_max=stability_rho_max)

    grad = np.zeros_like(params)
    if not np.isfinite(base):
        return np.inf, grad

    eps = 1e-5
    for i in range(len(params)):
        p2 = params.copy()
        p2[i] += eps
        v2 = mv_hawkes_negloglik(p2, times_list, d, stability_rho_max=stability_rho_max)
        grad[i] = (v2 - base) / eps

    return base, grad


def _mv_make_inits(times_list, d, n_starts=6):
    """Heuristic multi-start inits for mv Hawkes."""
    t_all, _ = _mv_stack_events(times_list)
    t_all = np.sort(t_all[np.isfinite(t_all)])
    if len(t_all) < 30:
        return []
    T_end = t_all[-1] - t_all[0]
    T_end = max(T_end, 1e-6)

    # per-dim observed rates
    rates = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        rates.append(len(a) / T_end)
    rates = np.asarray(rates, dtype=float)
    rates = np.maximum(rates, 1e-4)

    # global beta scale from mean inter-arrival
    ia = np.diff(t_all)
    ia = ia[ia > 0]
    beta0 = 1.0 / (np.mean(ia) if len(ia) else 1.0)
    beta0 = float(np.clip(beta0, 0.1, 50.0))

    inits = []
    for s in np.linspace(0.2, 0.9, n_starts):
        mu0 = rates * s
        # modest excitation so stability likely holds
        alpha0 = np.full((d, d), 0.12 * beta0 * s, dtype=float)
        beta0m = np.full((d, d), beta0 * (0.7 + 0.6*s), dtype=float)
        p = np.concatenate([mu0, alpha0.reshape(-1), beta0m.reshape(-1)])
        inits.append(p)
    return inits


def fit_mv_hawkes(times_list, label="", dim_names=None, stability_rho_max=0.99, n_starts=6, verbose=True):
    """
    Fit a d-dimensional Hawkes (exp kernels) by MLE with stability penalty.

    Returns dict with keys:
      mu, alpha, beta, A (branching matrix), rho, success
    """
    d = len(times_list)
    if dim_names is None:
        dim_names = [f"dim{i}" for i in range(d)]

    # count events
    n_total = 0
    for x in times_list:
        a = np.asarray(x, dtype=float) if x is not None else np.array([])
        a = a[np.isfinite(a)]
        n_total += len(a)

    if n_total < 30:
        if verbose:
            print(f"  ⚠  Not enough events for multivariate Hawkes ({label}). total={n_total}")
        return None

    # bounds: mu>0, alpha>=0, beta>0
    bounds = []
    bounds += [(1e-8, None)] * d          # mu
    bounds += [(0.0, None)] * (d*d)       # alpha
    bounds += [(1e-6, None)] * (d*d)      # beta

    best = None
    best_val = np.inf

    inits = _mv_make_inits(times_list, d, n_starts=n_starts)
    if not inits:
        if verbose:
            print(f"  ⚠  Could not create initial points ({label}).")
        return None

    obj = lambda p: mv_hawkes_negloglik_and_grad(p, times_list, d, stability_rho_max=stability_rho_max)

    for x0 in inits:
        res = minimize(fun=lambda p: obj(p)[0],
                       x0=x0,
                       method="L-BFGS-B",
                       jac=lambda p: obj(p)[1],
                       bounds=bounds,
                       options={"maxiter": 400})
        if res.success and np.isfinite(res.fun) and res.fun < best_val:
            best_val = float(res.fun)
            best = res

    if best is None:
        if verbose:
            print(f"  ✗ Multivariate Hawkes failed ({label}).")
        return None

    p = best.x
    mu = p[:d]
    alpha = p[d:d+d*d].reshape((d, d))
    beta  = p[d+d*d:].reshape((d, d))
    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)

    if verbose:
        print(f"\n  ✓ Multivariate Hawkes fit: {label}")
        print(f"    dims: {dim_names}")
        print("    μ: " + ", ".join([f"{dim_names[i]}={mu[i]:.4g}" for i in range(d)]))
        print(f"    spectral radius ρ(A) = {rho:.4f}  (stability requires < 1)")
        print("    Branching matrix A (rows=target, cols=source):")
        header = "           " + " ".join([f"{n:>10s}" for n in dim_names])
        print("    " + header)
        for i in range(d):
            row = " ".join([f"{A[i,j]:10.4f}" for j in range(d)])
            print(f"    {dim_names[i]:>10s} {row}")

    return {"mu": mu, "alpha": alpha, "beta": beta, "A": A, "rho": rho, "success": best.success}


def extract_up_down_midprice_times(df):
    """
    Up/Down mid-price move event times from best bid/ask.
    Robust to different LOBSTER column naming conventions.

    Returns:
      up_t, dn_t  (sorted numpy arrays of event times)
    """
    # Common variants
    ask_candidates = ["AskPrice1", "Ask Price 1", "ask_price_1", "AskPrice_1", "ASKPRICE1"]
    bid_candidates = ["BidPrice1", "Bid Price 1", "bid_price_1", "BidPrice_1", "BIDPRICE1"]

    def _pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        # case-insensitive fallback
        lower_map = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    ask_col = _pick(ask_candidates)
    bid_col = _pick(bid_candidates)

    if ask_col is None or bid_col is None:
        preview = list(df.columns[:40])
        raise KeyError(
            f"Cannot find best bid/ask columns. "
            f"Tried ask={ask_candidates}, bid={bid_candidates}. "
            f"First columns: {preview}"
        )

    mid = (df[ask_col].values.astype(float) + df[bid_col].values.astype(float)) / 2.0
    dmid = np.diff(mid, prepend=mid[0])

    t = df["Time"].values.astype(float)
    up_t = np.sort(t[dmid > 0])
    dn_t = np.sort(t[dmid < 0])

    return up_t, dn_t

# =============================================================================
# SECTION 6 — MAIN PIPELINE
# =============================================================================

def run_pipeline(tickers=None, start=START_DATE, end=END_DATE, data_path=DATA_PATH):
    """
    End-to-end pipeline: load → visualise LOB → stylised facts → Hawkes fit.
    Also saves ALL plots (including new 2D/4D Hawkes plots) to ./plots
    """
    import os

    PLOT_DIR = "plots"
    os.makedirs(PLOT_DIR, exist_ok=True)

    def _save_current_fig(filename):
        """Save the current active matplotlib figure into plots/ and close it."""
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=140, bbox_inches="tight")
        plt.close()

    if tickers is None:
        tickers = STOCKS

    # ── 2.0  Static LOB diagram ────────────────────────────────────────────
    print("\n" + "="*65)
    print("  STEP 0 — LOB Structure Diagram")
    print("="*65)
    plot_lob_diagram()
    # if plot_lob_diagram() already saves internally, this will create a duplicate save only if it leaves a figure open
    if plt.get_fignums():
        _save_current_fig("lob_diagram_extra.png")

    # ── 2.1  Per-stock loading and LOB plots ────────────────────────────────
    summaries = {}
    hawkes_params = {}

    for ticker in tickers:
        print(f"\n{'='*65}")
        print(f"  Loading {ticker} …")
        print(f"{'='*65}")

        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily  = loader.load()

        if not daily:
            print(f"  ⚠  Skipping {ticker} (no data found).")
            continue

        df = daily[0]

        # ── Visualisations ────────────────────────────────────────────────
        plot_lob_snapshot(df, ticker)
        if plt.get_fignums():
            _save_current_fig(f"lob_snapshot_{ticker}_extra.png")

        plot_midprice_and_spread(df, ticker)
        if plt.get_fignums():
            _save_current_fig(f"midprice_spread_{ticker}_extra.png")

        plot_event_breakdown(df, ticker)
        if plt.get_fignums():
            _save_current_fig(f"event_breakdown_{ticker}_extra.png")

        plot_depth_heatmap(df, ticker)
        if plt.get_fignums():
            _save_current_fig(f"depth_heatmap_{ticker}_extra.png")

        # ── Stylised facts on market orders ───────────────────────────────
        compute_stylised_facts(df, ticker)
        # stylised facts might generate multiple figures; save them all
        for i, fnum in enumerate(list(plt.get_fignums())):
            plt.figure(fnum)
            _save_current_fig(f"stylised_{ticker}_{i}.png")

        # ── VAR memory test (counts binned) ────────────────────────────────
        var_memory_test(df, ticker, bin_length=1.0, max_lags=10)
        for i, fnum in enumerate(list(plt.get_fignums())):
            plt.figure(fnum)
            _save_current_fig(f"var_test_{ticker}_{i}.png")

              # ── Summary stats for cross-stock comparison ───────────────────────
        def _pick_col(cands):
            for c in cands:
                if c in df.columns:
                    return c
            # case-insensitive fallback
            lower_map = {c.lower(): c for c in df.columns}
            for c in cands:
                if c.lower() in lower_map:
                    return lower_map[c.lower()]
            return None

        ask_p = _pick_col(["AskPrice1", "Ask Price 1", "AskPrice_1", "ask_price_1"])
        bid_p = _pick_col(["BidPrice1", "Bid Price 1", "BidPrice_1", "bid_price_1"])
        ask_s = _pick_col(["AskSize1",  "Ask Size 1",  "AskSize_1",  "ask_size_1"])
        bid_s = _pick_col(["BidSize1",  "Bid Size 1",  "BidSize_1",  "bid_size_1"])

        if ask_p is None or bid_p is None or ask_s is None or bid_s is None:
            raise KeyError(
                "Cannot find top-of-book columns for summary stats. "
                f"Found columns start: {list(df.columns[:40])}"
            )

        spread = (df[ask_p] - df[bid_p]).mean()
        depth1 = (df[ask_s] + df[bid_s]).mean()
        mo_cnt = int((df["Type"] == 4).sum())
        summaries[ticker] = {
            "mean_spread": float(spread),
            "mean_top_depth": float(depth1),
            "market_orders": mo_cnt
        }

        # ── Hawkes fits (1D, 2D, 4D) ───────────────────────────────────────

        # (A) 1D Hawkes on ALL market orders (original behaviour)
        mo_all = df[df["Type"] == 4]
        T_all  = np.sort(mo_all["Time"].values.astype(float))
        if len(T_all) >= 20:
            print(f"\n  ── 1D Hawkes Fit ({ticker}) — all market orders ──")
            params_all = fit_hawkes(T_all, label=f"{ticker} all market orders")
            if params_all is not None:
                mu1, alpha1, beta1 = params_all
                hawkes_params[ticker] = params_all

                plot_hawkes_intensity(T_all, mu1, alpha1, beta1, ticker)
                if plt.get_fignums():
                    _save_current_fig(f"hawkes_1d_intensity_{ticker}.png")

                plot_residual_qqplot(T_all, mu1, alpha1, beta1, ticker)
                if plt.get_fignums():
                    _save_current_fig(f"hawkes_1d_qqplot_{ticker}.png")
        else:
            print(f"  ⚠  Not enough market orders for 1D Hawkes ({ticker}).")

        # (B) 1D Hawkes on BUY-initiated vs SELL-initiated market orders
        mo_buy  = df[(df["Type"] == 4) & (df["TradeDirection"] == 1)]["Time"].values
        mo_sell = df[(df["Type"] == 4) & (df["TradeDirection"] == -1)]["Time"].values
        mo_buy  = np.sort(mo_buy[np.isfinite(mo_buy)].astype(float))
        mo_sell = np.sort(mo_sell[np.isfinite(mo_sell)].astype(float))

        print(f"\n  ── 1D Hawkes Fit ({ticker}) — buy vs sell market orders ──")
        br_buy = br_sell = None

        if len(mo_buy) >= 20:
            p_buy = fit_hawkes(mo_buy, label=f"{ticker} buy-side MO")
            if p_buy is not None:
                br_buy = p_buy[1] / p_buy[2]
                print(f"    buy-side branching ratio α/β = {br_buy:.4f}")
        else:
            print("    ⚠  Not enough buy-side MOs to fit.")

        if len(mo_sell) >= 20:
            p_sell = fit_hawkes(mo_sell, label=f"{ticker} sell-side MO")
            if p_sell is not None:
                br_sell = p_sell[1] / p_sell[2]
                print(f"    sell-side branching ratio α/β = {br_sell:.4f}")
        else:
            print("    ⚠  Not enough sell-side MOs to fit.")

        if (br_buy is not None) and (br_sell is not None):
            if abs(br_buy - br_sell) < 0.05:
                print("    → Buy and sell flows look similarly self-exciting (branching ratios close).")
            elif br_buy > br_sell:
                print("    → Buy flow appears MORE self-exciting than sell flow (higher branching ratio).")
            else:
                print("    → Sell flow appears MORE self-exciting than buy flow (higher branching ratio).")

        # (C) 2D Hawkes: Up vs Down mid-price moves
        up_t, dn_t = extract_up_down_midprice_times(df)
        print(f"\n  ── 2D Hawkes Fit ({ticker}) — up/down mid-price moves ──")
        res2 = fit_mv_hawkes([up_t, dn_t],
                             label=f"{ticker} 2D price moves",
                             dim_names=["UP", "DOWN"],
                             stability_rho_max=0.99,
                             n_starts=6,
                             verbose=True)
        if res2 is not None:
            # intensity plots (should create 2 separate figures)
            plot_mv_hawkes_intensity([up_t, dn_t], res2,
                                     label=f"{ticker} 2D (UP/DOWN)",
                                     dim_names=["UP", "DOWN"])
            # save all currently open figures produced by this call
            for fnum in list(plt.get_fignums()):
                fig = plt.figure(fnum)
                title = fig.axes[0].get_title() if fig.axes else ""
                if "UP" in title:
                    _save_current_fig(f"mv2d_intensity_UP_{ticker}.png")
                elif "DOWN" in title:
                    _save_current_fig(f"mv2d_intensity_DOWN_{ticker}.png")
                else:
                    _save_current_fig(f"mv2d_intensity_{ticker}_{fnum}.png")

            # branching heatmap (one figure)
            plot_branching_heatmap(res2["A"],
                                   dim_names=["UP", "DOWN"],
                                   title=f"{ticker} 2D branching matrix A (target rows, source cols)")
            if plt.get_fignums():
                _save_current_fig(f"mv2d_branching_{ticker}.png")

        # (D) 4D Hawkes: Up/Down + BuyMO/SellMO
        print(f"\n  ── 4D Hawkes Fit ({ticker}) — price moves + market orders ──")
        res4 = fit_mv_hawkes([up_t, dn_t, mo_buy, mo_sell],
                             label=f"{ticker} 4D (UP,DOWN,BUY_MO,SELL_MO)",
                             dim_names=["UP", "DOWN", "BUY_MO", "SELL_MO"],
                             stability_rho_max=0.99,
                             n_starts=5,
                             verbose=True)
        if res4 is not None:
            plot_mv_hawkes_intensity([up_t, dn_t, mo_buy, mo_sell], res4,
                                     label=f"{ticker} 4D",
                                     dim_names=["UP", "DOWN", "BUY_MO", "SELL_MO"])
            # save all currently open figures produced by this call
            for fnum in list(plt.get_fignums()):
                fig = plt.figure(fnum)
                title = fig.axes[0].get_title() if fig.axes else ""
                if "UP" in title:
                    _save_current_fig(f"mv4d_intensity_UP_{ticker}.png")
                elif "DOWN" in title:
                    _save_current_fig(f"mv4d_intensity_DOWN_{ticker}.png")
                elif "BUY_MO" in title:
                    _save_current_fig(f"mv4d_intensity_BUY_MO_{ticker}.png")
                elif "SELL_MO" in title:
                    _save_current_fig(f"mv4d_intensity_SELL_MO_{ticker}.png")
                else:
                    _save_current_fig(f"mv4d_intensity_{ticker}_{fnum}.png")

            plot_branching_heatmap(res4["A"],
                                   dim_names=["UP", "DOWN", "BUY_MO", "SELL_MO"],
                                   title=f"{ticker} 4D branching matrix A (target rows, source cols)")
            if plt.get_fignums():
                _save_current_fig(f"mv4d_branching_{ticker}.png")

    # ── 2.2  Cross-stock comparison ────────────────────────────────────────
    if len(summaries) > 1:
        print(f"\n{'='*65}")
        print("  STEP FINAL — Cross-stock Comparison")
        print(f"{'='*65}")
        plot_cross_stock_summary(summaries)
        if plt.get_fignums():
            _save_current_fig("cross_stock_summary_extra.png")

    # ── Hawkes parameter comparison (for the 1D all-market-orders fits) ─────
    if hawkes_params:
        tks      = list(hawkes_params.keys())
        mu_vals  = [hawkes_params[t][0] for t in tks]
        br_vals  = [hawkes_params[t][1] / hawkes_params[t][2] for t in tks]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("Hawkes Parameters — Cross-stock Comparison", fontweight="bold")

        ax1.bar(tks, mu_vals,  color=[COLORS.get(t, "grey") for t in tks])
        ax1.set_title("Background Rate μ (events/sec)")
        ax1.set_ylabel("μ")

        ax2.bar(tks, br_vals, color=[COLORS.get(t, "grey") for t in tks])
        ax2.axhline(1, color="red", ls="--", lw=1, label="Stationarity boundary")
        ax2.set_title("Branching Ratio α/β")
        ax2.set_ylabel("α/β")
        ax2.legend(fontsize=9)

        for ax in [ax1, ax2]:
            for bar in ax.patches:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * 1.01,
                        f"{bar.get_height():.3f}",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        # Save into plots/ (instead of project root)
        plt.savefig(os.path.join(PLOT_DIR, "hawkes_comparison.png"), dpi=140, bbox_inches="tight")
        plt.close()
        print(f"Saved: {os.path.join(PLOT_DIR, 'hawkes_comparison.png')}")

    print("\n✓  Pipeline complete.")
    return summaries, hawkes_params


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
    #   summaries, params = run_pipeline(
    #       tickers=["AMZN", "AAPL"],
    #       data_path="my_data/"
    #   )
    # ──────────────────────────────────────────────────────────────────────

    summaries, hawkes_params = run_pipeline()

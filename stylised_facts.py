"""
Stylised facts plots with Weibull inter-arrival baseline.

Recreates the 2-panel stylised-facts figure from main.py:
1) Market-order inter-arrival histogram (log-log) with Weibull overlay
2) Signed trade-size autocorrelation
"""

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, gamma, expon

from main import (
    Loader,
    STOCKS,
    COLORS,
    DATA_PATH,
    START_DATE,
    END_DATE,
)

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

PLOTS_DIR = os.path.join("plots", "stylised_facts_multi_dist")
os.makedirs(PLOTS_DIR, exist_ok=True)


def compute_stylised_facts_multi_dist(df, ticker):
    """Plot stylised facts with a Weibull fit for inter-arrival times."""
    mo = df[df["Type"] == 4].copy()
    if len(mo) < 10:
        print(f"  Not enough market orders for {ticker} to compute stylised facts.")
        return None

    inter_arrival = np.diff(mo["Time"].values)
    inter_arrival = inter_arrival[inter_arrival > 0]
    if len(inter_arrival) < 10:
        print(f"  Not enough positive inter-arrivals for {ticker}.")
        return None

    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(f"{ticker} - Stylised Facts ({df.Date.iloc[0]})",
                 fontsize=13, fontweight="bold")

    # Inter-arrival histogram (log-log) + fitted distribution overlays.
    ia_min = inter_arrival.min()
    ia_max = inter_arrival.max()
    if ia_min == ia_max:
        ia_max = ia_min * 1.0001
    log_bins = np.logspace(np.log10(ia_min), np.log10(ia_max), 60)

    hist_vals, _, _ = ax1.hist(inter_arrival, bins=log_bins, density=True,
                               color=COLORS.get(ticker, "steelblue"),
                               alpha=0.7, edgecolor="white")

    k, _, lam = weibull_min.fit(inter_arrival, floc=0)
    a_g, _, scale_g = gamma.fit(inter_arrival, floc=0)
    s_l, _, scale_l = lognorm.fit(inter_arrival, floc=0)
    _, scale_e = expon.fit(inter_arrival, floc=0)

    xs = np.logspace(np.log10(ia_min), np.log10(ia_max), 400)
    ys_weibull = weibull_min.pdf(xs, c=k, loc=0, scale=lam)
    ys_gamma = gamma.pdf(xs, a=a_g, loc=0, scale=scale_g)
    ys_lognorm = lognorm.pdf(xs, s=s_l, loc=0, scale=scale_l)
    ys_exp = expon.pdf(xs, loc=0, scale=scale_e)

    ax1.plot(xs, ys_weibull, color="#d62728", ls="--", lw=2,
             label=f"Weibull(k={k:.2f}, lambda={lam:.2f})")
    ax1.plot(xs, ys_gamma, color="#9467bd", ls="-.", lw=1.8,
             label=f"Gamma(a={a_g:.2f}, theta={scale_g:.2f})")
    ax1.plot(xs, ys_lognorm, color="#2ca02c", ls=":", lw=2.2,
             label=f"Lognormal(s={s_l:.2f}, scale={scale_l:.2f})")
    ax1.plot(xs, ys_exp, color="#ff7f0e", ls="-", lw=1.8,
             label=f"Exponential(lambda={1.0/scale_e:.2f})")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Inter-arrival time (s) - log")
    ax1.set_ylabel("Density")
    ax1.set_title("Market-order Inter-arrival Times")
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-5, top=1e4)

    # Signed-move autocorrelation.
    max_lag = min(30, len(X) - 2)
    if max_lag >= 1:
        lags = range(1, max_lag + 1)
        acf = [np.corrcoef(X[:-k], X[k:])[0, 1] for k in lags]
        ax2.bar(lags, acf, color=COLORS.get(ticker, "steelblue"), alpha=0.8)
        ax2.axhline(0, color="black", lw=0.8)
        ci = 1.96 / np.sqrt(len(X))
        ax2.axhline(ci, color="red", ls="--", lw=1, label="95% CI (i.i.d.)")
        ax2.axhline(-ci, color="red", ls="--", lw=1)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.axhline(0, color="black", lw=0.8)

    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Signed Trade-size Autocorrelation")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"stylised_facts_multi_dist_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")
    return {
        "ticker": ticker,
        "inter_arrival": inter_arrival,
        "acf": np.asarray(acf) if max_lag >= 1 else np.array([]),
        "gamma_a": float(a_g),
        "gamma_scale": float(scale_g),
        "exp_scale": float(scale_e),
        "date": str(df.Date.iloc[0]),
    }


def _ccdf(x):
    xs = np.sort(x)
    y = 1.0 - np.arange(1, len(xs) + 1) / len(xs)
    return xs, y


def plot_cross_ticker_stylised_comparison(results):
    """Compare stylised facts across tickers in one figure."""
    if len(results) < 2:
        print("  Skipping cross-ticker comparison: need at least 2 fitted tickers.")
        return

    dates = sorted({res["date"] for res in results.values()})
    date_label = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(f"Cross-Ticker Stylised Facts Comparison ({date_label})",
                 fontsize=13, fontweight="bold")

    # Left panel: inter-arrival CCDF + fitted Gamma and Exponential survival by ticker.
    for ticker, res in results.items():
        color = COLORS.get(ticker, None)
        ia = res["inter_arrival"]
        xs_emp, ys_emp = _ccdf(ia)
        ax1.loglog(xs_emp, ys_emp, lw=1.4, color=color, alpha=0.9, label=f"{ticker} empirical")

        xw = np.logspace(np.log10(ia.min()), np.log10(ia.max()), 300)
        yg = gamma.sf(xw, a=res["gamma_a"], loc=0, scale=res["gamma_scale"])
        ye = expon.sf(xw, loc=0, scale=res["exp_scale"])
        ax1.loglog(xw, yg, ls="--", lw=1.3, color=color, alpha=0.9, label=f"{ticker} Gamma")
        ax1.loglog(xw, ye, ls=":", lw=1.3, color=color, alpha=0.9, label=f"{ticker} Exponential")

    ax1.set_xlabel("Inter-arrival time (s)")
    ax1.set_ylabel("P(Δt > x)")
    ax1.set_title("Inter-arrival Tails (CCDF)")
    ax1.legend(fontsize=8, ncol=2)
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-4, top=1.1e0)
    

    # Right panel: signed-move ACF by ticker.
    for ticker, res in results.items():
        acf = res["acf"]
        if len(acf) == 0:
            continue
        lags = np.arange(1, len(acf) + 1)
        ax2.plot(lags, acf, marker="o", ms=2.5, lw=1.2,
                 color=COLORS.get(ticker, None), label=ticker)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Signed Trade-size ACF by Ticker")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "stylised_facts_multi_dist_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def run_stylised_facts_multi_dist(tickers=None, start=START_DATE, end=END_DATE,
                               data_path=DATA_PATH):
    """Run Weibull stylised-facts plots for each requested ticker."""
    if tickers is None:
        tickers = STOCKS

    comparison_results = {}
    for ticker in tickers:
        print(f"\n{'=' * 65}")
        print(f"  Loading {ticker} ...")
        print(f"{'=' * 65}")

        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily = loader.load()
        if not daily:
            print(f"  Skipping {ticker} (no data found).")
            continue

        df = daily[0]
        t_open_buffer = df["Time"].min() + 3600
        t_close_buffer = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_open_buffer) & (df["Time"] <= t_close_buffer)].copy()

        res = compute_stylised_facts_multi_dist(df, ticker)
        if res is not None:
            comparison_results[ticker] = res

    plot_cross_ticker_stylised_comparison(comparison_results)

    print(f"\nDone. Weibull stylised plots are in: {os.path.abspath(PLOTS_DIR)}")


if __name__ == "__main__":
    run_stylised_facts_multi_dist()
    

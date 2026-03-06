# =============================================================================
# Experiment 2 — Directional Asymmetry in Hawkes Processes
# =============================================================================
# Fits separate Hawkes models to buy- and sell-initiated market orders, then
# compares branching ratios and other parameters to answer:
#   "Are buy and sell order flows equally self-exciting?"
#
# Run after main.py has been set up (DATA_PATH, START_DATE, END_DATE).
# This script imports helpers from main.py and saves all plots to PLOTS_DIR.
#
# Usage:
#   python experiment_2.py
# ==============#===============================================================

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import ks_2samp

# ── Import shared infrastructure from main.py ────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as main_module
from main import (
    Loader,
    fit_hawkes,
    hawkes_loglik_grad,
    plot_hawkes_intensity,
    plot_residual_qqplot,
    DATA_PATH,
    START_DATE,
    END_DATE,
    COLORS,
)

# Save all Experiment 2 figures in a dedicated subfolder.
PLOTS_DIR = os.path.join("plots", "experiment_2")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Ensure plotting helpers imported from main.py also write to this folder.
main_module.PLOTS_DIR = PLOTS_DIR

plt.rcParams.update({
    "figure.figsize"  : (12, 5),
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "font.size"       : 11,
})

BUY_COLOR  = "#2196F3"   # blue  — buy-initiated
SELL_COLOR = "#F44336"   # red   — sell-initiated


# =============================================================================
# HELPER: compute compensator residuals (for GoF / KS test)
# =============================================================================

def compute_residuals(T, mu, alpha, beta):
    """Return Λ-transformed inter-arrivals; should be ~Exp(1) if fit is good."""
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)
    A, Lambda = 0.0, np.zeros(n)
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            A  = A * np.exp(-beta * dt)
            Lambda[i] = Lambda[i-1] + mu*dt + (alpha/beta)*(1 - np.exp(-beta*dt))*A
        A += 1.0
    return np.diff(Lambda)   # n-1 residuals, should be Exp(1)


# =============================================================================
# PLOT 1: Side-by-side intensity traces (buy vs sell on the same time axis)
# =============================================================================

def plot_dual_intensity(T_buy, params_buy, T_sell, params_sell, ticker, n_grid=1500):
    """
    Two-panel figure: fitted λ(t) for buy orders (top) and sell orders (bottom),
    sharing the same time axis so the viewer can see intra-day co-movement.
    """
    mu_b,  al_b,  be_b  = params_buy
    mu_s,  al_s,  be_s  = params_sell

    # Common time grid spanning both streams
    t_min = min(T_buy[0],  T_sell[0])
    t_max = max(T_buy[-1], T_sell[-1])
    tg    = np.linspace(t_min, t_max, n_grid)

    def intensity(T, mu, alpha, beta, t_grid):
        return np.array([
            mu + alpha * np.sum(np.exp(-beta * (t - T[T < t])))
            for t in t_grid
        ])

    lam_b = intensity(T_buy,  mu_b, al_b, be_b, tg)
    lam_s = intensity(T_sell, mu_s, al_s, be_s, tg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f"{ticker} — Hawkes Intensity: Buy vs Sell  (α/β shown)",
                 fontweight="bold")

    ax1.plot(tg, lam_b, color=BUY_COLOR,  lw=1.1, label=f"Buy  α/β={al_b/be_b:.3f}")
    ax1.axhline(mu_b, color=BUY_COLOR, ls="--", lw=0.9, alpha=0.6, label=f"μ={mu_b:.4f}")
    ax1.plot(T_buy, np.zeros_like(T_buy) - 0.02*lam_b.max(),
             "|", color=BUY_COLOR, alpha=0.25, ms=5)
    ax1.set_ylabel("λ(t)  [buy]")
    ax1.legend(fontsize=9, loc="upper right")

    ax2.plot(tg, lam_s, color=SELL_COLOR, lw=1.1, label=f"Sell α/β={al_s/be_s:.3f}")
    ax2.axhline(mu_s, color=SELL_COLOR, ls="--", lw=0.9, alpha=0.6, label=f"μ={mu_s:.4f}")
    ax2.plot(T_sell, np.zeros_like(T_sell) - 0.02*lam_s.max(),
             "|", color=SELL_COLOR, alpha=0.25, ms=5)
    ax2.set_ylabel("λ(t)  [sell]")
    ax2.set_xlabel("Time (s from open)")
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_dual_intensity_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# PLOT 2: Parameter comparison bar chart (μ, α, β, branching ratio)
# =============================================================================

def plot_parameter_comparison(params_buy, params_sell, ticker):
    """
    Four-panel bar chart comparing μ, α, β, and α/β across buy and sell sides.
    """
    mu_b,  al_b,  be_b  = params_buy
    mu_s,  al_s,  be_s  = params_sell
    br_b, br_s = al_b / be_b, al_s / be_s

    labels  = ["Buy", "Sell"]
    colors  = [BUY_COLOR, SELL_COLOR]

    param_data = [
        (r"$\mu$  (baseline rate, events/s)", [mu_b, mu_s]),
        (r"$\alpha$  (jump size)",             [al_b, al_s]),
        (r"$\beta$  (decay rate)",             [be_b, be_s]),
        (r"Branching ratio  $\alpha/\beta$",   [br_b, br_s]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"{ticker} — Hawkes Parameter Comparison: Buy vs Sell",
                 fontweight="bold")

    for ax, (title, vals) in zip(axes, param_data):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() * 1.02,
                    f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        # Red stationarity line on branching ratio panel
        if "Branching" in title:
            ax.axhline(1, color="red", ls="--", lw=1, label="Stationarity\nboundary")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_param_comparison_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# PLOT 3: Overlaid Q-Q plots (buy and sell residuals on the same axes)
# =============================================================================

def plot_joint_qqplot(T_buy, params_buy, T_sell, params_sell, ticker):
    """
    Single figure with two panels:
      Left  — Q-Q plot of residuals for buy and sell overlaid
      Right — Histogram of residuals for both sides vs Exp(1)
    """
    res_b = compute_residuals(T_buy,  *params_buy)
    res_s = compute_residuals(T_sell, *params_sell)

    q_th_b = -np.log(1 - np.linspace(0.01, 0.99, len(res_b)))
    q_th_s = -np.log(1 - np.linspace(0.01, 0.99, len(res_s)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{ticker} — Goodness-of-Fit Q-Q: Buy vs Sell", fontweight="bold")

    # Q-Q
    ax1.plot(q_th_b, np.sort(res_b), ".", color=BUY_COLOR,  alpha=0.4, ms=3, label="Buy")
    ax1.plot(q_th_s, np.sort(res_s), ".", color=SELL_COLOR, alpha=0.4, ms=3, label="Sell")
    lim = max(q_th_b.max(), q_th_s.max(), np.sort(res_b).max(), np.sort(res_s).max())
    ax1.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect fit")
    ax1.set_xlabel("Theoretical Exp(1) quantiles")
    ax1.set_ylabel("Empirical quantiles")
    ax1.set_title("Q-Q Plot")
    ax1.legend(fontsize=9)

    # Density histogram
    bins = np.linspace(0, min(res_b.max(), res_s.max(), 8), 50)
    ax2.hist(res_b, bins=bins, density=True, color=BUY_COLOR,  alpha=0.5, label="Buy",
             edgecolor="white")
    ax2.hist(res_s, bins=bins, density=True, color=SELL_COLOR, alpha=0.5, label="Sell",
             edgecolor="white")
    xs = np.linspace(0, bins[-1], 200)
    ax2.plot(xs, np.exp(-xs), "k--", lw=2, label="Exp(1)")
    ax2.set_xlabel("Residual inter-arrival Λ")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_joint_qqplot_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# PLOT 4: Event rate over the trading day (buy vs sell counts per 5-min bin)
# =============================================================================

def plot_intraday_rate(T_buy, T_sell, ticker, bin_secs=300):
    """
    Bar chart of buy and sell market-order counts in fixed time bins
    (default 5 minutes) to show intra-day clustering patterns.
    """
    t_start = min(T_buy[0],  T_sell[0])
    t_end   = max(T_buy[-1], T_sell[-1])
    edges   = np.arange(t_start, t_end + bin_secs, bin_secs)
    centres = (edges[:-1] + edges[1:]) / 2

    cnt_b, _ = np.histogram(T_buy,  bins=edges)
    cnt_s, _ = np.histogram(T_sell, bins=edges)

    # Convert centres from seconds-since-midnight to HH:MM strings
    def fmt_time(s):
        h, m = divmod(int(s) // 60, 60)
        return f"{h:02d}:{m:02d}"

    x_labels = [fmt_time(c) for c in centres]
    tick_gap  = max(1, len(x_labels) // 10)  # show ~10 ticks

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(centres))
    w = 0.4
    ax.bar(x - w/2, cnt_b, width=w, color=BUY_COLOR,  alpha=0.8, label="Buy")
    ax.bar(x + w/2, cnt_s, width=w, color=SELL_COLOR, alpha=0.8, label="Sell")

    ax.set_xticks(x[::tick_gap])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), tick_gap)],
                       rotation=45, ha="right")
    ax.set_xlabel(f"Time of day  ({bin_secs//60}-min bins)")
    ax.set_ylabel("Market-order count")
    ax.set_title(f"{ticker} — Intraday Buy vs Sell Order Rate", fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_intraday_rate_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# PLOT 5: Stylised facts panel (inter-arrival + signed-move autocorrelation)
# =============================================================================

def plot_interarrival_hist(T_buy, T_sell, df, ticker):
    """
    Log-binned histogram density of inter-arrival times on log-log axes.

    LOBSTER timestamps can contain ties (multiple events at the same second).
    np.diff on tied timestamps yields exact zeros which cannot be shown on a
    log x-axis and also distort density estimates, so we filter them out.
    """
    ia_b = np.diff(np.sort(T_buy))
    ia_s = np.diff(np.sort(T_sell))

    # Drop zero inter-arrivals from timestamp ties.
    ia_b = ia_b[ia_b > 0]
    ia_s = ia_s[ia_s > 0]
    if len(ia_b) == 0 or len(ia_s) == 0:
        print(f"  Skipping inter-arrival histogram for {ticker}: no positive inter-arrivals.")
        return

    all_ia = np.concatenate([ia_b, ia_s])
    log_bins = np.logspace(np.log10(all_ia.min()), np.log10(all_ia.max()), 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(f"{ticker} — Stylised Facts", fontsize=13, fontweight="bold")

    ax1.hist(ia_b, bins=log_bins, density=True, color=BUY_COLOR, alpha=0.45,
             edgecolor="white", label="Buy inter-arrivals")
    ax1.hist(ia_s, bins=log_bins, density=True, color=SELL_COLOR, alpha=0.45,
             edgecolor="white", label="Sell inter-arrivals")

    # Exponential PDF references (memoryless baseline), one per side.
    lam_b = 1.0 / ia_b.mean()
    lam_s = 1.0 / ia_s.mean()
    x_ref = np.linspace(0, np.percentile(all_ia, 97), 300)
    ax1.plot(x_ref, lam_b * np.exp(-lam_b * x_ref), color=BUY_COLOR, ls="--", lw=1.5,
             label=f"Buy Exp(λ={lam_b:.2f})")
    ax1.plot(x_ref, lam_s * np.exp(-lam_s * x_ref), color=SELL_COLOR, ls="--", lw=1.5,
             label=f"Sell Exp(λ={lam_s:.2f})")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Inter-arrival time (s) - log")
    ax1.set_ylabel("Density")
    ax1.set_title("Market-order Inter-arrival Times")
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-5, top=1e4)

    # Signed-move autocorrelation panel (same structure as main.py stylised facts)
    mo = df[df["Type"] == 4].copy()
    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values
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
    fname = os.path.join(PLOTS_DIR, f"exp2_stylised_facts_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# =============================================================================
# SUMMARY PRINT
# =============================================================================

def print_summary(params_buy, params_sell, T_buy, T_sell):
    mu_b,  al_b,  be_b  = params_buy
    mu_s,  al_s,  be_s  = params_sell
    br_b, br_s = al_b / be_b, al_s / be_s

    res_b = compute_residuals(T_buy,  *params_buy)
    res_s = compute_residuals(T_sell, *params_sell)
    ks_stat, ks_p = ks_2samp(res_b, res_s)

    print("\n" + "="*60)
    print("  EXPERIMENT 2 — DIRECTIONAL ASYMMETRY SUMMARY")
    print("="*60)
    print(f"  {'Parameter':<28} {'Buy':>10}  {'Sell':>10}")
    print(f"  {'-'*50}")
    print(f"  {'N events':<28} {len(T_buy):>10d}  {len(T_sell):>10d}")
    print(f"  {'μ (baseline, events/s)':<28} {mu_b:>10.5f}  {mu_s:>10.5f}")
    print(f"  {'α (excitation jump)':<28} {al_b:>10.5f}  {al_s:>10.5f}")
    print(f"  {'β (decay rate)':<28} {be_b:>10.5f}  {be_s:>10.5f}")
    print(f"  {'Branching ratio α/β':<28} {br_b:>10.4f}  {br_s:>10.4f}")
    print(f"  {'Mean inter-arrival (s)':<28} {np.mean(np.diff(T_buy)):>10.4f}  {np.mean(np.diff(T_sell)):>10.4f}")
    print(f"\n  KS test on residuals (buy vs sell):")
    print(f"    statistic = {ks_stat:.4f},  p-value = {ks_p:.4f}")
    if ks_p < 0.05:
        print("    → Residual distributions differ significantly (p < 0.05).")
    else:
        print("    → No significant difference in residuals (p ≥ 0.05).")

    print("\n  Interpretation:")
    diff = abs(br_b - br_s)
    if diff < 0.02:
        print("  → Buy and sell branching ratios are very similar.")
        print("     Order flow appears roughly symmetric in self-excitation.")
    elif br_b > br_s:
        print(f"  → Buy side is MORE self-exciting (α/β buy={br_b:.4f} > sell={br_s:.4f}).")
        print("     Buy orders cluster more tightly — possibly momentum-driven.")
    else:
        print(f"  → Sell side is MORE self-exciting (α/β sell={br_s:.4f} > buy={br_b:.4f}).")
        print("     Sell orders cluster more tightly — possibly liquidation / panic-selling.")
    print("="*60 + "\n")


# =============================================================================
# PLOT 6: Cross-stock branching ratio comparison (all tickers)
# =============================================================================

def plot_cross_stock_branching(all_results, tickers):
    """
    Grouped bar chart: buy vs sell branching ratio for every ticker.
    Makes it easy to see which stocks have the largest directional asymmetry.
    """
    br_buy  = [all_results[t]["params_buy"][1]  / all_results[t]["params_buy"][2]  for t in tickers]
    br_sell = [all_results[t]["params_sell"][1] / all_results[t]["params_sell"][2] for t in tickers]

    x = np.arange(len(tickers))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(x - w/2, br_buy,  width=w, color=BUY_COLOR,  alpha=0.85,
                    label="Buy α/β",  edgecolor="white")
    bars_s = ax.bar(x + w/2, br_sell, width=w, color=SELL_COLOR, alpha=0.85,
                    label="Sell α/β", edgecolor="white")

    ax.axhline(1, color="red", ls="--", lw=1.2, label="Stationarity boundary")

    for bars in [bars_b, bars_s]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("Branching ratio  α/β")
    ax.set_title("Cross-stock Hawkes Branching Ratio — Buy vs Sell",
                 fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "exp2_cross_stock_branching.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fname}")


# =============================================================================
# MAIN
# =============================================================================

def run_experiment_2(tickers=None, start=START_DATE, end=END_DATE,
                     data_path=DATA_PATH):
    """
    Run Experiment 2 for every ticker in `tickers` (default: all 5 stocks).
    Produces per-stock plots plus a cross-stock branching ratio comparison.
    """
    from main import STOCKS
    t0 = time.perf_counter()
    if tickers is None:
        tickers = STOCKS   # ["AMZN", "AAPL", "GOOG", "MSFT", "INTC"]

    all_results   = {}   # ticker -> {params_buy, params_sell, mo_buy, mo_sell}
    fitted_tickers = []

    for ticker in tickers:
        t_stock = time.perf_counter()
        print(f"\n{'='*60}")
        print(f"  Experiment 2 — Directional Asymmetry  [{ticker}]")
        print(f"{'='*60}")

        # ── Load data ────────────────────────────────────────────────────────
        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily  = loader.load()

        if not daily:
            print(f"  ⚠  No data found for {ticker}. Skipping.")
            continue

        df = daily[0]
        # Drop first and last hour (same buffer as the main pipeline)
        t_buf_lo = df["Time"].min() + 3600
        t_buf_hi = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_buf_lo) & (df["Time"] <= t_buf_hi)].copy()

        # ── Extract buy and sell market-order timestamps ──────────────────────
        mo_buy  = df[(df["Type"] == 4) & (df["TradeDirection"] ==  1)]["Time"].values
        mo_sell = df[(df["Type"] == 4) & (df["TradeDirection"] == -1)]["Time"].values

        mo_buy  = np.sort(mo_buy [np.isfinite(mo_buy )])
        mo_sell = np.sort(mo_sell[np.isfinite(mo_sell)])

        print(f"\n  Market orders — buy: {len(mo_buy)}, sell: {len(mo_sell)}")

        if len(mo_buy) < 20 or len(mo_sell) < 20:
            print("  ⚠  Not enough events to fit Hawkes models. Skipping.")
            continue

        # ── Fit Hawkes models ─────────────────────────────────────────────────
        print("\n  Fitting buy-side Hawkes …")
        params_buy  = fit_hawkes(mo_buy,  label=f"{ticker} buy-side")

        print("\n  Fitting sell-side Hawkes …")
        params_sell = fit_hawkes(mo_sell, label=f"{ticker} sell-side")

        if params_buy is None or params_sell is None:
            print("  ⚠  Hawkes fit failed for one or both sides. Skipping.")
            continue

        all_results[ticker] = dict(params_buy=params_buy, params_sell=params_sell,
                                   mo_buy=mo_buy, mo_sell=mo_sell)
        fitted_tickers.append(ticker)

        # ── Per-stock summary ─────────────────────────────────────────────────
        print_summary(params_buy, params_sell, mo_buy, mo_sell)

        # ── Per-stock plots ───────────────────────────────────────────────────
        print(f"\n  Generating plots for {ticker} …")

        # Individual intensity + Q-Q (main.py helpers)
        plot_hawkes_intensity(mo_buy,  *params_buy,  ticker=f"{ticker}_buy")
        plot_hawkes_intensity(mo_sell, *params_sell, ticker=f"{ticker}_sell")
        plot_residual_qqplot(mo_buy,  *params_buy,  ticker=f"{ticker}_buy")
        plot_residual_qqplot(mo_sell, *params_sell, ticker=f"{ticker}_sell")

        # Dual intensity (shared time axis)
        plot_dual_intensity(mo_buy, params_buy, mo_sell, params_sell, ticker)

        # Parameter bar chart
        plot_parameter_comparison(params_buy, params_sell, ticker)

        # Overlaid Q-Q + residual histogram
        plot_joint_qqplot(mo_buy, params_buy, mo_sell, params_sell, ticker)

        # Intra-day count rate
        plot_intraday_rate(mo_buy, mo_sell, ticker)

        # Stylised facts panel (inter-arrival + signed-move ACF)
        plot_interarrival_hist(mo_buy, mo_sell, df, ticker)

        print(f"  [{ticker}] done in {time.perf_counter() - t_stock:.1f}s")

    # ── Cross-stock comparison (all fitted tickers together) ──────────────────
    if len(fitted_tickers) > 1:
        print(f"\n{'='*60}")
        print("  Cross-stock branching ratio comparison …")
        plot_cross_stock_branching(all_results, fitted_tickers)

    elapsed = time.perf_counter() - t0
    print(f"\n  ✓  Experiment 2 complete  ({elapsed:.1f}s).  All plots in: {os.path.abspath(PLOTS_DIR)}/")
    return all_results


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_experiment_2()   # runs all 5 stocks by default

# =============================================================================
# Experiment 2 - Directional Asymmetry in Hawkes Processes
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
# =============================================================================

import os
import sys
import time

# Force a non-GUI backend to avoid Tk thread teardown issues with live progress.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

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

# Import shared infrastructure from main.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as main_module
from main import (
    Loader,
    fit_hawkes,
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
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

BUY_COLOR = "#2196F3"   # blue  - buy-initiated
SELL_COLOR = "#F44336"  # red   - sell-initiated

console = Console() if HAVE_RICH else None
_QUIET_LOGS = False


def set_quiet_logs(flag):
    global _QUIET_LOGS
    _QUIET_LOGS = bool(flag)


def _log(msg, force=False):
    if _QUIET_LOGS and not force:
        return
    if HAVE_RICH and console is not None:
        console.print(msg)
    else:
        print(msg)


def _intensity_on_grid(T, mu, alpha, beta, t_grid):
    """O(n_grid + n_events) intensity evaluation using recursive decay."""
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

    return lam


def _acf_fast(x, max_lag):
    if max_lag < 1 or len(x) < 2:
        return np.array([])
    xc = x - x.mean()
    var = float(np.dot(xc, xc))
    if var == 0.0:
        return np.zeros(max_lag)
    full = np.correlate(xc, xc, mode="full")
    mid = len(full) // 2
    return full[mid + 1: mid + 1 + max_lag] / var


# =============================================================================
# HELPER: compute compensator residuals (for GoF / KS test)
# =============================================================================

def compute_residuals(T, mu, alpha, beta):
    """Return Lambda-transformed inter-arrivals; should be ~Exp(1) if fit is good."""
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)
    A, Lambda = 0.0, np.zeros(n)
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            A = A * np.exp(-beta * dt)
            Lambda[i] = Lambda[i - 1] + mu * dt + (alpha / beta) * (1 - np.exp(-beta * dt)) * A
        A += 1.0
    return np.diff(Lambda)


# =============================================================================
# PLOT 1: Side-by-side intensity traces
# =============================================================================

def plot_dual_intensity(T_buy, params_buy, T_sell, params_sell, ticker, n_grid=1500):
    mu_b, al_b, be_b = params_buy
    mu_s, al_s, be_s = params_sell

    t_min = min(T_buy[0], T_sell[0])
    t_max = max(T_buy[-1], T_sell[-1])
    tg = np.linspace(t_min, t_max, n_grid)

    lam_b = _intensity_on_grid(T_buy, mu_b, al_b, be_b, tg)
    lam_s = _intensity_on_grid(T_sell, mu_s, al_s, be_s, tg)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    fig.suptitle(f"{ticker} - Hawkes Intensity: Buy vs Sell  (alpha/beta shown)", fontweight="bold")

    ax1.plot(tg, lam_b, color=BUY_COLOR, lw=1.1, label=f"Buy  alpha/beta={al_b/be_b:.3f}")
    ax1.axhline(mu_b, color=BUY_COLOR, ls="--", lw=0.9, alpha=0.6, label=f"mu={mu_b:.4f}")
    ax1.plot(T_buy, np.zeros_like(T_buy) - 0.02 * lam_b.max(), "|", color=BUY_COLOR, alpha=0.25, ms=5)
    ax1.set_ylabel("lambda(t) [buy]")
    ax1.legend(fontsize=9, loc="upper right")

    ax2.plot(tg, lam_s, color=SELL_COLOR, lw=1.1, label=f"Sell alpha/beta={al_s/be_s:.3f}")
    ax2.axhline(mu_s, color=SELL_COLOR, ls="--", lw=0.9, alpha=0.6, label=f"mu={mu_s:.4f}")
    ax2.plot(T_sell, np.zeros_like(T_sell) - 0.02 * lam_s.max(), "|", color=SELL_COLOR, alpha=0.25, ms=5)
    ax2.set_ylabel("lambda(t) [sell]")
    ax2.set_xlabel("Time (s from open)")
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_dual_intensity_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"  Saved: {fname}")


# =============================================================================
# PLOT 2: Parameter comparison bar chart
# =============================================================================

def plot_parameter_comparison(params_buy, params_sell, ticker):
    mu_b, al_b, be_b = params_buy
    mu_s, al_s, be_s = params_sell
    br_b, br_s = al_b / be_b, al_s / be_s

    labels = ["Buy", "Sell"]
    colors = [BUY_COLOR, SELL_COLOR]

    param_data = [
        (r"$\mu$  (baseline rate, events/s)", [mu_b, mu_s]),
        (r"$\alpha$  (jump size)", [al_b, al_s]),
        (r"$\beta$  (decay rate)", [be_b, be_s]),
        (r"Branching ratio  $\alpha/\beta$", [br_b, br_s]),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(f"{ticker} - Hawkes Parameter Comparison: Buy vs Sell", fontweight="bold")

    for ax, (title, vals) in zip(axes, param_data):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
        ax.set_title(title, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
        if "Branching" in title:
            ax.axhline(1, color="red", ls="--", lw=1, label="Stationarity\nboundary")
            ax.legend(fontsize=8)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_param_comparison_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"  Saved: {fname}")


# =============================================================================
# PLOT 3: Overlaid Q-Q plots
# =============================================================================

def plot_joint_qqplot(T_buy, params_buy, T_sell, params_sell, ticker):
    res_b = compute_residuals(T_buy, *params_buy)
    res_s = compute_residuals(T_sell, *params_sell)

    q_th_b = -np.log(1 - np.linspace(0.01, 0.99, len(res_b)))
    q_th_s = -np.log(1 - np.linspace(0.01, 0.99, len(res_s)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{ticker} - Goodness-of-Fit Q-Q: Buy vs Sell", fontweight="bold")

    ax1.plot(q_th_b, np.sort(res_b), ".", color=BUY_COLOR, alpha=0.4, ms=3, label="Buy")
    ax1.plot(q_th_s, np.sort(res_s), ".", color=SELL_COLOR, alpha=0.4, ms=3, label="Sell")
    lim = max(q_th_b.max(), q_th_s.max(), np.sort(res_b).max(), np.sort(res_s).max())
    ax1.plot([0, lim], [0, lim], "k--", lw=1.5, label="Perfect fit")
    ax1.set_xlabel("Theoretical Exp(1) quantiles")
    ax1.set_ylabel("Empirical quantiles")
    ax1.set_title("Q-Q Plot")
    ax1.legend(fontsize=9)

    bins = np.linspace(0, min(res_b.max(), res_s.max(), 8), 50)
    ax2.hist(res_b, bins=bins, density=True, color=BUY_COLOR, alpha=0.5, label="Buy", edgecolor="white")
    ax2.hist(res_s, bins=bins, density=True, color=SELL_COLOR, alpha=0.5, label="Sell", edgecolor="white")
    xs = np.linspace(0, bins[-1], 200)
    ax2.plot(xs, np.exp(-xs), "k--", lw=2, label="Exp(1)")
    ax2.set_xlabel("Residual inter-arrival Lambda")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_joint_qqplot_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"  Saved: {fname}")


# =============================================================================
# PLOT 4: Event rate over trading day
# =============================================================================

def plot_intraday_rate(T_buy, T_sell, ticker, bin_secs=300):
    t_start = min(T_buy[0], T_sell[0])
    t_end = max(T_buy[-1], T_sell[-1])
    edges = np.arange(t_start, t_end + bin_secs, bin_secs)
    centres = (edges[:-1] + edges[1:]) / 2

    cnt_b, _ = np.histogram(T_buy, bins=edges)
    cnt_s, _ = np.histogram(T_sell, bins=edges)

    def fmt_time(s):
        h, m = divmod(int(s) // 60, 60)
        return f"{h:02d}:{m:02d}"

    x_labels = [fmt_time(c) for c in centres]
    tick_gap = max(1, len(x_labels) // 10)

    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(centres))
    w = 0.4
    ax.bar(x - w / 2, cnt_b, width=w, color=BUY_COLOR, alpha=0.8, label="Buy")
    ax.bar(x + w / 2, cnt_s, width=w, color=SELL_COLOR, alpha=0.8, label="Sell")

    ax.set_xticks(x[::tick_gap])
    ax.set_xticklabels([x_labels[i] for i in range(0, len(x_labels), tick_gap)], rotation=45, ha="right")
    ax.set_xlabel(f"Time of day ({bin_secs // 60}-min bins)")
    ax.set_ylabel("Market-order count")
    ax.set_title(f"{ticker} - Intraday Buy vs Sell Order Rate", fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_intraday_rate_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"  Saved: {fname}")


# =============================================================================
# PLOT 5: Stylised facts panel
# =============================================================================

def plot_interarrival_hist(T_buy, T_sell, df, ticker):
    ia_b = np.diff(np.sort(T_buy))
    ia_s = np.diff(np.sort(T_sell))

    ia_b = ia_b[ia_b > 0]
    ia_s = ia_s[ia_s > 0]
    if len(ia_b) == 0 or len(ia_s) == 0:
        _log(f"  Skipping inter-arrival histogram for {ticker}: no positive inter-arrivals.", force=True)
        return

    all_ia = np.concatenate([ia_b, ia_s])
    log_bins = np.logspace(np.log10(all_ia.min()), np.log10(all_ia.max()), 60)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle(f"{ticker} - Stylised Facts", fontsize=13, fontweight="bold")

    ax1.hist(ia_b, bins=log_bins, density=True, color=BUY_COLOR, alpha=0.45, edgecolor="white", label="Buy inter-arrivals")
    ax1.hist(ia_s, bins=log_bins, density=True, color=SELL_COLOR, alpha=0.45, edgecolor="white", label="Sell inter-arrivals")

    lam_b = 1.0 / ia_b.mean()
    lam_s = 1.0 / ia_s.mean()
    x_ref = np.linspace(0, np.percentile(all_ia, 97), 300)
    ax1.plot(x_ref, lam_b * np.exp(-lam_b * x_ref), color=BUY_COLOR, ls="--", lw=1.5, label=f"Buy Exp(lambda={lam_b:.2f})")
    ax1.plot(x_ref, lam_s * np.exp(-lam_s * x_ref), color=SELL_COLOR, ls="--", lw=1.5, label=f"Sell Exp(lambda={lam_s:.2f})")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Inter-arrival time (s) - log")
    ax1.set_ylabel("Density")
    ax1.set_title("Market-order Inter-arrival Times")
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-5, top=1e4)

    mo = df[df["Type"] == 4].copy()
    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values
    max_lag = min(30, len(X) - 2)

    if max_lag >= 1:
        lags = np.arange(1, max_lag + 1)
        acf = _acf_fast(X, max_lag)
        ax2.bar(lags, acf, color=COLORS.get(ticker, "steelblue"), alpha=0.8)
        ax2.axhline(0, color="black", lw=0.8)
        ci = 1.96 / np.sqrt(len(X))
        ax2.axhline(ci, color="red", ls="--", lw=1, label="95% CI (i.i.d.)")
        ax2.axhline(-ci, color="red", ls="--", lw=1)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center", transform=ax2.transAxes)
        ax2.axhline(0, color="black", lw=0.8)

    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Signed Trade-size Autocorrelation")

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"exp2_stylised_facts_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"  Saved: {fname}")


# =============================================================================
# SUMMARY PRINT
# =============================================================================

def print_summary(params_buy, params_sell, T_buy, T_sell, quiet=False):
    mu_b, al_b, be_b = params_buy
    mu_s, al_s, be_s = params_sell
    br_b, br_s = al_b / be_b, al_s / be_s

    res_b = compute_residuals(T_buy, *params_buy)
    res_s = compute_residuals(T_sell, *params_sell)
    ks_stat, ks_p = ks_2samp(res_b, res_s)

    if quiet:
        _log(
            f"  Summary: N_buy={len(T_buy)}, N_sell={len(T_sell)}, "
            f"BR_buy={br_b:.3f}, BR_sell={br_s:.3f}, KS_p={ks_p:.4f}",
            force=True,
        )
        return

    _log("\n" + "=" * 60, force=True)
    _log("  EXPERIMENT 2 - DIRECTIONAL ASYMMETRY SUMMARY", force=True)
    _log("=" * 60, force=True)
    _log(f"  {'Parameter':<28} {'Buy':>10}  {'Sell':>10}", force=True)
    _log(f"  {'-'*50}", force=True)
    _log(f"  {'N events':<28} {len(T_buy):>10d}  {len(T_sell):>10d}", force=True)
    _log(f"  {'mu (baseline, events/s)':<28} {mu_b:>10.5f}  {mu_s:>10.5f}", force=True)
    _log(f"  {'alpha (excitation jump)':<28} {al_b:>10.5f}  {al_s:>10.5f}", force=True)
    _log(f"  {'beta (decay rate)':<28} {be_b:>10.5f}  {be_s:>10.5f}", force=True)
    _log(f"  {'Branching ratio alpha/beta':<28} {br_b:>10.4f}  {br_s:>10.4f}", force=True)
    _log(f"  {'Mean inter-arrival (s)':<28} {np.mean(np.diff(T_buy)):>10.4f}  {np.mean(np.diff(T_sell)):>10.4f}", force=True)
    _log("\n  KS test on residuals (buy vs sell):", force=True)
    _log(f"    statistic = {ks_stat:.4f},  p-value = {ks_p:.4f}", force=True)
    if ks_p < 0.05:
        _log("    -> Residual distributions differ significantly (p < 0.05).", force=True)
    else:
        _log("    -> No significant difference in residuals (p >= 0.05).", force=True)
    _log("=" * 60 + "\n", force=True)


# =============================================================================
# PLOT 6: Cross-stock branching ratio comparison
# =============================================================================

def plot_cross_stock_branching(all_results, tickers):
    br_buy = [all_results[t]["params_buy"][1] / all_results[t]["params_buy"][2] for t in tickers]
    br_sell = [all_results[t]["params_sell"][1] / all_results[t]["params_sell"][2] for t in tickers]

    x = np.arange(len(tickers))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_b = ax.bar(x - w / 2, br_buy, width=w, color=BUY_COLOR, alpha=0.85, label="Buy alpha/beta", edgecolor="white")
    bars_s = ax.bar(x + w / 2, br_sell, width=w, color=SELL_COLOR, alpha=0.85, label="Sell alpha/beta", edgecolor="white")

    ax.axhline(1, color="red", ls="--", lw=1.2, label="Stationarity boundary")

    for bars in [bars_b, bars_s]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(tickers, fontsize=11)
    ax.set_ylabel("Branching ratio alpha/beta")
    ax.set_title("Cross-stock Hawkes Branching Ratio - Buy vs Sell", fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "exp2_cross_stock_branching.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"\n  Saved: {fname}", force=True)


# =============================================================================
# MAIN
# =============================================================================

def run_experiment_2(tickers=None, start=START_DATE, end=END_DATE, data_path=DATA_PATH):
    """
    Run Experiment 2 for every ticker in `tickers` (default: all stocks).
    Produces per-stock plots plus a cross-stock branching ratio comparison.
    """
    from main import STOCKS

    t0 = time.perf_counter()
    if tickers is None:
        tickers = STOCKS

    all_results = {}
    fitted_tickers = []

    if HAVE_RICH and console is not None:
        console.print(Panel(
            f"[bold]Tickers[/bold] : {', '.join(tickers)}\n"
            f"[bold]Period[/bold]  : {start} -> {end}",
            title="[bold cyan]Experiment 2 - Directional Asymmetry[/bold cyan]",
            border_style="cyan",
        ))

    def _run_one_ticker(ticker, progress=None, stage_task=None):
        stage_names = [
            "loaded", "split buy/sell", "fit buy", "fit sell",
            "summary", "base plots", "comparison plots", "done",
        ]

        def _advance(ix):
            if progress is not None and stage_task is not None:
                progress.advance(stage_task)
                progress.update(stage_task, status=stage_names[ix])

        t_stock = time.perf_counter()

        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily = loader.load()
        if not daily:
            _log(f"  [yellow]⚠[/yellow] No data found for {ticker}. Skipping.", force=True)
            return

        df = daily[0]
        t_buf_lo = df["Time"].min() + 3600
        t_buf_hi = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_buf_lo) & (df["Time"] <= t_buf_hi)].copy()
        _advance(0)

        mo_buy = df[(df["Type"] == 4) & (df["TradeDirection"] == 1)]["Time"].values
        mo_sell = df[(df["Type"] == 4) & (df["TradeDirection"] == -1)]["Time"].values
        mo_buy = np.sort(mo_buy[np.isfinite(mo_buy)])
        mo_sell = np.sort(mo_sell[np.isfinite(mo_sell)])
        _advance(1)

        if len(mo_buy) < 20 or len(mo_sell) < 20:
            _log(f"  [yellow]⚠[/yellow] Not enough events for {ticker} (buy={len(mo_buy)}, sell={len(mo_sell)}).", force=True)
            return

        params_buy = fit_hawkes(mo_buy, label=f"{ticker} buy-side", quiet=True)
        _advance(2)
        params_sell = fit_hawkes(mo_sell, label=f"{ticker} sell-side", quiet=True)
        _advance(3)

        if params_buy is None or params_sell is None:
            _log(f"  [yellow]⚠[/yellow] Hawkes fit failed for one or both sides ({ticker}).", force=True)
            return

        all_results[ticker] = {
            "params_buy": params_buy,
            "params_sell": params_sell,
            "mo_buy": mo_buy,
            "mo_sell": mo_sell,
        }
        fitted_tickers.append(ticker)

        print_summary(params_buy, params_sell, mo_buy, mo_sell, quiet=True)
        _advance(4)

        # main.py helper plots
        plot_hawkes_intensity(mo_buy, *params_buy, ticker=f"{ticker}_buy", quiet=True)
        plot_hawkes_intensity(mo_sell, *params_sell, ticker=f"{ticker}_sell", quiet=True)
        plot_residual_qqplot(mo_buy, *params_buy, ticker=f"{ticker}_buy", quiet=True)
        plot_residual_qqplot(mo_sell, *params_sell, ticker=f"{ticker}_sell", quiet=True)
        _advance(5)

        # experiment-specific plots
        plot_dual_intensity(mo_buy, params_buy, mo_sell, params_sell, ticker)
        plot_parameter_comparison(params_buy, params_sell, ticker)
        plot_joint_qqplot(mo_buy, params_buy, mo_sell, params_sell, ticker)
        plot_intraday_rate(mo_buy, mo_sell, ticker)
        plot_interarrival_hist(mo_buy, mo_sell, df, ticker)
        _advance(6)

        br_b = params_buy[1] / params_buy[2]
        br_s = params_sell[1] / params_sell[2]
        _log(f"  [bold]{ticker}[/bold]  BR_buy={br_b:.3f}  BR_sell={br_s:.3f}  ({time.perf_counter() - t_stock:.1f}s)", force=True)
        _advance(7)

    if HAVE_RICH and console is not None:
        # Keep both files quiet while progress bars are active.
        set_quiet_logs(True)
        if hasattr(main_module, "set_quiet_logs"):
            main_module.set_quiet_logs(True)
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
                    stage_task = progress.add_task(f"[cyan]{ticker}[/cyan]", total=8, status="starting")
                    _run_one_ticker(ticker, progress=progress, stage_task=stage_task)
                    progress.advance(outer)
        finally:
            set_quiet_logs(False)
            if hasattr(main_module, "set_quiet_logs"):
                main_module.set_quiet_logs(False)
    else:
        for ticker in tickers:
            _log("\n" + "=" * 60, force=True)
            _log(f"  Experiment 2 - Directional Asymmetry [{ticker}]", force=True)
            _log("=" * 60, force=True)
            _run_one_ticker(ticker)

    if len(fitted_tickers) > 1:
        _log("\n[bold cyan]Cross-stock branching ratio comparison[/bold cyan]" if HAVE_RICH else "\nCross-stock branching ratio comparison", force=True)
        plot_cross_stock_branching(all_results, fitted_tickers)

    elapsed = time.perf_counter() - t0
    if HAVE_RICH and console is not None:
        console.print(Panel(
            f"[bold green]Done[/bold green] in [bold]{elapsed:.1f}s[/bold] · "
            f"plots -> [dim]{os.path.abspath(PLOTS_DIR)}[/dim]",
            border_style="green",
        ))
    else:
        _log(f"\nExperiment 2 complete ({elapsed:.1f}s).  All plots in: {os.path.abspath(PLOTS_DIR)}", force=True)

    return all_results


if __name__ == "__main__":
    run_experiment_2()


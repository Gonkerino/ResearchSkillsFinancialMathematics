"""
Experiment 3 — Power-law vs Exponential Hawkes Kernel

Fits a Hawkes process with power-law kernel:
    φ(t) = c / (1 + t/τ)^η

and compares against the exponential kernel from main.py:
    φ(t) = α · exp(−β · t)

using AIC.  Produces:
  • Per-stock intensity plots (power-law kernel)
  • Per-stock QQ residual diagnostics (both kernels side-by-side)
  • Cross-stock AIC comparison bar chart (ΔAIC)
  • Cross-stock parameter comparison
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from main import (
    Loader,
    STOCKS,
    COLORS,
    DATA_PATH,
    START_DATE,
    END_DATE,
    fit_hawkes,
    hawkes_loglik,
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

PLOTS_DIR = os.path.join("plots", "experiment_3")
os.makedirs(PLOTS_DIR, exist_ok=True)


# =============================================================================
# POWER-LAW HAWKES — LIKELIHOOD
# =============================================================================

def powerlaw_kernel(dt, c, tau, eta):
    """φ(t) = c / (1 + t/τ)^η"""
    return c / (1.0 + dt / tau) ** eta


def powerlaw_kernel_integral(dt, c, tau, eta):
    """
    Φ(t) = ∫₀ᵗ φ(s) ds = c·τ/(η−1) · [1 − (1 + t/τ)^(1−η)]
    Valid for η > 1.
    """
    return c * tau / (eta - 1.0) * (1.0 - (1.0 + dt / tau) ** (1.0 - eta))


def powerlaw_branching_ratio(c, tau, eta):
    """
    n = ∫₀^∞ φ(s) ds = c·τ/(η−1)
    Requires η > 1 for convergence.
    """
    return c * tau / (eta - 1.0)


def hawkes_loglik_powerlaw(params, T):
    """
    Negative log-likelihood for Hawkes with power-law kernel.

    Parameters: [μ, c, τ, η]
    Kernel: φ(t) = c / (1 + t/τ)^η

    Complexity: O(n²) — no recursion trick available for power-law.
    Uses a truncation horizon for speed.
    """
    mu, c, tau, eta = params

    if mu <= 0 or c < 0 or tau <= 0 or eta <= 1.0:
        return np.inf

    # Stationarity check
    br = powerlaw_branching_ratio(c, tau, eta)
    if br >= 1.0:
        return np.inf

    n = len(T)
    ll = 0.0

    # Truncation: ignore kernel contributions older than this
    # Power-law with η > 1 decays, so we cut where φ(t) < 1e-10 * c
    # c / (1 + t_cut/τ)^η = 1e-10 * c  =>  t_cut = τ * (1e10^(1/η) - 1)
    t_cutoff = tau * (1e10 ** (1.0 / eta) - 1.0)
    t_cutoff = min(t_cutoff, T[-1] - T[0])  # cap at data range

    # --- Sum of log-intensities at event times ---
    for i in range(n):
        # Find events within truncation window
        if i == 0:
            lam_i = mu
        else:
            # Only look back within cutoff
            lookback_start = np.searchsorted(T[:i], T[i] - t_cutoff)
            dts = T[i] - T[lookback_start:i]
            kernel_vals = powerlaw_kernel(dts, c, tau, eta)
            lam_i = mu + np.sum(kernel_vals)

        if lam_i <= 0:
            return np.inf
        ll += np.log(lam_i)

    # --- Compensator: ∫₀ᵀ λ(t) dt = μ·(T_n - T_0) + Σᵢ Φ(T_n - tᵢ) ---
    durations = T[-1] - T
    compensator = mu * (T[-1] - T[0]) + np.sum(
        powerlaw_kernel_integral(durations, c, tau, eta)
    )

    return -(ll - compensator)


# =============================================================================
# POWER-LAW HAWKES — FITTING
# =============================================================================

def _make_inits_powerlaw(T, n_starts=10):
    """
    Generate initial parameter guesses for power-law Hawkes.
    Parameters: [μ, c, τ, η]
    """
    mean_rate = len(T) / (T[-1] - T[0])
    mean_ia = np.mean(np.diff(T))

    inits = []
    for mu_frac in [0.2, 0.4, 0.6]:
        for eta_init in [1.5, 2.0, 3.0]:
            tau_init = mean_ia * 5.0
            # branching ratio = c*tau/(eta-1), target ~0.5
            c_init = 0.5 * (eta_init - 1.0) / tau_init
            inits.append(np.array([
                mean_rate * mu_frac,
                c_init,
                tau_init,
                eta_init,
            ]))
    return inits


def fit_hawkes_powerlaw(T, label=""):
    """
    Fit Hawkes process with power-law kernel to event times T.

    Returns (mu, c, tau, eta) or None if fitting fails.
    """
    T = np.sort(np.asarray(T, dtype=float))
    T = T - T[0]
    T = T[np.isfinite(T)]

    if len(T) < 20:
        print(f"  Not enough events to fit power-law Hawkes ({label}).")
        return None, np.inf

    best_res, best_val = None, np.inf

    bounds = [
        (1e-6, None),     # mu
        (1e-8, None),     # c
        (1e-6, None),     # tau
        (1.01, 10.0),     # eta  (must be > 1 for integrability)
    ]

    for init in _make_inits_powerlaw(T):
        try:
            res = minimize(
                hawkes_loglik_powerlaw, init, args=(T,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500},
            )
            if res.fun < best_val and res.success:
                best_val = res.fun
                best_res = res
        except Exception:
            continue

    # Fallback: accept best result even if not converged
    if best_res is None:
        all_results = []
        for init in _make_inits_powerlaw(T):
            try:
                res = minimize(
                    hawkes_loglik_powerlaw, init, args=(T,),
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-10, "fatol": 1e-10},
                )
                all_results.append(res)
            except Exception:
                continue
        if all_results:
            best_res = min(all_results, key=lambda r: r.fun)
            best_val = best_res.fun
        else:
            print(f"  Fitting failed completely for {label}.")
            return None, np.inf

    mu, c, tau, eta = best_res.x
    br = powerlaw_branching_ratio(c, tau, eta)
    nll = best_val

    print(f"\n{'─' * 55}")
    print(f"  Power-law Hawkes fit — {label}")
    print(f"  μ (baseline)       = {mu:.5f}  events/sec")
    print(f"  c (kernel scale)   = {c:.5f}")
    print(f"  τ (time scale)     = {tau:.5f}  sec")
    print(f"  η (decay exponent) = {eta:.4f}")
    print(f"  Branching ratio    = {br:.4f}")
    if br >= 1:
        print("  Branching ratio >= 1 -> non-stationary; check data quality.")
    else:
        print(f"  -> ~{br * 100:.1f}% of events are triggered by previous events.")
    print(f"{'─' * 55}\n")

    return (mu, c, tau, eta), nll


# =============================================================================
# DIAGNOSTICS — INTENSITY PLOT
# =============================================================================

def plot_powerlaw_intensity(T, mu, c, tau, eta, ticker, n_grid=2000):
    """Plot the fitted power-law Hawkes intensity against event times."""
    T = np.sort(np.asarray(T, dtype=float))
    t_grid = np.linspace(T[0], T[-1], n_grid)

    t_cutoff = tau * (1e10 ** (1.0 / eta) - 1.0)
    t_cutoff = min(t_cutoff, T[-1] - T[0])

    intensities = np.zeros(n_grid)
    for j, t in enumerate(t_grid):
        idx_end = np.searchsorted(T, t)
        idx_start = np.searchsorted(T[:idx_end], t - t_cutoff)
        past = T[idx_start:idx_end]
        if len(past) > 0:
            dts = t - past
            intensities[j] = mu + np.sum(powerlaw_kernel(dts, c, tau, eta))
        else:
            intensities[j] = mu

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
    fig.suptitle(f"{ticker} — Power-law Hawkes Intensity", fontweight="bold")

    ax1.plot(t_grid, intensities, color=COLORS.get(ticker, "steelblue"),
             lw=0.8, alpha=0.9, label="λ(t) power-law")
    ax1.axhline(mu, color="red", ls="--", lw=1, alpha=0.7, label=f"μ = {mu:.4f}")
    ax1.set_ylabel("Intensity λ(t)")
    ax1.set_title("Conditional Intensity")
    ax1.legend(fontsize=9)

    ax2.eventplot([T], lineoffsets=0, linelengths=1,
                  color=COLORS.get(ticker, "steelblue"), alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Events")
    ax2.set_yticks([])

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"powerlaw_intensity_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============================================================================
# DIAGNOSTICS — QQ PLOT (side-by-side: exponential vs power-law)
# =============================================================================

def _compute_residuals_exponential(T, mu, alpha, beta):
    """Compute compensator increments for exponential Hawkes."""
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)
    A = 0.0
    Lambda = np.zeros(n)
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            A = A * np.exp(-beta * dt)
            Lambda[i] = (Lambda[i - 1]
                         + mu * dt
                         + (alpha / beta) * (1 - np.exp(-beta * dt)) * A)
        A += 1.0
    return np.diff(Lambda)


def _compute_residuals_powerlaw(T, mu, c, tau, eta):
    """Compute compensator increments for power-law Hawkes."""
    T = np.sort(np.asarray(T, dtype=float))
    n = len(T)

    t_cutoff = tau * (1e10 ** (1.0 / eta) - 1.0)
    t_cutoff = min(t_cutoff, T[-1] - T[0])

    Lambda = np.zeros(n)
    for i in range(1, n):
        dt = T[i] - T[i - 1]

        # Compensator increment = ∫_{t_{i-1}}^{t_i} λ(s) ds
        # = μ·dt + Σ_{k < i} [Φ(t_i - t_k) - Φ(t_{i-1} - t_k)]
        idx_start = max(0, np.searchsorted(T[:i], T[i] - t_cutoff))
        past = T[idx_start:i]

        kernel_integral_new = powerlaw_kernel_integral(T[i] - past, c, tau, eta)
        kernel_integral_old = powerlaw_kernel_integral(T[i - 1] - past, c, tau, eta)
        # For events that occurred before t_{i-1}, both integrals are valid.
        # For the event at exactly t_{i-1} (if idx = i-1), old integral is Φ(0) = 0.

        Lambda[i] = mu * dt + np.sum(kernel_integral_new - kernel_integral_old)

    return Lambda[1:]  # compensator increments (should be ~Exp(1))


def plot_qq_comparison(T_raw, exp_params, pl_params, ticker):
    """
    Side-by-side QQ plots for exponential and power-law Hawkes.
    """
    T = np.sort(np.asarray(T_raw, dtype=float))
    T_zeroed = T - T[0]

    mu_e, alpha_e, beta_e = exp_params
    mu_p, c_p, tau_p, eta_p = pl_params

    resid_exp = _compute_residuals_exponential(T_zeroed, mu_e, alpha_e, beta_e)
    resid_pl = _compute_residuals_powerlaw(T_zeroed, mu_p, c_p, tau_p, eta_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{ticker} — Residual QQ Plots (Exponential vs Power-law Kernel)",
                 fontweight="bold")

    for ax, resid, label, color in [
        (axes[0], resid_exp, "Exponential kernel", "#2ca02c"),
        (axes[1], resid_pl, "Power-law kernel", "#d62728"),
    ]:
        resid = resid[resid > 0]  # filter any numerical artefacts
        quantiles_emp = np.sort(resid)
        quantiles_th = -np.log(1 - np.linspace(0.01, 0.99, len(quantiles_emp)))

        ax.plot(quantiles_th, quantiles_emp, ".", alpha=0.4, color=color, ms=3)
        lim = max(quantiles_th.max(), quantiles_emp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.5, alpha=0.6, label="Perfect fit")
        ax.set_xlabel("Theoretical Exp(1) quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"qq_comparison_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============================================================================
# AIC COMPARISON
# =============================================================================

def compute_aic(neg_loglik, k):
    """AIC = 2k - 2ℓ = 2k + 2·(neg_loglik)"""
    return 2 * k + 2 * neg_loglik


def plot_aic_comparison(results):
    """
    Grouped bar chart of ΔAIC per ticker (exponential vs power-law).
    """
    tickers = list(results.keys())
    aic_exp = [results[t]["aic_exp"] for t in tickers]
    aic_pl = [results[t]["aic_pl"] for t in tickers]

    # Compute ΔAIC per ticker (relative to the winner)
    delta_exp = []
    delta_pl = []
    winners = []
    for i, t in enumerate(tickers):
        best = min(aic_exp[i], aic_pl[i])
        delta_exp.append(aic_exp[i] - best)
        delta_pl.append(aic_pl[i] - best)
        winners.append("Power-law" if aic_pl[i] < aic_exp[i] else "Exponential")

    x = np.arange(len(tickers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars1 = ax.bar(x - width / 2, delta_exp, width, label="Exponential kernel (k=3)",
                   color="#2ca02c", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width / 2, delta_pl, width, label="Power-law kernel (k=4)",
                   color="#d62728", alpha=0.85, edgecolor="white")

    # Annotate winner
    for i, t in enumerate(tickers):
        y_max = max(delta_exp[i], delta_pl[i])
        ax.text(x[i], y_max * 1.05 + 0.5, f"Winner: {winners[i]}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
                color="#333333")

    ax.set_xlabel("Ticker")
    ax.set_ylabel("ΔAIC (lower is better; winner = 0)")
    ax.set_title("Hawkes Kernel Comparison — ΔAIC per Ticker\n"
                 "(Exponential vs Power-law)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend(fontsize=10)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "aic_kernel_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")

    # Console summary table
    print(f"\n{'=' * 70}")
    print(f"  AIC Kernel Comparison Summary")
    print(f"{'=' * 70}")
    print(f"  {'Ticker':<8} {'AIC (Exp)':>14} {'AIC (PL)':>14} {'ΔAIC (Exp)':>12} {'ΔAIC (PL)':>12} {'Winner':>12}")
    print(f"  {'─' * 66}")
    for i, t in enumerate(tickers):
        print(f"  {t:<8} {aic_exp[i]:>14.2f} {aic_pl[i]:>14.2f} "
              f"{delta_exp[i]:>12.2f} {delta_pl[i]:>12.2f} {winners[i]:>12}")
    print(f"{'=' * 70}\n")


# =============================================================================
# CROSS-STOCK PARAMETER COMPARISON
# =============================================================================

def plot_parameter_comparison(results):
    """Compare branching ratios and decay characteristics across stocks."""
    tickers = [t for t in results if results[t]["pl_params"] is not None]
    if len(tickers) < 2:
        print("  Not enough stocks for parameter comparison.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Experiment 3 — Cross-stock Parameter Comparison",
                 fontweight="bold", fontsize=13)

    # Panel 1: Branching ratios (both kernels)
    ax = axes[0, 0]
    br_exp = [results[t]["exp_params"][1] / results[t]["exp_params"][2] for t in tickers]
    br_pl = [powerlaw_branching_ratio(*results[t]["pl_params"][1:]) for t in tickers]
    x = np.arange(len(tickers))
    w = 0.35
    ax.bar(x - w / 2, br_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, br_pl, w, label="Power-law", color="#d62728", alpha=0.85)
    ax.axhline(1, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Branching ratio")
    ax.set_title("Branching Ratio (α/β vs cτ/(η−1))")
    ax.legend(fontsize=9)

    # Panel 2: Baseline rates
    ax = axes[0, 1]
    mu_exp = [results[t]["exp_params"][0] for t in tickers]
    mu_pl = [results[t]["pl_params"][0] for t in tickers]
    ax.bar(x - w / 2, mu_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
    ax.bar(x + w / 2, mu_pl, w, label="Power-law", color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("μ (events/sec)")
    ax.set_title("Baseline Rate μ")
    ax.legend(fontsize=9)

    # Panel 3: Power-law η (decay exponent)
    ax = axes[1, 0]
    etas = [results[t]["pl_params"][3] for t in tickers]
    colors = [COLORS.get(t, "grey") for t in tickers]
    ax.bar(tickers, etas, color=colors, alpha=0.85)
    ax.axhline(2, color="grey", ls=":", lw=1, alpha=0.7, label="η = 2 (fast decay)")
    ax.set_ylabel("η (decay exponent)")
    ax.set_title("Power-law Decay Exponent η")
    ax.legend(fontsize=9)
    for i, (t, e) in enumerate(zip(tickers, etas)):
        ax.text(i, e + 0.02, f"{e:.2f}", ha="center", va="bottom", fontsize=9)

    # Panel 4: Power-law τ (time scale)
    ax = axes[1, 1]
    taus = [results[t]["pl_params"][2] for t in tickers]
    ax.bar(tickers, taus, color=colors, alpha=0.85)
    ax.set_ylabel("τ (seconds)")
    ax.set_title("Power-law Time Scale τ")
    for i, (t, v) in enumerate(zip(tickers, taus)):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "parameter_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_experiment3(tickers=None, start=START_DATE, end=END_DATE,
                    data_path=DATA_PATH):
    """
    Full Experiment 3 pipeline:
      1. Load data per stock
      2. Fit exponential Hawkes (reuse from main.py)
      3. Fit power-law Hawkes
      4. Compare AIC
      5. QQ diagnostics side-by-side
      6. Cross-stock comparison
    """
    if tickers is None:
        tickers = STOCKS

    results = {}

    for ticker in tickers:
        print(f"\n{'=' * 65}")
        print(f"  Experiment 3 — {ticker}")
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

        mo = df[df["Type"] == 4]
        T = mo["Time"].values
        T = np.sort(T[np.isfinite(T)])

        if len(T) < 20:
            print(f"  Not enough market orders for {ticker}.")
            continue

        # ── Fit exponential kernel (from main.py) ─────────────────────────
        print(f"\n  ── Exponential Kernel ──")
        T_zeroed = T - T[0]
        exp_params = fit_hawkes(T, label=f"{ticker} exponential")
        if exp_params is None:
            print(f"  Exponential fit failed for {ticker}.")
            continue
        mu_e, alpha_e, beta_e = exp_params
        nll_exp = hawkes_loglik([mu_e, alpha_e, beta_e], T_zeroed)

        # ── Fit power-law kernel ──────────────────────────────────────────
        print(f"\n  ── Power-law Kernel ──")
        pl_params, nll_pl = fit_hawkes_powerlaw(T, label=f"{ticker} power-law")
        if pl_params is None:
            print(f"  Power-law fit failed for {ticker}.")
            continue

        # ── AIC ───────────────────────────────────────────────────────────
        aic_exp = compute_aic(nll_exp, k=3)
        aic_pl = compute_aic(nll_pl, k=4)

        print(f"\n  AIC comparison for {ticker}:")
        print(f"    Exponential: AIC = {aic_exp:.2f}  (k=3, -ℓ={nll_exp:.2f})")
        print(f"    Power-law:   AIC = {aic_pl:.2f}  (k=4, -ℓ={nll_pl:.2f})")
        winner = "Power-law" if aic_pl < aic_exp else "Exponential"
        print(f"    Winner: {winner}  (ΔAIC = {abs(aic_exp - aic_pl):.2f})")

        results[ticker] = {
            "exp_params": exp_params,
            "pl_params": pl_params,
            "nll_exp": nll_exp,
            "nll_pl": nll_pl,
            "aic_exp": aic_exp,
            "aic_pl": aic_pl,
            "T": T,
        }

        # ── Per-stock plots ───────────────────────────────────────────────
        mu_p, c_p, tau_p, eta_p = pl_params
        plot_powerlaw_intensity(T, mu_p, c_p, tau_p, eta_p, ticker)
        plot_qq_comparison(T, exp_params, pl_params, ticker)

    # ── Cross-stock comparison ────────────────────────────────────────────
    if len(results) >= 2:
        plot_aic_comparison(results)
        plot_parameter_comparison(results)
    elif len(results) == 1:
        print("\n  Only one stock fitted — skipping cross-stock plots.")

    print(f"\nExperiment 3 complete. Plots saved to: {os.path.abspath(PLOTS_DIR)}")
    return results


# =============================================================================
# DIRECTIONAL VERSION OF EXPERIMENT 3
# =============================================================================

def plot_directional_aic_comparison(results):
    """
    Plot ΔAIC comparison for ALL / BUY / SELL market orders by ticker.
    Lower is better; the winning kernel has ΔAIC = 0.
    """
    tickers = [t for t in results if len(results[t]) > 0]
    if not tickers:
        print("  No directional results to plot.")
        return

    directions = ["ALL_MO", "BUY_MO", "SELL_MO"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    fig.suptitle("Directional Kernel Comparison — ΔAIC by Event Type", fontweight="bold")

    for ax, direction in zip(axes, directions):
        valid_tickers = [t for t in tickers if direction in results[t]]
        if not valid_tickers:
            ax.set_title(direction.replace("_", " "))
            ax.text(0.5, 0.5, "No fitted results", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue

        aic_exp = [results[t][direction]["aic_exp"] for t in valid_tickers]
        aic_pl = [results[t][direction]["aic_pl"] for t in valid_tickers]

        delta_exp, delta_pl = [], []
        for ae, ap in zip(aic_exp, aic_pl):
            best = min(ae, ap)
            delta_exp.append(ae - best)
            delta_pl.append(ap - best)

        x = np.arange(len(valid_tickers))
        w = 0.35
        ax.bar(x - w / 2, delta_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
        ax.bar(x + w / 2, delta_pl,  w, label="Power-law",  color="#d62728", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_tickers)
        ax.set_title(direction.replace("_", " "))
        ax.set_ylabel("ΔAIC (lower is better)")
        if direction == "ALL_MO":
            ax.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "directional_aic_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def plot_directional_branching_comparison(results):
    """
    Compare branching ratios for ALL / BUY / SELL under both kernels.
    """
    tickers = [t for t in results if len(results[t]) > 0]
    if not tickers:
        print("  No directional branching results to plot.")
        return

    directions = ["ALL_MO", "BUY_MO", "SELL_MO"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    fig.suptitle("Directional Branching Ratio Comparison", fontweight="bold")

    for ax, direction in zip(axes, directions):
        valid_tickers = [t for t in tickers if direction in results[t]]
        if not valid_tickers:
            ax.set_title(direction.replace("_", " "))
            ax.text(0.5, 0.5, "No fitted results", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])
            continue

        br_exp = [results[t][direction]["br_exp"] for t in valid_tickers]
        br_pl = [results[t][direction]["br_pl"] for t in valid_tickers]

        x = np.arange(len(valid_tickers))
        w = 0.35
        ax.bar(x - w / 2, br_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
        ax.bar(x + w / 2, br_pl,  w, label="Power-law",  color="#d62728", alpha=0.85)
        ax.axhline(1, color="black", ls="--", lw=1, alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_tickers)
        ax.set_title(direction.replace("_", " "))
        ax.set_ylabel("Branching ratio")
        if direction == "ALL_MO":
            ax.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "directional_branching_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def run_experiment3_directional(tickers=None, start=START_DATE, end=END_DATE, data_path=DATA_PATH):
    """
    Extended Experiment 3:
      Compare exponential vs power-law Hawkes separately for
      ALL market orders, BUY market orders, and SELL market orders.

    Returns
    -------
    dict
        results[ticker][label] with label in {"ALL_MO", "BUY_MO", "SELL_MO"}
    """
    if tickers is None:
        tickers = STOCKS

    results = {}

    for ticker in tickers:
        print(f"\n{'=' * 70}")
        print(f"  Directional Experiment 3 — {ticker}")
        print(f"{'=' * 70}")

        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily = loader.load()
        if not daily:
            print(f"  Skipping {ticker} (no data found).")
            continue

        df = daily[0]

        # same intraday filter as original experiment_3.py
        t_open_buffer = df["Time"].min() + 3600
        t_close_buffer = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_open_buffer) & (df["Time"] <= t_close_buffer)].copy()

        event_sets = {
            "ALL_MO":  df[df["Type"] == 4]["Time"].values,
            "BUY_MO":  df[(df["Type"] == 4) & (df["TradeDirection"] == 1)]["Time"].values,
            "SELL_MO": df[(df["Type"] == 4) & (df["TradeDirection"] == -1)]["Time"].values,
        }

        results[ticker] = {}

        for label, T in event_sets.items():
            T = np.sort(np.asarray(T, dtype=float))
            T = T[np.isfinite(T)]

            print(f"\n  ── {label} ──")
            print(f"  Number of events: {len(T)}")

            if len(T) < 20:
                print(f"  Not enough events for {ticker} {label}.")
                continue

            # ---------- Exponential kernel ----------
            print("  [Exponential kernel]")
            exp_params = fit_hawkes(T, label=f"{ticker} {label} exponential")
            if exp_params is None:
                print(f"  Exponential fit failed for {ticker} {label}.")
                continue

            T_zeroed = T - T[0]
            mu_e, alpha_e, beta_e = exp_params
            nll_exp = hawkes_loglik([mu_e, alpha_e, beta_e], T_zeroed)
            br_exp = alpha_e / beta_e

            # ---------- Power-law kernel ----------
            print("  [Power-law kernel]")
            pl_params, nll_pl = fit_hawkes_powerlaw(T, label=f"{ticker} {label} power-law")
            if pl_params is None:
                print(f"  Power-law fit failed for {ticker} {label}.")
                continue

            mu_p, c_p, tau_p, eta_p = pl_params
            br_pl = powerlaw_branching_ratio(c_p, tau_p, eta_p)

            # ---------- AIC ----------
            aic_exp = compute_aic(nll_exp, k=3)
            aic_pl = compute_aic(nll_pl, k=4)
            winner = "Power-law" if aic_pl < aic_exp else "Exponential"

            print(f"    AIC (Exp) = {aic_exp:.2f}")
            print(f"    AIC (PL)  = {aic_pl:.2f}")
            print(f"    Winner    = {winner}")
            print(f"    BR (Exp)  = {br_exp:.4f}")
            print(f"    BR (PL)   = {br_pl:.4f}")

            results[ticker][label] = {
                "T": T,
                "exp_params": exp_params,
                "pl_params": pl_params,
                "nll_exp": nll_exp,
                "nll_pl": nll_pl,
                "aic_exp": aic_exp,
                "aic_pl": aic_pl,
                "br_exp": br_exp,
                "br_pl": br_pl,
            }

            # ---------- QQ plot per direction ----------
            plot_qq_comparison(T, exp_params, pl_params, f"{ticker}_{label}")

            # ---------- Power-law intensity per direction ----------
            plot_powerlaw_intensity(T, mu_p, c_p, tau_p, eta_p, f"{ticker}_{label}")

    # ---------- Cross-stock directional plots ----------
    if len(results) >= 1:
        plot_directional_aic_comparison(results)
        plot_directional_branching_comparison(results)

    # ---------- Console summary ----------
    print(f"\n{'=' * 80}")
    print("  Directional Experiment 3 Summary")
    print(f"{'=' * 80}")
    for ticker in results:
        for label in ["ALL_MO", "BUY_MO", "SELL_MO"]:
            if label not in results[ticker]:
                continue
            r = results[ticker][label]
            winner = "Power-law" if r["aic_pl"] < r["aic_exp"] else "Exponential"
            print(
                f"  {ticker:<8} {label:<8} "
                f"AIC_exp={r['aic_exp']:>10.2f}  "
                f"AIC_pl={r['aic_pl']:>10.2f}  "
                f"BR_exp={r['br_exp']:.4f}  "
                f"BR_pl={r['br_pl']:.4f}  "
                f"Winner={winner}"
            )
    print(f"{'=' * 80}")
    print(f"Directional Experiment 3 complete. Plots saved to: {os.path.abspath(PLOTS_DIR)}")

    return results

if __name__ == "__main__":
    results = run_experiment3_directional()
    
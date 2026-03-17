"""
Power-law Hawkes fit (exact MLE) using the existing `main.py` pipeline.

What this file does
-------------------
- Reuses the LOBSTER loader and exponential Hawkes fit from `main.py`
- Implements Experiment 3 with a power-law kernel
- Fits the exact univariate power-law Hawkes model in branching-ratio form
- Compares power-law vs exponential using AIC on market-order times
- Produces power-law intensity and residual diagnostics
- Supports single-ticker or batch multi-ticker analysis

Model
-----
Intensity:
    lambda(t) = mu + sum_{t_j < t} h(t - t_j)

Power-law kernel in branching-ratio form:
    h(dt) = n * (eta - 1) / tau * (1 + dt / tau)^(-eta)

where
    mu  > 0      baseline intensity
    0 <= n < 1   branching ratio / kernel mass
    tau > 0      time scale
    eta > 1      tail exponent

Integrated kernel:
    H(x) = int_0^x h(s) ds = n * [1 - (1 + x/tau)^(1-eta)]

Performance
-----------
All O(N²) hotpaths (likelihood + gradient, compensator, intensity path)
are JIT-compiled via Numba with parallel execution where beneficial.
The multi-start seed ranking and full optimisation loops benefit from
50–100× speedups over the pure-Python original.

Notes
-----
1) This is the exact power-law likelihood, not an exponential-mixture surrogate.
2) To stay consistent with `main.py`, the event definition here is market
   orders (Type == 4) after the same 1-hour opening/closing buffer.
3) The code handles duplicated timestamps by a tiny deterministic de-tying jitter
   so the point process remains strictly ordered.

Usage
-----
Place this file next to `main.py`, then run (all tickers by default):

    python power_hawkes_goog.py

Or run a single ticker:

    python power_hawkes_goog.py --ticker GOOG

Or specify a date range / path explicitly:

    python power_hawkes_goog.py --start 2012-06-21 --end 2012-06-21 --data-path data/

All plots are saved to plots/power_hawkes/.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

# Force non-GUI backend before pyplot import to avoid Tk thread teardown issues.
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import kstest

import numba as nb

# ---------------------------------------------------------------------------
# Rich — standard install preferred; graceful fallback to plain print
# ---------------------------------------------------------------------------
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
    from rich.table import Table
    from rich import box
    HAVE_RICH = True
except ModuleNotFoundError:
    HAVE_RICH = False

import main as base


# =============================================================================
# Numba configuration
# =============================================================================
# Set the threading layer before any JIT compilation occurs.
# TBB is preferred for nested parallelism; fall back to OpenMP or workqueue.
_NB_PARALLEL  = True
_NB_CACHE     = True
_NB_FASTMATH  = True

# Number of threads — Numba will use this for prange parallelism.
# Defaults to all logical cores; override via NUMBA_NUM_THREADS env var.
_NUM_THREADS = nb.config.NUMBA_NUM_THREADS


# =============================================================================
# Local config
# =============================================================================
DEFAULT_TICKER = "GOOG"
ALL_TICKERS    = ["GOOG", "AMZN", "AAPL", "INTC", "MSFT"]

# All figures saved to a single flat folder
PLOTS_DIR = os.path.join("plots", "power_hawkes")
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.figsize"   : (12, 5),
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "font.size"        : 11,
})

# ---------------------------------------------------------------------------
# Console and quiet-log infrastructure — mirrors main.py / experiment_2.py
# ---------------------------------------------------------------------------
console      = Console() if HAVE_RICH else None
_QUIET_LOGS  = False


def set_quiet_logs(flag: bool) -> None:
    """Enable/disable non-critical logging (used during live progress rendering)."""
    global _QUIET_LOGS
    _QUIET_LOGS = bool(flag)


def _log(msg: str, force: bool = False) -> None:
    """Unified logger: rich if available, plain print otherwise.

    ``force=True`` overrides the quiet flag — use for summary lines that
    must always appear even when a progress bar is active.
    """
    if _QUIET_LOGS and not force:
        return
    if HAVE_RICH and console is not None:
        console.print(msg)
    else:
        print(msg)


# ---------------------------------------------------------------------------
# No-op progress stub — used when Rich is unavailable or quiet=True.
# Satisfies the full Progress + console.print API used throughout.
# ---------------------------------------------------------------------------
class _NoProgress:
    """Drop-in for rich.progress.Progress when Rich is absent or bars are off.

    All task methods are no-ops.  ``console.print`` strips Rich markup tags
    and writes plain text to stdout, so key messages still appear.
    """
    def add_task(self, *a, **kw):    return None
    def update(self, *a, **kw):      pass
    def advance(self, *a, **kw):     pass
    def __enter__(self):             return self
    def __exit__(self, *a):          pass

    class _FallbackConsole:
        @staticmethod
        def print(msg: str) -> None:
            import re
            print(re.sub(r"\[/?[^\]]*\]", "", str(msg)))

    console = _FallbackConsole()


# =============================================================================
# Directory helper
# =============================================================================
def _plots_dir(ticker: str = "") -> str:
    """Return the shared plots directory (ticker argument kept for call-site compat)."""
    return PLOTS_DIR


# =============================================================================
# Data classes
# =============================================================================
@dataclass
class PowerFitResult:
    mu:       float
    n:        float
    tau:      float
    eta:      float
    nll:      float
    success:  bool
    nit:      int
    message:  str
    ks_stat:  Optional[float] = None
    ks_pvalue: Optional[float] = None

    @property
    def c(self) -> float:
        """Kernel prefactor c = n(eta-1)/tau."""
        return self.n * (self.eta - 1.0) / self.tau

    @property
    def aic(self) -> float:
        return 2 * 4 + 2 * self.nll


@dataclass
class ExpFitSummary:
    mu:    float
    alpha: float
    beta:  float
    nll:   float

    @property
    def br(self) -> float:
        return self.alpha / self.beta

    @property
    def aic(self) -> float:
        return 2 * 3 + 2 * self.nll


# =============================================================================
# Data loading helpers
# =============================================================================
def detie_timestamps(
    T: np.ndarray,
    eps: Optional[float] = None,
) -> Tuple[np.ndarray, int, float]:
    """
    Make event times strictly increasing with a tiny deterministic jitter.

    Returns
    -------
    T_out, n_adjusted, eps_used
    """
    T = np.asarray(T, dtype=float).copy()
    if T.size <= 1:
        return T, 0, 0.0

    T.sort()
    diffs = np.diff(T)
    pos   = diffs[diffs > 0]

    if eps is None:
        eps = max(1e-9, 0.1 * float(pos.min())) if pos.size else 1e-6

    adjusted  = 0
    run_start = 0
    n         = T.size
    while run_start < n:
        run_end = run_start + 1
        while run_end < n and T[run_end] == T[run_start]:
            run_end += 1
        run_len = run_end - run_start
        if run_len > 1:
            adjusted += run_len - 1
            T[run_start:run_end] += eps * np.arange(run_len, dtype=float)
        run_start = run_end

    return T, adjusted, float(eps)


def load_market_orders(
    ticker:    str = DEFAULT_TICKER,
    start:     str = base.START_DATE,
    end:       str = base.END_DATE,
    data_path: str = base.DATA_PATH,
    quiet:     bool = False,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Load data using the same logic as `main.py` and return market-order times."""
    loader = base.Loader(ticker, start, end, dataPath=data_path, nlevels=10)
    daily  = loader.load()
    if not daily:
        raise FileNotFoundError(
            f"No {ticker} data found in {os.path.abspath(data_path)!r} for {start} -> {end}."
        )

    df = daily[0].copy()
    t_open_buffer  = float(df["Time"].min()) + 3600.0
    t_close_buffer = float(df["Time"].max()) - 3600.0
    df = df[(df["Time"] >= t_open_buffer) & (df["Time"] <= t_close_buffer)].copy()

    mo = df[df["Type"] == 4].copy()
    T  = np.sort(mo["Time"].values.astype(float))
    T  = T[np.isfinite(T)]

    T_detied, n_adjusted, eps_used = detie_timestamps(T)
    if n_adjusted and not quiet:
        _log(
            f"[cyan]ℹ[/cyan] {ticker}: de-tied {n_adjusted} duplicated timestamps "
            f"at ~{eps_used:.6g}s resolution.",
            force=True,
        )

    meta = {
        "n_events" : int(T_detied.size),
        "n_adjusted": int(n_adjusted),
        "eps_used"  : float(eps_used),
        "date"      : str(df["Date"].iloc[0]),
        "data_path" : os.path.abspath(data_path),
    }
    return T_detied, meta


# =============================================================================
# Power-law Hawkes model — Numba-accelerated kernels
# =============================================================================

@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
         cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _kernel_scalar(dt: float, n: float, tau: float, eta: float) -> float:
    """Scalar kernel evaluation: h(dt) = n(eta-1)/tau * (1 + dt/tau)^(-eta)."""
    coeff = n * (eta - 1.0) / tau
    return coeff * (1.0 + dt / tau) ** (-eta)


@nb.njit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64),
         cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _kernel_int_scalar(x: float, n: float, tau: float, eta: float) -> float:
    """Scalar integrated kernel: H(x) = n * [1 - (1 + x/tau)^(1-eta)]."""
    q = 1.0 + x / tau
    return n * (1.0 - q ** (1.0 - eta))


@nb.njit(cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _power_hawkes_nll_grad_core(
    T:          np.ndarray,
    mu:         float,
    n:          float,
    tau:        float,
    eta:        float,
    tau_lo:     float,
    n_cap:      float,
    pen_weight: float,
) -> Tuple:
    """
    Core NLL + gradient computation for the power-law Hawkes model,
    with optional log-barrier penalties to prevent degenerate solutions.

    This is the innermost O(N²) hotpath. Compiled to native code via Numba
    with fastmath enabled for aggressive FP optimisation.

    Penalty terms (active only when pen_weight > 0)
    ------------------------------------------------
    Two one-sided log-barriers that only repel parameters from boundaries,
    without attracting them toward any reference point:

        P_tau = -pen_weight * log((tau - tau_lo) / tau_lo)
            → +inf as tau → tau_lo from above, negligible for tau >> tau_lo

        P_n = -pen_weight * log(1 - n / n_cap)
            → +inf as n → n_cap from below, negligible for n << n_cap

    The pen_weight is used directly (NOT multiplied by N), so the caller
    controls the absolute strength.

    Returns
    -------
    (nll, g_mu, g_n, g_tau, g_eta)
    Returns (inf, 0, 0, 0, 0) on invalid input.
    """
    N   = T.shape[0]
    INF = np.inf

    if mu <= 0.0 or n < 0.0 or n >= 1.0 or tau <= 0.0 or eta <= 1.0:
        return INF, 0.0, 0.0, 0.0, 0.0
    if N < 2:
        return INF, 0.0, 0.0, 0.0, 0.0

    T_end   = T[N - 1]
    em1     = eta - 1.0
    coeff   = n * em1 / tau
    inv_tau = 1.0 / tau
    inv_em1 = 1.0 / em1

    # ── Part 1: sum of log-intensities ──────────────────────────────────
    ll       = 0.0
    d_mu_ll  = 0.0
    d_n_ll   = 0.0
    d_tau_ll = 0.0
    d_eta_ll = 0.0

    for i in range(N):
        ti       = T[i]
        lam      = mu
        dlam_n   = 0.0
        dlam_tau = 0.0
        dlam_eta = 0.0

        for j in range(i):
            dt      = ti - T[j]
            q       = 1.0 + dt * inv_tau
            log_q   = math.log(q)
            q_neg_eta = math.exp(-eta * log_q)   # q^(-eta)
            h       = coeff * q_neg_eta

            lam     += h
            # Guard: if n is essentially zero, the kernel contribution is
            # zero and dh/dn = (eta-1)/tau * q^(-eta) (the limit of h/n).
            if n > 1e-15:
                dlam_n += h / n
            else:
                dlam_n += em1 * inv_tau * q_neg_eta
            dlam_tau += h * (-inv_tau + eta * dt * inv_tau * inv_tau / q)
            dlam_eta += h * (inv_em1 - log_q)

        if lam <= 0.0 or not math.isfinite(lam):
            return INF, 0.0, 0.0, 0.0, 0.0

        log_lam  = math.log(lam)
        ll      += log_lam
        inv_lam  = 1.0 / lam
        d_mu_ll  += inv_lam
        d_n_ll   += dlam_n   * inv_lam
        d_tau_ll += dlam_tau * inv_lam
        d_eta_ll += dlam_eta * inv_lam

    # ── Part 2: compensator ─────────────────────────────────────────────
    compensator  = mu * T_end
    d_mu_comp    = T_end
    d_n_comp     = 0.0
    d_tau_comp   = 0.0
    d_eta_comp   = 0.0

    one_m_eta = 1.0 - eta
    for k in range(N):
        x       = T_end - T[k]
        qx      = 1.0 + x * inv_tau
        log_qx  = math.log(qx)
        qx_1me  = math.exp(one_m_eta * log_qx)   # qx^(1-eta)

        H_k      = n * (1.0 - qx_1me)
        compensator  += H_k

        d_n_comp   += 1.0 - qx_1me
        d_tau_comp += n * one_m_eta * x * inv_tau * inv_tau * math.exp(-eta * log_qx)
        d_eta_comp += n * log_qx * qx_1me

    nll   = -(ll - compensator)
    g_mu  = -(d_mu_ll)  + d_mu_comp
    g_n   = -(d_n_ll)   + d_n_comp
    g_tau = -(d_tau_ll) + d_tau_comp
    g_eta = -(d_eta_ll) + d_eta_comp

    # ── Part 3: one-sided log-barrier penalties ─────────────────────────
    # These only repel from boundaries — they do NOT attract toward any
    # interior reference point.  pen_weight is used as-is (not scaled by N).
    if pen_weight > 0.0:
        # tau barrier: -w * log((tau - tau_lo) / tau_lo)
        slack_tau = tau - tau_lo
        if slack_tau > 0.0 and tau_lo > 0.0:
            nll   += -pen_weight * math.log(slack_tau / tau_lo)
            g_tau += -pen_weight / slack_tau

        # n barrier: -w * log(1 - n / n_cap)
        ratio_n = n / n_cap
        if ratio_n < 1.0 and n_cap > 0.0:
            nll += -pen_weight * math.log(1.0 - ratio_n)
            g_n += pen_weight / (n_cap - n)

    if not (math.isfinite(g_mu) and math.isfinite(g_n) and
            math.isfinite(g_tau) and math.isfinite(g_eta) and
            math.isfinite(nll)):
        return INF, 0.0, 0.0, 0.0, 0.0

    return nll, g_mu, g_n, g_tau, g_eta


def power_hawkes_nll_grad(
    params:     np.ndarray,
    T:          np.ndarray,
    tau_lo:     float = 1e-4,
    n_cap:      float = 0.95,
    pen_weight: float = 0.0,
) -> Tuple[float, np.ndarray]:
    """
    Negative log-likelihood and analytic gradient for the exact power-law Hawkes model.

    Thin wrapper around the Numba-compiled core that satisfies the
    scipy.optimize interface (params as single array, returns (f, grad)).

    Parameters are in branching-ratio form:
        params = [mu, n, tau, eta]

    Passing the gradient alongside the objective halves L-BFGS-B evaluations.
    """
    mu, n, tau, eta = float(params[0]), float(params[1]), float(params[2]), float(params[3])
    T = np.ascontiguousarray(T, dtype=np.float64)

    nll, g_mu, g_n, g_tau, g_eta = _power_hawkes_nll_grad_core(
        T, mu, n, tau, eta, tau_lo, n_cap, pen_weight,
    )
    grad = np.array([g_mu, g_n, g_tau, g_eta], dtype=np.float64)
    return float(nll), grad


# ── Vectorised kernel evaluations (NumPy-level, used for plotting) ───────────
def power_kernel(dt: np.ndarray, n: float, tau: float, eta: float) -> np.ndarray:
    """h(dt) = n(eta-1)/tau * (1 + dt/tau)^(-eta).  Vectorised over dt."""
    coeff = n * (eta - 1.0) / tau
    return coeff * np.power(1.0 + np.asarray(dt) / tau, -eta)


def power_kernel_int(x: np.ndarray, n: float, tau: float, eta: float) -> np.ndarray:
    """H(x) = int_0^x h(s) ds.  Vectorised over x."""
    q = 1.0 + np.asarray(x) / tau
    return n * (1.0 - np.power(q, 1.0 - eta))


# =============================================================================
# Parallel seed evaluation
# =============================================================================
@nb.njit(cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _eval_seed(
    T:          np.ndarray,
    mu:         float,
    n:          float,
    tau:        float,
    eta:        float,
    tau_lo:     float,
    n_cap:      float,
    pen_weight: float,
) -> float:
    """Evaluate penalised NLL at a single seed point (returns scalar NLL only)."""
    nll, _, _, _, _ = _power_hawkes_nll_grad_core(
        T, mu, n, tau, eta, tau_lo, n_cap, pen_weight,
    )
    return nll


@nb.njit(parallel=_NB_PARALLEL, cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _rank_seeds_parallel(
    T:          np.ndarray,
    seeds:      np.ndarray,
    tau_lo:     float,
    n_cap:      float,
    pen_weight: float,
) -> np.ndarray:
    """
    Evaluate penalised NLL for all seed points in parallel using Numba prange.

    Parameters
    ----------
    T          : 1-D array of event times (coarse-subsampled)
    seeds      : 2-D array of shape (n_seeds, 4), each row = [mu, n, tau, eta]
    tau_lo     : lower bound on tau for the barrier
    n_cap      : upper cap for n barrier
    pen_weight : barrier strength

    Returns
    -------
    nll_values : 1-D array of NLL values, one per seed.
    """
    n_seeds    = seeds.shape[0]
    nll_values = np.empty(n_seeds, dtype=np.float64)

    for s in nb.prange(n_seeds):
        nll_values[s] = _eval_seed(
            T, seeds[s, 0], seeds[s, 1], seeds[s, 2], seeds[s, 3],
            tau_lo, n_cap, pen_weight,
        )

    return nll_values


# =============================================================================
# Compensator and residuals — Numba-accelerated
# =============================================================================
@nb.njit(cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _power_compensator_core(
    T:   np.ndarray,
    mu:  float,
    n:   float,
    tau: float,
    eta: float,
) -> np.ndarray:
    """
    Cumulative compensator at each event time.

    Lambda(t_i) = mu * t_i + sum_{j<i} H(t_i - t_j)

    where H(x) = n * [1 - (1 + x/tau)^(1-eta)].

    T is assumed zero-indexed (T[0] = 0).  This is O(N²) but JIT-compiled.
    """
    N         = T.shape[0]
    Lambda    = np.empty(N, dtype=np.float64)
    inv_tau   = 1.0 / tau
    one_m_eta = 1.0 - eta

    for i in range(N):
        ti  = T[i]
        acc = mu * ti
        for j in range(i):
            x   = ti - T[j]
            qx  = 1.0 + x * inv_tau
            acc += n * (1.0 - qx ** one_m_eta)
        Lambda[i] = acc

    return Lambda


def power_compensator(T: np.ndarray, fit: PowerFitResult) -> np.ndarray:
    """Compensator values at event times for time-change diagnostics."""
    T = np.ascontiguousarray(T, dtype=np.float64)
    return _power_compensator_core(T, fit.mu, fit.n, fit.tau, fit.eta)


def power_residuals(T: np.ndarray, fit: PowerFitResult) -> np.ndarray:
    """
    Residual inter-arrivals under the Papangelou time-change theorem.

    If the model is correctly specified, these should be i.i.d. Exp(1).
    """
    Lambda = power_compensator(T, fit)
    resid  = np.diff(Lambda)
    return resid[np.isfinite(resid)]


# =============================================================================
# Intensity path — Numba-accelerated with parallel grid evaluation
# =============================================================================
@nb.njit(parallel=_NB_PARALLEL, cache=_NB_CACHE, fastmath=_NB_FASTMATH)
def _power_intensity_path_core(
    T:      np.ndarray,
    t_grid: np.ndarray,
    mu:     float,
    n:      float,
    tau:    float,
    eta:    float,
) -> np.ndarray:
    """
    Exact intensity path on a time grid, parallelised over grid points via prange.

    For each grid point t: lambda(t) = mu + sum_{T_j < t} h(t - T_j).

    Grid points are independent, so this trivially parallelises across cores.
    A binary search locates the right-most past event for each grid point,
    giving O(n_grid × log(N_events) + n_grid × N_events_before_t) complexity.
    """
    N_grid    = t_grid.shape[0]
    N_events  = T.shape[0]
    lam       = np.empty(N_grid, dtype=np.float64)
    coeff     = n * (eta - 1.0) / tau
    inv_tau   = 1.0 / tau

    for i in nb.prange(N_grid):
        t   = t_grid[i]
        val = mu

        # Binary search for the count of events strictly before t.
        lo = 0
        hi = N_events
        while lo < hi:
            mid = (lo + hi) >> 1
            if T[mid] < t:
                lo = mid + 1
            else:
                hi = mid

        for j in range(lo):
            dt  = t - T[j]
            q   = 1.0 + dt * inv_tau
            val += coeff * q ** (-eta)

        lam[i] = val

    return lam


def power_intensity_path(
    T:      np.ndarray,
    fit:    PowerFitResult,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Exact intensity path on a grid (Numba-parallel over grid points)."""
    T      = np.ascontiguousarray(np.sort(T), dtype=np.float64)
    t_grid = np.ascontiguousarray(t_grid, dtype=np.float64)
    return _power_intensity_path_core(T, t_grid, fit.mu, fit.n, fit.tau, fit.eta)


# =============================================================================
# Multi-start fitting
# =============================================================================
def _make_power_inits(
    T:          np.ndarray,
    tau_lower:  float,
    n_upper:    float,
) -> np.ndarray:
    """
    Multi-start seeds built around empirical time scales.

    Empirical moment matching gives a good starting region — the same
    philosophy used in main.py's _make_inits() for exponential Hawkes.
    Seeds are returned as a (n_seeds, 4) array for direct use with
    _rank_seeds_parallel.
    """
    duration  = float(T[-1])
    mean_rate = len(T) / duration if duration > 0 else 1.0
    diffs     = np.diff(T)
    mean_ia   = float(np.mean(diffs)) if diffs.size else 1.0

    mu_grid  = mean_rate * np.array([0.20, 0.50, 0.85])
    n_grid   = np.array([0.15, 0.40, min(0.70, n_upper * 0.85)])

    # tau seeds: log-spaced from data-driven lower bound up to ~10× mean IA
    tau_lo   = max(tau_lower, 1e-4)
    tau_hi   = max(10.0 * mean_ia, 2.0 * tau_lo)
    tau_grid = np.exp(np.linspace(np.log(tau_lo), np.log(tau_hi), 3))

    eta_grid = np.array([1.20, 1.80, 3.00])

    seeds = []
    for mu0 in mu_grid:
        for n0 in n_grid:
            for tau0 in tau_grid:
                for eta0 in eta_grid:
                    seeds.append([mu0, n0, tau0, eta0])

    return np.array(seeds, dtype=np.float64)


def _coarse_subsample(T: np.ndarray, target_n: int = 400, seed: int = 42) -> np.ndarray:
    """
    Reproducible random subsampling for cheap seed ranking.

    Random subsampling (rather than evenly-spaced thinning) preserves the
    inter-arrival distribution, giving more accurate seed ranking.
    Uses a deterministic per-ticker seed to match stylised_facts.py's pattern.
    """
    if T.size <= target_n:
        return T
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(T.size, target_n, replace=False))
    return T[idx]


def _warmup_jit(T_sample: np.ndarray) -> None:
    """
    Trigger Numba compilation of all JIT-compiled functions before the
    timed optimisation loop.  Pays the compile cost upfront rather than
    contaminating the first seed evaluation — mirrors the warm-up pattern
    in kernel_sum_exp.py.
    """
    tiny      = T_sample[:min(10, T_sample.size)].copy()
    tiny_grid = np.linspace(tiny[0], tiny[-1], 5)
    dummy_seeds = np.array([[0.5, 0.5, 1.0, 2.0]], dtype=np.float64)

    _power_hawkes_nll_grad_core(tiny, 0.5, 0.5, 1.0, 2.0, 1.0, 0.95, 1.0)
    _rank_seeds_parallel(tiny, dummy_seeds, 1.0, 0.95, 1.0)
    _power_compensator_core(tiny, 0.5, 0.5, 1.0, 2.0)
    _power_intensity_path_core(tiny, tiny_grid, 0.5, 0.5, 1.0, 2.0)


def fit_power_hawkes(
    T:         np.ndarray,
    label:     str  = "",
    quiet:     bool = False,
    _plog            = None,
    _progress        = None,
) -> Optional[PowerFitResult]:
    """
    Fit the exact power-law Hawkes model by direct MLE with Numba acceleration.

    Multi-stage optimisation strategy
    ----------------------------------
    Stage 1: Pure MLE (no penalty) with data-driven bounds.  Run from
             multiple empirically-initialised starts.
    Stage 2: If Stage 1 hits a degenerate boundary (n ≈ n_cap, tau ≈
             tau_lower, or n≈0 Poisson collapse), re-run with a gentle
             log-barrier penalty to nudge the solution inward, then refine
             with pure MLE from that starting point.

    The penalty is only activated as a rescue mechanism, not by default.
    Uses quiet= flag to suppress output when called from a progress bar,
    consistent with main.py's fit_hawkes() and experiment_2.py pattern.
    """
    T = np.sort(np.ascontiguousarray(T, dtype=np.float64))
    T = T[np.isfinite(T)]
    if T.size < 20:
        _log(f"  [yellow]⚠[/yellow] Not enough events to fit power-law Hawkes ({label}).", force=not quiet)
        return None

    # _emit: routes to progress.console.print when inside a progress bar,
    # otherwise falls through to the normal _log machinery.
    def _emit(msg):
        if _plog is not None:
            _plog(msg)
        elif not quiet:
            _log(msg, force=True)

    T = T - T[0]   # zero-index for numerical stability (same as main.py)

    # ── Data-driven bounds ──────────────────────────────────────────────
    diffs      = np.diff(T)
    pos_diffs  = diffs[diffs > 0]
    mean_ia    = float(np.mean(diffs)) if diffs.size else 1.0

    # tau lower bound: 5th percentile of positive inter-arrivals.
    # Prevents the kernel collapsing to a delta — excitation faster than
    # the data's own time resolution is unphysical.
    if pos_diffs.size >= 20:
        tau_lower = float(np.percentile(pos_diffs, 5))
    else:
        tau_lower = float(pos_diffs.min()) if pos_diffs.size else 0.01
    tau_lower = max(tau_lower, 1e-6)

    tau_upper = max(50.0 * mean_ia, 1.0)
    n_upper   = 0.95   # branching ratio above this is near-critical and unphysical

    bounds = [
        (1e-8,      None),           # mu
        (1e-6,      n_upper),        # n
        (tau_lower, tau_upper),      # tau
        (1.01,      50.0),           # eta  
    ]

    _emit(
        f"  [{label}] N={T.size:,}  τ_lo={tau_lower:.4g}  τ_hi={tau_upper:.4g}  "
        f"n_cap={n_upper}  warming up Numba ({_NUM_THREADS} threads)..."
    )
    _warmup_jit(T)

    # ── Parallel seed ranking on coarse subsample ────────────────────────
    ticker_seed = abs(hash(label)) % (2 ** 31)
    coarse_T    = np.ascontiguousarray(
        _coarse_subsample(T, target_n=400, seed=ticker_seed), dtype=np.float64
    )
    seeds = _make_power_inits(T, tau_lower=tau_lower, n_upper=n_upper)

    # Show an indeterminate spinner while the parallel Numba ranking runs —
    # there are no iterations to count here, just one parallel kernel call.
    _rank_task = None
    if _progress is not None and HAVE_RICH:
        _rank_task = _progress.add_task(
            f"    [dim]{label[:18]}[/dim] ranking {len(seeds)} seeds",
            total=None,   # indeterminate — shows spinner not bar
            status=f"N_coarse={coarse_T.size}",
            visible=True,
        )

    nll_values  = _rank_seeds_parallel(coarse_T, seeds, tau_lower, n_upper, 0.0)
    finite_mask = np.isfinite(nll_values)

    if _rank_task is not None:
        _progress.update(_rank_task, visible=False)

    n_candidates = 10
    if np.any(finite_mask):
        order = np.argsort(nll_values)
        candidate_starts = [
            seeds[idx].copy()
            for idx in order
            if finite_mask[idx]
        ][:n_candidates]
    else:
        candidate_starts = [seeds[i] for i in range(min(n_candidates, len(seeds)))]

    n_finite = int(finite_mask.sum())
    _emit(
        f"  [{label}] seed ranking: {n_finite}/{len(seeds)} finite  ·  "
        f"coarse N={coarse_T.size}  ·  top {len(candidate_starts)} starts selected"
    )

    # ── Stage 1: Pure MLE from multiple starts ──────────────────────────
    best_x   = None
    best_val = np.inf
    best_info: dict = {"warnflag": 1, "nit": -1, "task": "no optimization run"}
    all_runs = []

    _opt_task = None
    if _progress is not None and HAVE_RICH:
        _opt_task = _progress.add_task(
            f"    [dim]{label[:18]}[/dim] stage 1  0/{len(candidate_starts)}",
            total=len(candidate_starts),
            status="starting...",
            visible=True,
        )

    for i, init in enumerate(candidate_starts):
        if _progress is not None and _opt_task is not None:
            _progress.update(
                _opt_task,
                description=(
                    f"    [dim]{label[:18]}[/dim] "
                    f"stage 1  {i+1}/{len(candidate_starts)}"
                ),
                status=(
                    f"best nll={best_val:.1f}"
                    if np.isfinite(best_val) else "running..."
                ),
            )
        x_opt, f_opt, info = fmin_l_bfgs_b(
            func=power_hawkes_nll_grad,
            x0=init,
            args=(T, tau_lower, n_upper, 0.0),
            bounds=bounds,
            factr=1e7,
            pgtol=1e-8,
            maxiter=1000,
        )
        all_runs.append((x_opt, float(f_opt), info))
        if np.isfinite(f_opt) and f_opt < best_val:
            best_x, best_val, best_info = x_opt, float(f_opt), info
        if _progress is not None and _opt_task is not None:
            _progress.advance(_opt_task)

    if _progress is not None and _opt_task is not None:
        _progress.update(_opt_task, visible=False)

    # Fallback: if no run converged, take best finite result — mirrors
    # main.py's `if best_res is None: finite = [r for r in all_res if ...]`
    if best_x is None:
        finite_runs = [(x, v, i) for x, v, i in all_runs if np.isfinite(v)]
        if not finite_runs:
            _log(f"  [yellow]⚠[/yellow] Power-law Hawkes optimisation failed ({label}).", force=not quiet)
            return None
        best_x, best_val, best_info = min(finite_runs, key=lambda t: t[1])

    mu_s1, n_s1, tau_s1, eta_s1 = map(float, best_x)
    conv_s1 = int(best_info.get("warnflag", 1)) == 0
    _emit(
        f"  [{label}] stage 1 (pure MLE):  "
        f"μ={mu_s1:.4f}  n={n_s1:.4f}  τ={tau_s1:.4f}  η={eta_s1:.4f}  "
        f"nll={best_val:.2f}  nit={best_info.get('nit', '?')}  "
        f"conv={'✓' if conv_s1 else '~'}"
    )

    # ── Stage 2: Rescue degenerate solutions ────────────────────────────
    at_n_cap    = n_s1   > n_upper   * 0.98
    at_tau_floor = tau_s1 < tau_lower * 1.05
    at_n_zero   = n_s1   < 1e-4
    at_eta_cap  = eta_s1 > 9.9
    degenerate  = at_n_cap or at_tau_floor or (at_n_zero and at_eta_cap)

    if degenerate:
        if True:   # always emit — degenerate solutions are always worth reporting
            reasons = []
            if at_n_cap:      reasons.append(f"n={n_s1:.4f} near cap")
            if at_tau_floor:  reasons.append(f"τ={tau_s1:.6f} near floor")
            if at_n_zero and at_eta_cap:
                              reasons.append("n≈0, η≈10 (Poisson collapse)")
            _emit(
                f"  [{label}] [yellow]⚠[/yellow] stage 1 degenerate: {', '.join(reasons)}  "
                f"→ stage 2 with barrier (w={max(1.0, math.sqrt(T.size)*0.1):.2f})..."
            )

        # Scale pen_weight by sqrt(N) so penalty shrinks relative to O(N) likelihood
        pen_weight = max(1.0, math.sqrt(T.size) * 0.1)

        nll_pen    = _rank_seeds_parallel(coarse_T, seeds, tau_lower, n_upper, pen_weight)
        finite_pen = np.isfinite(nll_pen)
        if np.any(finite_pen):
            order_pen  = np.argsort(nll_pen)
            pen_starts = [
                seeds[idx].copy()
                for idx in order_pen
                if finite_pen[idx]
            ][:n_candidates]
        else:
            pen_starts = candidate_starts

        best_x_pen   = None
        best_val_pen = np.inf

        _opt_task2 = None
        if _progress is not None and HAVE_RICH:
            _opt_task2 = _progress.add_task(
                f"    [dim]{label[:18]}[/dim] stage 2  0/{len(pen_starts)}",
                total=len(pen_starts),
                status="starting...",
                visible=True,
            )

        for i, init in enumerate(pen_starts):
            if _progress is not None and _opt_task2 is not None:
                _progress.update(
                    _opt_task2,
                    description=(
                        f"    [dim]{label[:18]}[/dim] "
                        f"stage 2  {i+1}/{len(pen_starts)}"
                    ),
                    status=(
                        f"best nll={best_val_pen:.1f}"
                        if np.isfinite(best_val_pen) else "running..."
                    ),
                )
            x_opt, f_opt, info = fmin_l_bfgs_b(
                func=power_hawkes_nll_grad,
                x0=init,
                args=(T, tau_lower, n_upper, pen_weight),
                bounds=bounds,
                factr=1e7,
                pgtol=1e-8,
                maxiter=1000,
            )
            if np.isfinite(f_opt) and f_opt < best_val_pen:
                best_x_pen, best_val_pen = x_opt, float(f_opt)
            if _progress is not None and _opt_task2 is not None:
                _progress.advance(_opt_task2)

        if _progress is not None and _opt_task2 is not None:
            _progress.update(_opt_task2, visible=False)

        if best_x_pen is not None:
            # Refine from penalised solution with pure MLE
            x_refined, f_refined, info_refined = fmin_l_bfgs_b(
                func=power_hawkes_nll_grad,
                x0=best_x_pen,
                args=(T, tau_lower, n_upper, 0.0),
                bounds=bounds,
                factr=1e7,
                pgtol=1e-8,
                maxiter=1000,
            )
            if np.isfinite(f_refined):
                mu_r, n_r, tau_r, eta_r = map(float, x_refined)
                still_degenerate = (n_r < 1e-4 and eta_r > 9.9) or (n_r > n_upper * 0.98)
                if not still_degenerate:
                    best_x, best_val, best_info = x_refined, float(f_refined), info_refined
                    _emit(
                        f"  [{label}] stage 2 refined:   "
                        f"μ={mu_r:.4f}  n={n_r:.4f}  τ={tau_r:.4f}  η={eta_r:.4f}  "
                        f"nll={f_refined:.2f}  ✓"
                    )
                else:
                    _emit(f"  [{label}] [yellow]⚠[/yellow] stage 2 still degenerate — keeping stage 1.")
            else:
                _emit(f"  [{label}] [yellow]⚠[/yellow] stage 2 refinement non-finite — keeping stage 1.")

    mu, n, tau, eta = map(float, best_x)

    # Final NLL always evaluated without penalty
    final_nll, _ = power_hawkes_nll_grad(best_x, T, tau_lower, n_upper, 0.0)

    success = int(best_info.get("warnflag", 1)) == 0
    result  = PowerFitResult(
        mu=mu, n=n, tau=tau, eta=eta,
        nll=float(final_nll),
        success=success,
        nit=int(best_info.get("nit", -1)),
        message=str(best_info.get("task", "")),
    )

    if not success:
        _emit(
            f"  [{label}] [yellow]⚠[/yellow] optimiser did not fully converge: "
            f"warnflag={best_info.get('warnflag')}  task={best_info.get('task')}"
        )

    # Goodness-of-fit via Papangelou time-change theorem
    residuals = power_residuals(T, result)
    if residuals.size >= 5 and np.all(np.isfinite(residuals)):
        ks = kstest(residuals, "expon")
        result.ks_stat   = float(ks.statistic)
        result.ks_pvalue = float(ks.pvalue)

    ks_str = (f"  KS={result.ks_stat:.4f} (p={result.ks_pvalue:.4g})"
              if result.ks_stat is not None else "")
    _emit(
        f"  [{label}] [bold green]fit complete[/bold green]  "
        f"μ={result.mu:.6f}  n={result.n:.6f}  τ={result.tau:.6f}s  "
        f"η={result.eta:.6f}  c={result.c:.6f}  "
        f"AIC={result.aic:.2f}  nit={result.nit}{ks_str}"
    )

    return result


# =============================================================================
# Plotting
# =============================================================================
def plot_power_hawkes_intensity(
    T:         np.ndarray,
    fit:       PowerFitResult,
    ticker:    str = DEFAULT_TICKER,
    plots_dir: Optional[str] = None,
) -> str:
    """
    Plot the fitted power-law Hawkes intensity λ(t) against the raw event times.

    Follows the exact figure-save pattern from main.py:
        tight_layout → savefig(dpi=300, bbox_inches="tight") → close() → _log
    Includes a rug plot of event times below the intensity trace.
    """
    if plots_dir is None:
        plots_dir = _plots_dir(ticker)

    T      = np.sort(np.ascontiguousarray(T, dtype=np.float64))
    t_grid = np.linspace(T[0], T[-1], 2000)
    lam    = power_intensity_path(T, fit, t_grid)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_grid, lam, color=base.COLORS.get(ticker, "steelblue"),
            lw=1.2, label="power-law λ(t)")
    ax.axhline(fit.mu, color="red", ls="--", lw=1,
               label=f"baseline μ = {fit.mu:.4f}")
    # Rug plot of event times (see reference §2.5)
    ax.plot(T, np.zeros_like(T) - 0.02 * lam.max(), "|",
            color="black", alpha=0.3, ms=6)
    ax.set_xlabel("Time (s from first event)")
    ax.set_ylabel("Intensity λ(t)")
    ax.set_title(
        f"{ticker} — Power-law Hawkes Intensity  "
        f"(n = {fit.n:.3f}, η = {fit.eta:.3f})",
        fontweight="bold",
    )
    ax.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(plots_dir, f"power_hawkes_intensity_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")
    return fname


def plot_power_residual_qqplot(
    T:         np.ndarray,
    fit:       PowerFitResult,
    ticker:    str = DEFAULT_TICKER,
    plots_dir: Optional[str] = None,
) -> str:
    """
    Goodness-of-fit QQ plot and residual histogram.

    Two-panel layout mirrors main.py's plot_residual_qqplot().
    Uses Blom's plotting positions — unbiased expected order statistics.
    """
    if plots_dir is None:
        plots_dir = _plots_dir(ticker)

    residuals = power_residuals(T, fit)
    if residuals.size == 0:
        raise ValueError("No residuals available for QQ plot.")

    quantiles_emp = np.sort(residuals)
    n_resid       = residuals.size
    # Blom's positions: (i - 0.375) / (n + 0.25) — standard for QQ plots
    probs         = (np.arange(1, n_resid + 1) - 0.375) / (n_resid + 0.25)
    quantiles_th  = -np.log(1.0 - probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{ticker} — Power-law Hawkes Goodness-of-Fit", fontweight="bold")

    ax1.plot(quantiles_th, quantiles_emp, ".", alpha=0.45,
             color=base.COLORS.get(ticker, "steelblue"), ms=3)
    lim = max(float(quantiles_th.max()), float(quantiles_emp.max()))
    ax1.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect fit")
    ax1.set_xlabel("Theoretical Exp(1) quantiles")
    ax1.set_ylabel("Empirical quantiles")
    ax1.set_title("Q-Q Plot (residual inter-arrivals)")
    ax1.legend(fontsize=9)

    ax2.hist(residuals, bins=40, density=True,
             color=base.COLORS.get(ticker, "steelblue"),
             alpha=0.7, edgecolor="white")
    xs = np.linspace(0.0, max(float(residuals.max()), 1e-6), 200)
    ax2.plot(xs, np.exp(-xs), "r--", lw=2, label="Exp(1)")
    ax2.set_xlabel("Residual inter-arrival")
    ax2.set_ylabel("Density")
    ax2.set_title("Residual Distribution vs Exp(1)")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fname = os.path.join(plots_dir, f"power_hawkes_qqplot_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")
    return fname


def plot_exp_vs_power_comparison(
    exp_fit:   ExpFitSummary,
    pow_fit:   PowerFitResult,
    ticker:    str = DEFAULT_TICKER,
    plots_dir: Optional[str] = None,
) -> str:
    """
    Three-panel bar chart comparing exponential and power-law fits.

    Bar value annotations follow the same pattern as main.py's cross-stock
    summary plots (bar.get_height() * 1.01, ha='center', fontsize=8).
    """
    if plots_dir is None:
        plots_dir = _plots_dir(ticker)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f"{ticker} — Exponential vs Power-law Hawkes", fontweight="bold")

    clr_exp = "#777777"
    clr_pow = base.COLORS.get(ticker, "steelblue")

    axes[0].bar(["Exp", "Power"], [exp_fit.aic, pow_fit.aic],
                color=[clr_exp, clr_pow], alpha=0.85, edgecolor="white")
    axes[0].set_title("AIC")
    axes[0].set_ylabel("Lower is better")

    axes[1].bar(["Exp", "Power"], [exp_fit.br, pow_fit.n],
                color=[clr_exp, clr_pow], alpha=0.85, edgecolor="white")
    axes[1].axhline(1.0, color="red", ls="--", lw=1, label="stationarity boundary")
    axes[1].set_title("Branching ratio")
    axes[1].legend(fontsize=8)

    axes[2].bar(["Exp μ", "Power μ"], [exp_fit.mu, pow_fit.mu],
                color=[clr_exp, clr_pow], alpha=0.85, edgecolor="white")
    axes[2].set_title("Baseline intensity")
    axes[2].set_ylabel("events / sec")

    # Consistent bar value annotation pattern (reference §2.6)
    for ax in axes:
        for patch in ax.patches:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                height * 1.02 if height > 0 else 0.01,
                f"{height:.3f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    fname = os.path.join(plots_dir, f"exp_vs_power_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}")
    return fname


# =============================================================================
# Rich summary table (mirrors kernel_sum_exp.py's _rich_per_ticker_table)
# =============================================================================
def _rich_fit_table(exp_fit: ExpFitSummary, pow_fit: PowerFitResult, ticker: str) -> "Table":
    """
    Side-by-side comparison table for exponential vs power-law Hawkes.
    Best-per-metric cells get a green ★.
    """
    t = Table(
        title=f"[bold]{ticker}[/bold] — Exponential vs Power-law Hawkes",
        box=box.SIMPLE_HEAD,
        header_style="bold cyan",
        show_lines=False,
        title_justify="left",
    )
    for col, kw in [
        ("Model",  dict(style="bold")),
        ("μ",      dict(justify="right")),
        ("BR / n", dict(justify="right")),
        ("AIC",    dict(justify="right")),
        ("KS",     dict(justify="right")),
        ("OK",     dict(justify="center")),
    ]:
        t.add_column(col, **kw)

    star   = " [bold green]★[/bold green]"
    exp_ok = "✓"
    pow_ok = "✓" if pow_fit.success else "~"

    exp_aic_cell = f"{exp_fit.aic:.1f}" + (star if exp_fit.aic <= pow_fit.aic else "")
    pow_aic_cell = f"{pow_fit.aic:.1f}" + (star if pow_fit.aic <  exp_fit.aic else "")

    pow_ks = (
        f"{pow_fit.ks_stat:.4f} (p={pow_fit.ks_pvalue:.3f})"
        if pow_fit.ks_stat is not None else "—"
    )

    t.add_row("Exponential", f"{exp_fit.mu:.5f}", f"{exp_fit.br:.4f}",
              exp_aic_cell, "—", exp_ok)
    t.add_row("Power-law",   f"{pow_fit.mu:.5f}", f"{pow_fit.n:.4f}",
              pow_aic_cell, pow_ks, pow_ok,
              style="bold" if pow_fit.aic < exp_fit.aic else "")
    return t


# =============================================================================
# Shared progress-bar column layout
# =============================================================================
def _make_progress() -> "Progress":
    """Return a consistently-configured Progress instance on the shared console."""
    return Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]{task.description:<32}"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=console,
        refresh_per_second=10,
    )


# =============================================================================
# Single-ticker runner
# =============================================================================
def run_powerlaw_analysis(
    ticker:    str  = DEFAULT_TICKER,
    start:     str  = base.START_DATE,
    end:       str  = base.END_DATE,
    data_path: str  = base.DATA_PATH,
    quiet:     bool = False,
) -> Dict[str, object]:
    """
    Load data, fit exponential + power-law Hawkes, save diagnostics,
    and return summaries.

    All output — header panel, progress bars and sub-bars, console prints,
    results table, and footer panel — flows through the single shared
    ``console`` / ``Progress`` instance so nothing interleaves with live bars.

    Progress hierarchy
    ------------------
    Progress (1 context, opened for the whole run)
      [cyan]{ticker}[/cyan]         5-stage top-level bar
        ↳ ranking {N} seeds         indeterminate spinner (Numba call)
        ↳ stage 1  k/10             counted bar, status = best nll so far
        ↳ stage 2  k/10             counted bar, only shown when needed
    """
    t0        = time.perf_counter()
    plots_dir = _plots_dir(ticker)
    _N_STAGES = 5   # load, exp fit, pow fit, plots, done

    # ── Header panel ────────────────────────────────────────────────────
    if HAVE_RICH and console is not None and not quiet:
        console.print(Panel(
            f"[bold]Ticker[/bold]    : [cyan]{ticker}[/cyan]\n"
            f"[bold]Period[/bold]    : {start}  →  {end}\n"
            f"[bold]Data[/bold]      : {os.path.abspath(data_path)}\n"
            f"[bold]Plots[/bold]     : {os.path.abspath(plots_dir)}\n"
            f"[bold]Numba[/bold]     : {_NUM_THREADS} threads\n"
            f"[bold]Opt[/bold]       : 81 seeds ranked on coarse N=400 · top 10 full fits",
            title=f"[bold cyan]{ticker} — Power-law Hawkes (Experiment 3)[/bold cyan]",
            border_style="cyan",
        ))

    # ── Inner work — runs inside whichever progress context is active ────
    def _do_run(progress):
        stage_task = progress.add_task(
            f"[cyan]{ticker}[/cyan]",
            total=_N_STAGES,
            status="starting",
        )

        def _plog(msg):
            progress.console.print(msg)

        def _advance(label: str, status: str = ""):
            progress.advance(stage_task)
            progress.update(
                stage_task,
                description=f"[cyan]{ticker}[/cyan] {label}",
                status=status or label,
            )

        # ── Stage: load ───────────────────────────────────────────────
        progress.update(stage_task,
                        description=f"[cyan]{ticker}[/cyan] loading",
                        status="loading...")
        t_load = time.perf_counter()
        T_raw, meta = load_market_orders(
            ticker=ticker, start=start, end=end,
            data_path=data_path, quiet=True,
        )
        t_load = time.perf_counter() - t_load
        if T_raw.size < 20:
            raise RuntimeError(
                f"Only {T_raw.size} events found after filtering; need ≥ 20."
            )
        T         = T_raw - T_raw[0]
        duration  = T[-1]
        mean_rate = T_raw.size / duration if duration > 0 else float("nan")
        mean_ia   = float(np.mean(np.diff(T))) if T.size > 1 else float("nan")
        _plog(
            f"  [{ticker}] loaded [bold]{T_raw.size:,}[/bold] market orders  ·  "
            f"duration {duration:.0f}s  ·  "
            f"rate {mean_rate:.2f} ev/s  ·  "
            f"mean IA {mean_ia*1e3:.2f}ms  ·  "
            f"date {meta['date']}"
        )
        _advance("loaded ✓", f"N={T_raw.size:,}  rate={mean_rate:.2f}/s")

        # ── Stage: exponential fit ────────────────────────────────────
        progress.update(stage_task,
                        description=f"[cyan]{ticker}[/cyan] exp fit",
                        status="fitting exp Hawkes...")
        t_exp      = time.perf_counter()
        exp_params = base.fit_hawkes(T, label=f"{ticker} market orders", quiet=True)
        if exp_params is None:
            raise RuntimeError("Exponential Hawkes fit failed.")
        exp_mu, exp_alpha, exp_beta = map(float, exp_params)
        raw_exp_ll = float(base.hawkes_loglik(
            np.array([exp_mu, exp_alpha, exp_beta], dtype=float), T
        ))
        exp_nll = -raw_exp_ll if raw_exp_ll < 0 else raw_exp_ll
        exp_fit = ExpFitSummary(mu=exp_mu, alpha=exp_alpha, beta=exp_beta, nll=exp_nll)
        t_exp   = time.perf_counter() - t_exp
        _plog(
            f"  [{ticker}] exp fit  "
            f"μ={exp_mu:.4f}  α={exp_alpha:.4f}  β={exp_beta:.4f}  "
            f"BR={exp_fit.br:.3f}  AIC={exp_fit.aic:.1f}  ({t_exp:.1f}s)"
        )
        _advance("exp ✓", f"BR={exp_fit.br:.3f}  AIC={exp_fit.aic:.0f}")

        # ── Stage: power-law fit ──────────────────────────────────────
        progress.update(stage_task,
                        description=f"[cyan]{ticker}[/cyan] pow fit",
                        status="ranking seeds...")
        t_pow   = time.perf_counter()
        pow_fit = fit_power_hawkes(
            T, label=f"{ticker} market orders", quiet=True,
            _plog=_plog, _progress=progress,
        )
        if pow_fit is None:
            raise RuntimeError("Power-law Hawkes fit failed.")
        t_pow = time.perf_counter() - t_pow
        ks_str = (f"  KS={pow_fit.ks_stat:.3f}(p={pow_fit.ks_pvalue:.3f})"
                  if pow_fit.ks_stat is not None else "")
        _plog(
            f"  [{ticker}] pow fit  "
            f"μ={pow_fit.mu:.4f}  n={pow_fit.n:.4f}  "
            f"τ={pow_fit.tau:.4f}s  η={pow_fit.eta:.3f}  "
            f"AIC={pow_fit.aic:.1f}{ks_str}  "
            f"conv={'✓' if pow_fit.success else '~'}  ({t_pow:.1f}s)"
        )
        _advance("pow ✓",
                 f"η={pow_fit.eta:.3f}  n={pow_fit.n:.3f}  AIC={pow_fit.aic:.0f}")

        # ── Stage: plots ──────────────────────────────────────────────
        progress.update(stage_task,
                        description=f"[cyan]{ticker}[/cyan] plots",
                        status="saving plots...")
        t_plot = time.perf_counter()
        intensity_path = plot_power_hawkes_intensity(
            T, pow_fit, ticker=ticker, plots_dir=plots_dir)
        qq_path = plot_power_residual_qqplot(
            T, pow_fit, ticker=ticker, plots_dir=plots_dir)
        cmp_path = plot_exp_vs_power_comparison(
            exp_fit, pow_fit, ticker=ticker, plots_dir=plots_dir)
        t_plot = time.perf_counter() - t_plot
        _advance("plots ✓", "3 figures saved")

        # ── Stage: done ───────────────────────────────────────────────
        winner  = "Power-law" if pow_fit.aic < exp_fit.aic else "Exponential"
        delta   = abs(pow_fit.aic - exp_fit.aic)
        elapsed = time.perf_counter() - t0
        _plog(
            f"  [{ticker}] [bold green]done[/bold green]  "
            f"winner=[bold]{winner}[/bold]  "
            f"ΔAIC={delta:.1f}  "
            f"({elapsed:.1f}s total  "
            f"load={t_load:.1f}s  exp={t_exp:.1f}s  "
            f"pow={t_pow:.1f}s  plots={t_plot:.1f}s)"
        )
        _advance(f"[green]done[/green] → {winner}",
                 f"ΔAIC={delta:.1f}  {elapsed:.1f}s")

        return exp_fit, pow_fit, meta, winner, elapsed, {
            "power_intensity":   intensity_path,
            "power_residual_qq": qq_path,
            "exp_vs_power":      cmp_path,
        }, {
            "load": t_load, "exp_fit": t_exp,
            "pow_fit": t_pow, "plots": t_plot,
            "total": elapsed,
        }

    # ── Run inside the single shared Progress context ────────────────────
    if HAVE_RICH and console is not None and not quiet:
        set_quiet_logs(True)
        if hasattr(base, "set_quiet_logs"):
            base.set_quiet_logs(True)
        try:
            with _make_progress() as progress:
                exp_fit, pow_fit, meta, winner, elapsed, plots, timing = \
                    _do_run(progress)
        finally:
            set_quiet_logs(False)
            if hasattr(base, "set_quiet_logs"):
                base.set_quiet_logs(False)
    else:
        exp_fit, pow_fit, meta, winner, elapsed, plots, timing = \
            _do_run(_NoProgress())

    # ── Results table ─────────────────────────────────────────────────────
    if HAVE_RICH and console is not None and not quiet:
        console.rule("[dim]results[/dim]")
        console.print(_rich_fit_table(exp_fit, pow_fit, ticker))

    # ── Footer panel ─────────────────────────────────────────────────────
    if HAVE_RICH and console is not None and not quiet:
        console.print(Panel(
            f"[bold green]Done[/bold green]  [bold]{elapsed:.1f}s[/bold] total  ·  "
            f"load={timing['load']:.1f}s  "
            f"exp={timing['exp_fit']:.1f}s  "
            f"pow={timing['pow_fit']:.1f}s  "
            f"plots={timing['plots']:.1f}s\n"
            f"winner=[bold]{winner}[/bold]  "
            f"ΔAIC={abs(pow_fit.aic - exp_fit.aic):.1f}  "
            f"η={pow_fit.eta:.3f}  n={pow_fit.n:.3f}\n"
            f"plots → [dim]{os.path.abspath(plots_dir)}[/dim]",
            border_style="green",
        ))
    else:
        print(f"  {ticker}  winner={winner}  "
              f"ΔAIC={abs(pow_fit.aic - exp_fit.aic):.1f}  "
              f"total={elapsed:.1f}s")
        print(f"  plots → {os.path.abspath(plots_dir)}")

    return {
        "ticker" : ticker,
        "meta"   : meta,
        "winner" : winner,
        "exp"    : exp_fit,
        "power"  : pow_fit,
        "plots"  : plots,
        "timing" : timing,
    }


# =============================================================================
# Multi-ticker runner
# =============================================================================
def plot_cross_ticker_summary(results: List[Dict[str, object]]) -> str:
    """
    Six-panel cross-ticker summary figure:
    AIC comparison, ΔAIC, branching ratios, baseline intensities,
    tail exponents, and time scales.

    Bar value annotations follow the reference pattern (§2.6):
    centred 2% above bar top, fontsize=8.
    """
    tickers  = [r["ticker"] for r in results]
    n_tickers = len(tickers)

    exp_aics  = [r["exp"].aic   for r in results]
    pow_aics  = [r["power"].aic for r in results]
    exp_brs   = [r["exp"].br    for r in results]
    pow_brs   = [r["power"].n   for r in results]
    exp_mus   = [r["exp"].mu    for r in results]
    pow_mus   = [r["power"].mu  for r in results]
    pow_etas  = [r["power"].eta for r in results]
    pow_taus  = [r["power"].tau for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Power-law vs Exponential Hawkes — Cross-Ticker Summary",
        fontweight="bold", fontsize=14,
    )

    x = np.arange(n_tickers)
    w = 0.35

    # ── AIC comparison ──────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.bar(x - w / 2, exp_aics, w, label="Exponential", color="#777777",   alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, pow_aics, w, label="Power-law",   color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("AIC (lower is better)"); ax.legend(fontsize=8)

    # ── ΔAIC ────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    delta_aics = [e - p for e, p in zip(exp_aics, pow_aics)]
    colors_d   = ["steelblue" if d > 0 else "#cc4444" for d in delta_aics]
    ax.bar(x, delta_aics, color=colors_d, alpha=0.85, edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("ΔAIC (Exp − Power)  [>0 = Power wins]")
    for i, v in enumerate(delta_aics):
        ax.text(i, v + (50 if v > 0 else -50), f"{v:.0f}",
                ha="center", va="bottom" if v > 0 else "top", fontsize=7)

    # ── Branching ratios ────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.bar(x - w / 2, exp_brs, w, label="Exponential", color="#777777",   alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, pow_brs, w, label="Power-law",   color="steelblue", alpha=0.85, edgecolor="white")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="stationarity")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("Branching Ratio"); ax.legend(fontsize=7)

    # ── Baseline intensities ────────────────────────────────────────────
    ax = axes[1, 0]
    ax.bar(x - w / 2, exp_mus, w, label="Exponential μ", color="#777777",   alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, pow_mus, w, label="Power-law μ",   color="steelblue", alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("Baseline Intensity (events/sec)"); ax.legend(fontsize=8)

    # ── Tail exponents ──────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.bar(x, pow_etas, color=[base.COLORS.get(t, "steelblue") for t in tickers],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("Power-law η (tail exponent)")
    ax.axhline(2.0, color="red", ls="--", lw=1, alpha=0.5, label="η=2 (finite variance)")
    ax.legend(fontsize=8)
    for i, v in enumerate(pow_etas):
        ax.text(i, v * 1.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # ── Time scales (log y-axis) ────────────────────────────────────────
    ax = axes[1, 2]
    ax.bar(x, pow_taus, color=[base.COLORS.get(t, "steelblue") for t in tickers],
           alpha=0.85, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_title("Power-law τ (time scale, sec)")
    ax.set_yscale("log")
    for i, v in enumerate(pow_taus):
        ax.text(i, v * 1.3, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    summary_dir = PLOTS_DIR
    fname = os.path.join(summary_dir, "cross_ticker_summary.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"Saved: {fname}", force=True)
    return fname


def _rich_batch_table(results: List[Dict[str, object]]) -> "Table":
    """
    Cross-ticker summary table: one row per fitted ticker.
    Matches the style of kernel_sum_exp.py's _rich_cross_ticker_table().
    """
    t = Table(
        title="Power-law Hawkes — Batch Summary",
        box=box.SIMPLE_HEAD,
        header_style="bold magenta",
        title_justify="left",
    )
    for col, kw in [
        ("Ticker",  dict(style="bold")),
        ("Winner",  dict(justify="left")),
        ("ΔAIC",    dict(justify="right")),
        ("BR_exp",  dict(justify="right")),
        ("BR_pow",  dict(justify="right")),
        ("η",       dict(justify="right")),
        ("τ (s)",   dict(justify="right")),
        ("μ_pow",   dict(justify="right")),
        ("OK",      dict(justify="center")),
    ]:
        t.add_column(col, **kw)

    star = " [bold green]★[/bold green]"
    for r in results:
        delta   = r["exp"].aic - r["power"].aic
        is_pow  = r["winner"] == "Power-law"
        ok_cell = "✓" if r["power"].success else "~"
        t.add_row(
            r["ticker"],
            ("Power-law" + star) if is_pow else "Exponential",
            f"{delta:+.1f}",
            f"{r['exp'].br:.3f}",
            f"{r['power'].n:.3f}",
            f"{r['power'].eta:.3f}",
            f"{r['power'].tau:.6f}",
            f"{r['power'].mu:.4f}",
            ok_cell,
            style="bold" if is_pow else "",
        )
    return t


def run_all_tickers(
    tickers:   Optional[List[str]] = None,
    start:     str  = base.START_DATE,
    end:       str  = base.END_DATE,
    data_path: str  = base.DATA_PATH,
    quiet:     bool = False,
) -> List[Dict[str, object]]:
    """
    Run the power-law Hawkes analysis for every ticker in ``tickers``.

    A single shared ``Progress`` / ``Console`` instance is opened before any
    work starts and threaded into every sub-routine, so every bar — the outer
    ticker counter, each per-ticker stage bar, the seed-ranking spinner, and
    the stage-1/2 optimisation bars — renders in one contiguous block with no
    interleaving from ``console.print`` lines.

    Progress hierarchy (Rich path)
    ──────────────────────────────
    Progress  (one context for the entire batch)
      tickers  (i/N: TKTR)           outer counted bar
        [cyan]{ticker}[/cyan]         5-stage bar per ticker
          ↳ ranking N seeds           indeterminate spinner (Numba)
          ↳ stage 1  k/10             counted, status = rolling best NLL
          ↳ stage 2  k/10             counted, only shown when stage 1 degenerate
    """
    if tickers is None:
        tickers = ALL_TICKERS

    t_total = time.perf_counter()
    results: List[Dict[str, object]] = []
    failed:  List[Tuple[str, str]]   = []
    _N_STAGES = 5   # load, exp fit, pow fit, plots, done

    # ── Header panel (printed before Progress opens) ─────────────────────
    if HAVE_RICH and console is not None and not quiet:
        console.print(Panel(
            f"[bold]Tickers[/bold]   : [cyan]{', '.join(tickers)}[/cyan]\n"
            f"[bold]Period[/bold]    : {start}  →  {end}\n"
            f"[bold]Data[/bold]      : {os.path.abspath(data_path)}\n"
            f"[bold]Plots[/bold]     : {os.path.abspath(PLOTS_DIR)}\n"
            f"[bold]Numba[/bold]     : {_NUM_THREADS} threads\n"
            f"[bold]Opt[/bold]       : 81 seeds ranked on coarse N=400 · top 10 full fits\n"
            f"[bold]Stages[/bold]    : load → exp fit → pow fit (rank+s1[+s2]) → plots",
            title="[bold cyan]Power-law Hawkes — Batch Run[/bold cyan]",
            border_style="cyan",
        ))

    # ── Per-ticker worker — runs inside the shared Progress context ───────
    def _run_one_ticker(ticker: str, progress, stage_task) -> None:
        """Drives one ticker through all stages; appends to outer ``results``."""

        def _plog(msg: str) -> None:
            progress.console.print(msg)

        def _advance(label: str, status: str = "") -> None:
            progress.advance(stage_task)
            progress.update(
                stage_task,
                description=f"  [cyan]{ticker}[/cyan] {label}",
                status=status or label,
            )

        t_stock = time.perf_counter()
        pdir    = _plots_dir(ticker)

        # ── Load ──────────────────────────────────────────────────────
        progress.update(stage_task,
                        description=f"  [cyan]{ticker}[/cyan] loading",
                        status="loading...")
        try:
            T_raw, meta = load_market_orders(
                ticker=ticker, start=start, end=end,
                data_path=data_path, quiet=True,
            )
        except Exception as e:
            _plog(f"  [red]✗[/red] [{ticker}] load error: {e}")
            progress.update(stage_task, status="[red]load error[/red]")
            return
        if T_raw.size < 20:
            _plog(f"  [yellow]⚠[/yellow] [{ticker}] only {T_raw.size} events — skipping.")
            progress.update(stage_task, status="[red]too few events[/red]")
            return
        T         = T_raw - T_raw[0]
        duration  = T[-1]
        mean_rate = T_raw.size / duration if duration > 0 else float("nan")
        mean_ia   = float(np.mean(np.diff(T))) if T.size > 1 else float("nan")
        _plog(
            f"  [{ticker}] loaded [bold]{T_raw.size:,}[/bold] orders  ·  "
            f"duration {duration:.0f}s  ·  "
            f"rate {mean_rate:.2f} ev/s  ·  "
            f"mean IA {mean_ia*1e3:.2f}ms  ·  "
            f"date {meta['date']}"
        )
        _advance("loaded ✓", f"N={T_raw.size:,}  {mean_rate:.2f}/s")

        # ── Exponential fit ───────────────────────────────────────────
        progress.update(stage_task,
                        description=f"  [cyan]{ticker}[/cyan] exp fit",
                        status="fitting exp Hawkes...")
        t_exp      = time.perf_counter()
        exp_params = base.fit_hawkes(T, label=f"{ticker} market orders", quiet=True)
        if exp_params is None:
            _plog(f"  [yellow]⚠[/yellow] [{ticker}] exp fit failed — skipping.")
            progress.update(stage_task, status="[red]exp fit failed[/red]")
            return
        exp_mu, exp_alpha, exp_beta = map(float, exp_params)
        raw_exp_ll = float(base.hawkes_loglik(
            np.array([exp_mu, exp_alpha, exp_beta], dtype=float), T
        ))
        exp_nll = -raw_exp_ll if raw_exp_ll < 0 else raw_exp_ll
        exp_fit = ExpFitSummary(mu=exp_mu, alpha=exp_alpha, beta=exp_beta, nll=exp_nll)
        t_exp   = time.perf_counter() - t_exp
        _plog(
            f"  [{ticker}] exp fit  "
            f"μ={exp_mu:.4f}  α={exp_alpha:.4f}  β={exp_beta:.4f}  "
            f"BR={exp_fit.br:.3f}  AIC={exp_fit.aic:.1f}  ({t_exp:.1f}s)"
        )
        _advance("exp ✓", f"BR={exp_fit.br:.3f}  AIC={exp_fit.aic:.0f}")

        # ── Power-law fit ─────────────────────────────────────────────
        progress.update(stage_task,
                        description=f"  [cyan]{ticker}[/cyan] pow fit",
                        status="ranking seeds...")
        t_pow   = time.perf_counter()
        pow_fit = fit_power_hawkes(
            T, label=f"{ticker} market orders", quiet=True,
            _plog=_plog, _progress=progress,
        )
        if pow_fit is None:
            _plog(f"  [yellow]⚠[/yellow] [{ticker}] pow fit failed — skipping.")
            progress.update(stage_task, status="[red]pow fit failed[/red]")
            return
        t_pow = time.perf_counter() - t_pow
        ks_str = (f"  KS={pow_fit.ks_stat:.3f}(p={pow_fit.ks_pvalue:.3f})"
                  if pow_fit.ks_stat is not None else "")
        _plog(
            f"  [{ticker}] pow fit  "
            f"μ={pow_fit.mu:.4f}  n={pow_fit.n:.4f}  "
            f"τ={pow_fit.tau:.4f}s  η={pow_fit.eta:.3f}  "
            f"AIC={pow_fit.aic:.1f}{ks_str}  "
            f"conv={'✓' if pow_fit.success else '~'}  ({t_pow:.1f}s)"
        )
        _advance("pow ✓",
                 f"η={pow_fit.eta:.3f}  n={pow_fit.n:.3f}  AIC={pow_fit.aic:.0f}")

        # ── Plots ──────────────────────────────────────────────────────
        progress.update(stage_task,
                        description=f"  [cyan]{ticker}[/cyan] plots",
                        status="saving plots...")
        t_plot = time.perf_counter()
        plot_power_hawkes_intensity(T, pow_fit, ticker=ticker, plots_dir=pdir)
        plot_power_residual_qqplot(T, pow_fit, ticker=ticker, plots_dir=pdir)
        plot_exp_vs_power_comparison(exp_fit, pow_fit, ticker=ticker, plots_dir=pdir)
        t_plot = time.perf_counter() - t_plot
        _advance("plots ✓", "3 figures saved")

        # ── Done ───────────────────────────────────────────────────────
        winner     = "Power-law" if pow_fit.aic < exp_fit.aic else "Exponential"
        delta      = abs(pow_fit.aic - exp_fit.aic)
        elapsed_tk = time.perf_counter() - t_stock
        results.append({
            "ticker": ticker, "meta": meta, "winner": winner,
            "exp": exp_fit, "power": pow_fit,
        })
        _plog(
            f"  [{ticker}] [bold green]done[/bold green]  "
            f"winner=[bold]{winner}[/bold]  ΔAIC={delta:.1f}  "
            f"η={pow_fit.eta:.3f}  n={pow_fit.n:.3f}  "
            f"({elapsed_tk:.1f}s)"
        )
        _advance(f"[green]done[/green] → {winner}", f"ΔAIC={delta:.1f}")

    # ── Outer orchestration — registered into same Progress object ────────
    def _run_batch(progress) -> None:
        outer = progress.add_task(
            f"[bold]tickers[/bold]  (1/{len(tickers)}: {tickers[0]})",
            total=len(tickers),
            status=f"0/{len(tickers)} done",
        )
        for i, ticker in enumerate(tickers):
            progress.update(
                outer,
                description=f"[bold]tickers[/bold]  ({i+1}/{len(tickers)}: {ticker})",
                status=f"{i}/{len(tickers)} done",
            )
            stage_task = progress.add_task(
                f"  [cyan]{ticker}[/cyan]",
                total=_N_STAGES,
                status="queued",
            )
            try:
                _run_one_ticker(ticker, progress=progress, stage_task=stage_task)
            except Exception as e:
                failed.append((ticker, str(e)))
                progress.console.print(f"  [red]✗[/red] [{ticker}] error: {e}")
                progress.update(stage_task, status="[red]error[/red]")
            progress.advance(outer)
        progress.update(
            outer,
            description=f"[bold]tickers[/bold]  (all done)",
            status=f"{len(results)}/{len(tickers)} ok",
        )

    # ── Single shared Progress context ───────────────────────────────────
    if HAVE_RICH and console is not None and not quiet:
        set_quiet_logs(True)
        if hasattr(base, "set_quiet_logs"):
            base.set_quiet_logs(True)
        try:
            with _make_progress() as progress:
                _run_batch(progress)
        finally:
            set_quiet_logs(False)
            if hasattr(base, "set_quiet_logs"):
                base.set_quiet_logs(False)
    else:
        for i, ticker in enumerate(tickers, 1):
            print(f"\n{'=' * 60}")
            print(f"  [{i}/{len(tickers)}]  {ticker}")
            print(f"{'=' * 60}")
            try:
                _run_one_ticker(ticker, progress=_NoProgress(), stage_task=None)
            except Exception as e:
                failed.append((ticker, str(e)))
                print(f"  ✗ {ticker} failed: {e}")

    # ── Rich batch summary table ────────────────────────────────────────
    if HAVE_RICH and console is not None and results and not quiet:
        console.rule("[dim]results[/dim]")
        console.print(_rich_batch_table(results))

    # ── Plain-text summary (always printed) ────────────────────────────
    elapsed = time.perf_counter() - t_total
    _log(f"\n{'=' * 60}", force=True)
    _log(f"  SUMMARY  ({len(results)} succeeded, {len(failed)} failed)", force=True)
    _log(f"{'=' * 60}", force=True)
    _log(
        f"  {'Ticker':<8} {'Winner':<12} {'ΔAIC':>10} {'BR_exp':>8} "
        f"{'BR_pow':>8} {'η':>6} {'τ':>10} {'μ_pow':>8}",
        force=True,
    )
    _log(f"  {'-' * 78}", force=True)
    for r in results:
        delta = r["exp"].aic - r["power"].aic
        _log(
            f"  {r['ticker']:<8} {r['winner']:<12} {delta:>10.1f} "
            f"{r['exp'].br:>8.3f} {r['power'].n:>8.3f} "
            f"{r['power'].eta:>6.3f} {r['power'].tau:>10.6f} "
            f"{r['power'].mu:>8.4f}",
            force=True,
        )
    if failed:
        _log(f"\n  Failed tickers:", force=True)
        for ticker, err in failed:
            _log(f"    {ticker}: {err}", force=True)
    _log(f"\n  Total elapsed: {elapsed:.1f}s", force=True)

    # ── Cross-ticker summary plot ───────────────────────────────────────
    if len(results) >= 2:
        summary_plot = plot_cross_ticker_summary(results)
        _log(f"  Summary plot → {os.path.abspath(summary_plot)}", force=True)

    # ── Footer panel ───────────────────────────────────────────────────
    if HAVE_RICH and console is not None and not quiet:
        pow_winners = sum(1 for r in results if r["winner"] == "Power-law")
        console.print(Panel(
            f"[bold green]Done[/bold green]  "
            f"[bold]{len(results)}/{len(tickers)}[/bold] tickers succeeded  ·  "
            f"[bold]{elapsed:.1f}s[/bold] total\n"
            f"Power-law won [bold]{pow_winners}/{len(results)}[/bold] tickers\n"
            f"plots → [dim]{os.path.abspath(PLOTS_DIR)}[/dim]",
            border_style="green",
        ))

    return results


# =============================================================================
# CLI
# =============================================================================
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fit exact power-law Hawkes model to market-order times."
    )
    p.add_argument("--ticker",    default=None,
                   help="Single ticker symbol (default: GOOG)")
    p.add_argument("--all",       action="store_true", dest="run_all",
                   help=f"Run all tickers: {', '.join(ALL_TICKERS)}")
    p.add_argument("--start",     default=base.START_DATE,
                   help="Start date YYYY-MM-DD")
    p.add_argument("--end",       default=base.END_DATE,
                   help="End date YYYY-MM-DD")
    p.add_argument("--data-path", default=base.DATA_PATH,
                   help="Folder containing LOBSTER CSV files")
    p.add_argument("--quiet",     action="store_true",
                   help="Reduce logging")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()

    if args.ticker and not args.run_all:
        run_powerlaw_analysis(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            data_path=args.data_path,
            quiet=args.quiet,
        )
    else:
        run_all_tickers(
            tickers=ALL_TICKERS,
            start=args.start,
            end=args.end,
            data_path=args.data_path,
            quiet=args.quiet,
        )

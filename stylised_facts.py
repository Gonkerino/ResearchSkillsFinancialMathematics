"""
Stylised facts plots with multi-distribution inter-arrival fitting.

Architecture
────────────
Parallelism has two orthogonal levels that do NOT nest:

  1. Ticker-level  — a ThreadPoolExecutor fans out across tickers.
     Threads are cheap (no pickling, shared memory) and are used solely for
     I/O-bound loading and to schedule the compute phase of each ticker.
     Each ticker's fitting work is handed off to its own dedicated
     ProcessPoolExecutor (see point 2), so threads never block each other
     on CPU work.

  2. Grid-level — each ticker owns one ProcessPoolExecutor for the
     duration of its fitting phase only.  Workers are initialised with that
     ticker's inter-arrival array via _pool_initializer (data sent ONCE
     per worker process, not once per job).  Grid jobs carry only lightweight
     parameter tuples.

Why not one shared process pool across all tickers?
  ProcessPoolExecutor does not support re-initialisation after creation.
  With different data per ticker we need per-ticker pools.  Creating all
  pools simultaneously would oversubscribe CPUs, so ticker concurrency is
  capped at MAX_TICKER_WORKERS (default: 2) and each pool is capped at
  N_WORKERS processes.  Total simultaneous worker processes
  <= MAX_TICKER_WORKERS * N_WORKERS.

Coarse-grid sampling
  Burr12, GenGamma, and Mittag-Leffler coarse grids now use scrambled Sobol
  sequences instead of Cartesian products.  Sobol gives better space-filling
  coverage in fewer points (2^m points used for full quality).  Parameters
  are mapped via log-space transforms where the range spans orders of magnitude.
  The fine phase still uses a deterministic Cartesian neighbourhood around the
  coarse winner.  Each ticker gets a distinct scramble via a ticker-derived
  seed so different tickers explore slightly different parts of parameter space.

  Point counts:
    Burr12   coarse: 2^7 = 128  (was 6*5*4 = 120, Cartesian with correlated structure)
    GenGamma coarse: 2^7 = 128  (was 8*6*4 = 192)
    ML       coarse: 2^6 =  64  (was 7*5   =  35)

Other optimisations preserved from previous refactor
  * Worker initializer — data sent once per process, not per job
  * Persistent pool per ticker — reused across all grid phases of that ticker
  * ACF via np.correlate (vectorised, not a Python loop)
  * AIC reuses already-fitted parameters — no redundant log-likelihood runs
  * Deterministic subsampling via seeded RNG
  * Numba JIT on E_alpha kernel (optional, auto-detected)

NOTE: all multi-start optimisations are LOCAL-SEARCH HEURISTICS.  There is
NO guarantee that the global optimum is found.  Results depend on the
starting-point grid and may differ from a full global search.
"""

import os
import time
import threading
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import (weibull_min, lognorm, gamma, expon,
                         invgauss, fisk, burr12, genpareto, gengamma, lomax)
from scipy.optimize import minimize
from scipy.special import gamma as gamma_fn
from scipy.stats.qmc import Sobol
from rich.progress import (Progress, BarColumn, MofNCompleteColumn,
                           TimeElapsedColumn, TimeRemainingColumn,
                           TextColumn, SpinnerColumn)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

_console = Console()

try:
    from numba import njit as _njit  # type: ignore
    _NUMBA = True
except ImportError:
    _NUMBA = False

from main import Loader, STOCKS, COLORS, DATA_PATH, START_DATE, END_DATE

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

_CPU = os.cpu_count() or 2

# Worker processes per ticker pool.  Capped at 4: each worker imports
# numpy/scipy (~100 MB), so more than 4 quickly exhausts the Windows paging file.
N_WORKERS = min(4, max(1, _CPU - 1))

# How many tickers to fit concurrently.  Each ticker spawns N_WORKERS processes,
# so total live processes <= MAX_TICKER_WORKERS * N_WORKERS.
# Default 2 keeps total processes <= 8, which is safe on most machines.
MAX_TICKER_WORKERS = max(1, min(2, _CPU // max(N_WORKERS, 1)))

# Global random seed for all Sobol scrambling and subsampling.
RANDOM_SEED = 42


# ===========================================================================
# Numba-JIT Mittag-Leffler kernel
# ===========================================================================

if _NUMBA:
    @_njit(cache=True)
    def _gamma_nb(z):
        """Lanczos approximation for Gamma(z), z > 0.  Numba-compatible."""
        if z < 0.5:
            return np.pi / (np.sin(np.pi * z) * _gamma_nb(1.0 - z))
        z -= 1.0
        x = (0.99999999999980993
             + 676.5203681218851    / (z + 1)
             - 1259.1392167224028   / (z + 2)
             + 771.32342877765313   / (z + 3)
             - 176.61502916214059   / (z + 4)
             + 12.507343278686905   / (z + 5)
             - 0.13857109526572012  / (z + 6)
             + 9.9843695780195716e-6 / (z + 7)
             + 1.5056327351493116e-7  / (z + 8))
        t = z + 7.5
        return np.sqrt(2.0 * np.pi) * (t ** (z + 0.5)) * np.exp(-t) * x

    @_njit(cache=True)
    def _e_alpha_jit(x_arr, alpha, n_small=60, n_large=5, thresh=8.0):
        """
        E_alpha(-x) for a 1-D float64 array.
        Gamma table is pre-computed once per call to avoid N*n_terms redundant calls.
        """
        n = len(x_arr)
        result = np.empty(n)

        gamma_small = np.empty(n_small)
        for ki in range(n_small):
            gamma_small[ki] = _gamma_nb(alpha * ki + 1.0)

        gamma_large = np.empty(n_large)
        for ki in range(n_large):
            g = _gamma_nb(1.0 - alpha * (ki + 1))
            gamma_large[ki] = g if (g == g and abs(g) > 1e-300) else 0.0

        for i in range(n):
            x = x_arr[i]
            if x <= thresh:
                val = 0.0
                for ki in range(n_small):
                    sign = 1.0 if ki % 2 == 0 else -1.0
                    g = gamma_small[ki]
                    if g != 0.0:
                        val += sign * (x ** ki) / g
                result[i] = -1.0 if val < -1.0 else (1.0 if val > 1.0 else val)
            else:
                if abs(alpha - 1.0) < 1e-6:
                    result[i] = np.exp(-x)
                else:
                    val = 0.0
                    for ki in range(n_large):
                        g = gamma_large[ki]
                        if g != 0.0:
                            sign = 1.0 if (ki + 1) % 2 == 1 else -1.0
                            val += sign / (g * x ** (ki + 1))
                    result[i] = 0.0 if val < 0.0 else (1.0 if val > 1.0 else val)
        return result

    _e_alpha_jit(np.array([1.0, 9.0]), 0.7)   # warm-up / cache compilation

else:
    def _e_alpha_jit(x_arr, alpha, n_small=60, n_large=5, thresh=8.0):
        """Pure-Python fallback: E_alpha(-x) via series/asymptotic expansion."""
        x_arr  = np.asarray(x_arr, dtype=float)
        result = np.zeros_like(x_arr)
        small  = x_arr <= thresh
        large  = ~small

        if np.any(small):
            xs      = x_arr[small]
            g_cache = np.array([gamma_fn(alpha * ki + 1.0) for ki in range(n_small)])
            powers  = np.array([xs ** ki for ki in range(n_small)])
            signs   = np.array([(-1.0) ** ki for ki in range(n_small)])
            vals    = (signs[:, None] * powers / g_cache[:, None]).sum(axis=0)
            result[small] = np.clip(vals, -1.0, 1.0)

        if np.any(large):
            xl = x_arr[large]
            if abs(alpha - 1.0) < 1e-6:
                result[large] = np.exp(-xl)
            else:
                vals = np.zeros(xl.shape)
                for ki in range(1, n_large + 1):
                    g = gamma_fn(1.0 - alpha * ki)
                    if not np.isfinite(g) or abs(g) < 1e-300:
                        continue
                    sign  = (-1.0) ** (ki + 1)
                    vals += sign / (g * xl ** ki)
                result[large] = np.clip(vals, 0.0, 1.0)

        return result


# ===========================================================================
# MittagLeffler distribution class
# ===========================================================================

class MittagLeffler:
    """
    Mittag-Leffler distribution: S(t) = E_alpha(-(t/scale)^alpha).
    alpha=1 recovers the Exponential.

    PDF is obtained by finite-differencing the SF; no cheap closed-form exists
    in terms of elementary functions.

    NOTE: fitting is a multi-start local-search heuristic — global optimum
    is NOT guaranteed.
    """

    @staticmethod
    def _e_alpha(x, alpha):
        return _e_alpha_jit(np.atleast_1d(np.asarray(x, dtype=np.float64)),
                            float(alpha))

    @classmethod
    def _ml_sf(cls, t, alpha, scale):
        x = (np.asarray(t, dtype=float) / scale) ** alpha
        return np.clip(cls._e_alpha(x, alpha), 0.0, 1.0)

    @classmethod
    def _ml_pdf(cls, t, alpha, scale, eps=1e-6):
        """Central finite-difference of the SF to approximate the PDF."""
        t = np.asarray(t, dtype=float)
        h = np.maximum(t * eps, 1e-12)
        return (cls._ml_sf(np.maximum(t - h, 1e-15), alpha, scale)
                - cls._ml_sf(t + h, alpha, scale)) / (2.0 * h)

    @classmethod
    def logpdf(cls, data, alpha, scale):
        pdf_vals = np.maximum(cls._ml_pdf(data, alpha, scale), 1e-300)
        lp = np.log(pdf_vals)
        return np.where(np.isfinite(lp), lp, -np.inf)

    @classmethod
    def pdf(cls, xs, alpha, scale):
        return cls._ml_pdf(xs, alpha, scale)

    @classmethod
    def sf(cls, xs, alpha, scale):
        return cls._ml_sf(xs, alpha, scale)


# ===========================================================================
# Worker initializer — data sent ONCE per process via pool initializer
# ===========================================================================

_WORKER_DATA = np.empty(0)


def _pool_initializer(data):
    """
    Runs once per worker process at pool creation.  Stores the inter-arrival
    array in module-level _WORKER_DATA so grid jobs need not carry data.
    """
    global _WORKER_DATA
    _WORKER_DATA = data


# ===========================================================================
# Worker functions — read _WORKER_DATA, carry only parameter tuples
# ===========================================================================

def _worker_robust_fit(args):
    """
    Single-start MLE for burr12 / gengamma.
    NOTE: one step of a multi-start heuristic — NOT a global optimum.
    Returns (log_likelihood: float, params: list | None).
    """
    dist_name, x0, bounds = args
    dist = burr12 if dist_name == "burr12" else gengamma
    data = _WORKER_DATA

    def _neg_ll(p):
        try:
            lp = dist.logpdf(data, *p[:-1], loc=0, scale=p[-1])
            v  = -np.sum(lp[np.isfinite(lp)])
            return float(v) if np.isfinite(v) else 1e15
        except Exception:
            return 1e15

    try:
        res = minimize(_neg_ll, x0=list(x0), method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8})
        ll = -_neg_ll(res.x)
        if np.isfinite(ll):
            return (float(ll), res.x.tolist())
    except Exception:
        pass
    return (float("-inf"), None)


def _worker_ml_fit(args):
    """
    Single-start MLE for MittagLeffler.
    NOTE: one step of a multi-start heuristic — NOT a global optimum.
    Returns (neg_log_likelihood: float, params: list | None).
    """
    (x0,) = args
    data  = _WORKER_DATA

    def _ml_neg_ll(params):
        alpha, scale = params
        if alpha <= 0 or alpha > 1 or scale <= 0:
            return 1e15
        pdf_vals = np.maximum(MittagLeffler._ml_pdf(data, alpha, scale), 1e-300)
        if not np.all(np.isfinite(pdf_vals)):
            return 1e15
        return float(-np.log(pdf_vals).sum())

    try:
        res = minimize(_ml_neg_ll, x0=list(x0), method="L-BFGS-B",
                       bounds=[(0.05, 1.0), (1e-9, None)],
                       options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7})
        if np.isfinite(res.fun):
            return (float(res.fun), res.x.tolist())
    except Exception:
        pass
    return (float("inf"), None)


# ===========================================================================
# Sobol coarse-grid generators
# ===========================================================================

def _sobol_burr12_coarse(ia_p10, ia_p90, seed, m=7):
    """
    2^m scrambled Sobol points for the Burr12 coarse grid.
    Dimensions (c, d, scale) all mapped in log-space for better coverage of
    the heavy-tailed parameter region.

    m=7 -> 128 points vs. original 120 Cartesian points, with superior
    space-filling and no correlated structure across shape/scale dimensions.
    """
    sampler = Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random_base2(m=m)   # shape (2^m, 3) in [0, 1]
    log_c_lo,  log_c_hi  = np.log(0.1),             np.log(10.0)
    log_d_lo,  log_d_hi  = np.log(0.1),             np.log(10.0)
    log_sc_lo, log_sc_hi = np.log(max(ia_p10, 1e-12)), np.log(max(ia_p90, 1e-9))
    c_vals     = np.exp(raw[:, 0] * (log_c_hi  - log_c_lo)  + log_c_lo)
    d_vals     = np.exp(raw[:, 1] * (log_d_hi  - log_d_lo)  + log_d_lo)
    scale_vals = np.exp(raw[:, 2] * (log_sc_hi - log_sc_lo) + log_sc_lo)
    return list(zip(c_vals.tolist(), d_vals.tolist(), scale_vals.tolist()))


def _sobol_gengamma_coarse(ia_p10, ia_p90, seed, m=7):
    """
    2^m scrambled Sobol points for the GenGamma coarse grid.
    Dimensions (a, c, scale) all mapped in log-space.

    m=7 -> 128 points vs. original 192 Cartesian points, with better coverage.
    """
    sampler = Sobol(d=3, scramble=True, seed=seed)
    raw = sampler.random_base2(m=m)
    log_a_lo,  log_a_hi  = np.log(0.05),            np.log(10.0)
    log_c_lo,  log_c_hi  = np.log(0.05),            np.log(5.0)
    log_sc_lo, log_sc_hi = np.log(max(ia_p10, 1e-12)), np.log(max(ia_p90, 1e-9))
    a_vals     = np.exp(raw[:, 0] * (log_a_hi  - log_a_lo)  + log_a_lo)
    c_vals     = np.exp(raw[:, 1] * (log_c_hi  - log_c_lo)  + log_c_lo)
    scale_vals = np.exp(raw[:, 2] * (log_sc_hi - log_sc_lo) + log_sc_lo)
    return list(zip(a_vals.tolist(), c_vals.tolist(), scale_vals.tolist()))


def _sobol_ml_coarse(ia_p10, ia_p90, seed, m=6):
    """
    2^m scrambled Sobol points for the Mittag-Leffler coarse grid.
    alpha is mapped linearly in [0.05, 1.0]; scale is mapped in log-space.

    m=6 -> 64 points vs. original 35 Cartesian points, with better 2-D coverage.
    """
    sampler = Sobol(d=2, scramble=True, seed=seed)
    raw = sampler.random_base2(m=m)
    alpha_vals = raw[:, 0] * (1.0 - 0.05) + 0.05   # linear: alpha is bounded [0,1]
    log_sc_lo  = np.log(max(ia_p10, 1e-12))
    log_sc_hi  = np.log(max(ia_p90, 1e-9))
    scale_vals = np.exp(raw[:, 1] * (log_sc_hi - log_sc_lo) + log_sc_lo)
    return list(zip(alpha_vals.tolist(), scale_vals.tolist()))


# ===========================================================================
# Grid-search dispatch helpers — accept a caller-supplied pool
# ===========================================================================

def _grid_robust_fit(dist_name, param_grid, bounds, pool, desc,
                     progress=None, task_id=None):
    """
    Dispatch a parameter grid to the shared pool; return the best parameters.
    NOTE: multi-start local optimisation — global optimum NOT guaranteed.
    """
    jobs    = [(dist_name, list(x0), bounds) for x0 in param_grid]
    best_ll = float("-inf")
    best_par = None

    if progress is not None and task_id is not None:
        progress.reset(task_id, total=len(jobs),
                       description=f"  {desc}", visible=True)

    for fut in as_completed({pool.submit(_worker_robust_fit, j): j for j in jobs}):
        ll, par = fut.result()
        if ll > best_ll:
            best_ll, best_par = ll, par
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    if progress is not None and task_id is not None:
        progress.update(task_id, visible=False)

    return np.array(best_par) if best_par is not None else None


def _grid_ml_fit(param_grid, pool, desc, progress=None, task_id=None):
    """
    Dispatch a Mittag-Leffler grid to the shared pool.
    NOTE: multi-start local optimisation — global optimum NOT guaranteed.
    Returns (best_neg_ll: float, best_params: np.ndarray | None).
    """
    jobs        = [(list(x0),) for x0 in param_grid]
    best_neg_ll = float("inf")
    best_par    = None

    if progress is not None and task_id is not None:
        progress.reset(task_id, total=len(jobs),
                       description=f"  {desc}", visible=True)

    for fut in as_completed({pool.submit(_worker_ml_fit, j): j for j in jobs}):
        neg_ll, par = fut.result()
        if neg_ll < best_neg_ll:
            best_neg_ll, best_par = neg_ll, par
        if progress is not None and task_id is not None:
            progress.advance(task_id)

    if progress is not None and task_id is not None:
        progress.update(task_id, visible=False)

    return best_neg_ll, (np.array(best_par) if best_par is not None else None)


# ===========================================================================
# Fine neighbourhood builder (Cartesian, centred on coarse winner)
# ===========================================================================

def _neighbourhood(centre, param_bounds, n_fine=4):
    """Cartesian n_fine^D grid centred at centre, clipped to bounds."""
    grids = []
    for val, (lo, hi) in zip(centre, param_bounds):
        lo_v = max(lo if lo is not None else 1e-9, val * 0.5)
        hi_v = min(hi if hi is not None else val * 10, val * 1.5)
        grids.append(np.linspace(lo_v, hi_v, n_fine))
    mesh = np.array(np.meshgrid(*grids)).T.reshape(-1, len(grids))
    return [tuple(p) for p in mesh]


# ===========================================================================
# ACF helper (vectorised)
# ===========================================================================

def _acf(x, max_lag):
    """Sample ACF for lags 1..max_lag via np.correlate (no Python loop)."""
    if max_lag < 1 or len(x) < 2:
        return np.array([])
    xc  = x - x.mean()
    var = float(np.dot(xc, xc))
    if var == 0.0:
        return np.zeros(max_lag)
    full = np.correlate(xc, xc, mode="full")
    mid  = len(full) // 2
    return full[mid + 1: mid + 1 + max_lag] / var


# ===========================================================================
# CCDF helper
# ===========================================================================

def _ccdf(x):
    xs = np.sort(x)
    return xs, 1.0 - np.arange(1, len(xs) + 1) / len(xs)


# ===========================================================================
# Core per-ticker computation
# ===========================================================================

def _compute_ticker(df, ticker, pool, progress=None, stage_task=None,
                    grid_task=None):
    """
    Fit all 11 distributions for one ticker and produce its per-ticker plot.

    pool must already have been created with _pool_initializer seeded with
    this ticker's inter-arrival data (handled by the caller).

    Coarse grids use scrambled Sobol sequences derived from a ticker-specific
    seed for deterministic but well-distributed starting points.
    Fine grids use a Cartesian neighbourhood around the coarse winner.

    NOTE: Burr12, GenGamma, and Mittag-Leffler fitting are multi-start
    LOCAL-SEARCH HEURISTICS.  The global optimum is NOT guaranteed.
    """
    def _log(msg):
        if progress is not None:
            progress.console.print(msg)
        else:
            print(msg)

    mo = df[df["Type"] == 4].copy()
    if len(mo) < 10:
        _log(f"  [{ticker}] Skipping — not enough market orders.")
        return None

    inter_arrival = np.diff(mo["Time"].values)
    inter_arrival = inter_arrival[inter_arrival > 0]
    if len(inter_arrival) < 10:
        _log(f"  [{ticker}] Skipping — not enough positive inter-arrivals.")
        return None

    if progress is not None and stage_task is not None:
        progress.reset(stage_task, total=6,
                       description=f"[cyan]{ticker}[/cyan]", visible=True)

    mo["SignedMove"] = mo["TradeDirection"] * mo["Size"]
    X = mo["SignedMove"].values

    ia_p10    = float(np.percentile(inter_arrival, 10))
    ia_p90    = float(np.percentile(inter_arrival, 90))
    ia_median = float(np.median(inter_arrival))
    ia_min    = float(inter_arrival.min())
    ia_max    = float(inter_arrival.max())
    if ia_min == ia_max:
        ia_max = ia_min * 1.0001

    # Per-ticker Sobol seed: deterministic but distinct per ticker.
    ticker_seed = RANDOM_SEED ^ (hash(ticker) & 0xFFFF)

    def _advance(label):
        if progress is not None and stage_task is not None:
            progress.advance(stage_task)
            progress.update(stage_task,
                            description=f"[cyan]{ticker}[/cyan] {label}")

    # ------------------------------------------------------------------
    # Stage 1 — closed-form / single-start fits
    # ------------------------------------------------------------------
    k,     _, lam      = weibull_min.fit(inter_arrival, floc=0)
    a_g,   _, scale_g  = gamma.fit(inter_arrival, floc=0)
    s_l,   _, scale_l  = lognorm.fit(inter_arrival, floc=0)
    _,        scale_e  = expon.fit(inter_arrival, floc=0)
    mu_ig, _, scale_ig = invgauss.fit(inter_arrival, floc=0)
    c_fl,  _, scale_fl = fisk.fit(inter_arrival, floc=0)
    c_gp,  _, scale_gp = genpareto.fit(inter_arrival, floc=0)
    c_lx,  _, scale_lx = lomax.fit(inter_arrival, floc=0)
    _advance("simple fits ✓")

    # ------------------------------------------------------------------
    # Stage 2 — Burr12  (Sobol coarse -> Cartesian fine)
    # ------------------------------------------------------------------
    # Burr12 optimisation bounds.
    # Upper limit on c/d set to 30 (not 50) and lower limit on d set to 0.05
    # to prevent L-BFGS-B from escaping to degenerate boundary solutions
    # (c→50, d→0) that produce spuriously high log-likelihoods by concentrating
    # all mass at a single point.  The Sobol coarse grid already samples c,d
    # in [0.1, 10] so this tighter outer bound does not restrict valid fits.
    BURR_C_MAX = 30.0
    BURR_D_MIN = 0.05
    burr_bounds = [(0.05, BURR_C_MAX), (BURR_D_MIN, 30.0), (1e-9, None)]

    burr_coarse = _sobol_burr12_coarse(ia_p10, ia_p90, seed=ticker_seed, m=7)
    bc_res = _grid_robust_fit("burr12", burr_coarse, burr_bounds, pool,
                              desc=f"{ticker} Burr12 coarse ({len(burr_coarse)})",
                              progress=progress, task_id=grid_task)
    if bc_res is not None:
        bf_grid = _neighbourhood(bc_res, [(0.05, BURR_C_MAX), (BURR_D_MIN, 30.0), (1e-9, None)])
        bf_res  = _grid_robust_fit("burr12", bf_grid, burr_bounds, pool,
                                   desc=f"{ticker} Burr12 fine ({len(bf_grid)})",
                                   progress=progress, task_id=grid_task)
        burr_res = bf_res if bf_res is not None else bc_res
    else:
        burr_res = None

    def _burr_at_boundary(c, d):
        """True if optimiser converged to a degenerate boundary solution."""
        return c > BURR_C_MAX * 0.95 or d < BURR_D_MIN * 1.05

    if burr_res is not None and not _burr_at_boundary(burr_res[0], burr_res[1]):
        # Sobol/fine search found a valid interior solution.
        c_b, d_b, scale_b = burr_res
        _burr12_ok = True
    else:
        # Either coarse search failed entirely, or the best result was at a
        # degenerate boundary.  Fall back to scipy.fit and validate it.
        if burr_res is not None:
            _log(f"  [{ticker}] Warning: Burr12 Sobol result at parameter boundary "
                 f"(c={burr_res[0]:.2f}, d={burr_res[1]:.4f}) — retrying with scipy.fit.")
        c_b, d_b, _, scale_b = burr12.fit(inter_arrival, floc=0)
        _check = burr12.logpdf(inter_arrival[:min(500, len(inter_arrival))],
                               c=c_b, d=d_b, loc=0, scale=scale_b)
        frac_finite = np.isfinite(_check).mean()
        if _burr_at_boundary(c_b, d_b) or frac_finite < 0.80:
            _log(f"  [{ticker}] Warning: Burr12 scipy fallback also degenerate "
                 f"(c={c_b:.2f}, d={d_b:.4f}, finite={frac_finite:.0%}) — "
                 f"marking Burr12 as failed for this ticker.")
            c_b, d_b, scale_b = float(k), 1.0, float(lam)
            _burr12_ok = False
        else:
            _burr12_ok = True
    _advance(f"Burr12 {'✓' if _burr12_ok else '✗'} c={c_b:.2f} d={d_b:.2f}")

    # ------------------------------------------------------------------
    # Stage 3 — GenGamma  (Sobol coarse -> Cartesian fine)
    # ------------------------------------------------------------------
    gg_bounds = [(1e-4, 50.0), (1e-4, 50.0), (1e-9, None)]

    gg_coarse = _sobol_gengamma_coarse(ia_p10, ia_p90, seed=ticker_seed + 1, m=7)
    gc_res = _grid_robust_fit("gengamma", gg_coarse, gg_bounds, pool,
                              desc=f"{ticker} GenGamma coarse ({len(gg_coarse)})",
                              progress=progress, task_id=grid_task)
    if gc_res is not None:
        gf_grid = _neighbourhood(gc_res, [(1e-4, 50), (1e-4, 50), (1e-9, None)])
        gf_res  = _grid_robust_fit("gengamma", gf_grid, gg_bounds, pool,
                                   desc=f"{ticker} GenGamma fine ({len(gf_grid)})",
                                   progress=progress, task_id=grid_task)
        gg_res = gf_res if gf_res is not None else gc_res
    else:
        gg_res = None

    if gg_res is not None:
        a_gg, c_gg, scale_gg = gg_res
    else:
        a_gg, c_gg, _, scale_gg = gengamma.fit(inter_arrival, floc=0)
    _advance(f"GenGamma ✓ a={a_gg:.2f} c={c_gg:.2f}")

    # ------------------------------------------------------------------
    # Stage 4 — MittagLeffler  (Sobol coarse on subsample -> Cartesian fine)
    # Subsampling uses a seeded numpy RNG for deterministic results.
    # NOTE: multi-start heuristic — global optimum NOT guaranteed.
    # ------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)
    ml_c_data = (inter_arrival if len(inter_arrival) <= 500
                 else rng.choice(inter_arrival, 500, replace=False))

    ml_coarse = _sobol_ml_coarse(
        float(np.percentile(ml_c_data, 10)),
        float(np.percentile(ml_c_data, 90)),
        seed=ticker_seed + 2,
        m=6,
    )
    _, ml_c_par = _grid_ml_fit(ml_coarse, pool,
                                desc=f"{ticker} ML coarse ({len(ml_coarse)})",
                                progress=progress, task_id=grid_task)

    if ml_c_par is not None:
        a_c, sc_c  = float(ml_c_par[0]), float(ml_c_par[1])
        alpha_vals = np.clip(np.linspace(a_c * 0.7, a_c * 1.3, 5), 0.05, 1.0)
        scale_vals = np.linspace(sc_c * 0.5, sc_c * 1.5, 5)
        ml_fine    = [(float(a), float(s))
                      for a in alpha_vals for s in scale_vals]
        # Fine phase evaluates on the full inter_arrival held in _WORKER_DATA.
        _, ml_f_par = _grid_ml_fit(ml_fine, pool,
                                   desc=f"{ticker} ML fine ({len(ml_fine)})",
                                   progress=progress, task_id=grid_task)
        ml_best = ml_f_par if ml_f_par is not None else ml_c_par
    else:
        ml_best = None

    if ml_best is not None:
        ml_alpha  = float(np.clip(ml_best[0], 0.05, 1.0))
        ml_scale  = float(abs(ml_best[1]))
        ml_fit_ok = True
        _advance(f"ML ✓ alpha={ml_alpha:.2f}")
    else:
        _log(f"  [{ticker}] Warning: Mittag-Leffler fit failed — excluded from AIC.")
        ml_fit_ok = False
        ml_alpha, ml_scale = 0.7, ia_median
        _advance("ML ✗")

    # ------------------------------------------------------------------
    # Stage 5 — Plot
    # ------------------------------------------------------------------
    log_bins = np.logspace(np.log10(ia_min), np.log10(ia_max), 60)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle(f"{ticker} - Stylised Facts ({df.Date.iloc[0]})",
                 fontsize=13, fontweight="bold")

    ax1.hist(inter_arrival, bins=log_bins, density=True,
             color=COLORS.get(ticker, "steelblue"), alpha=0.7, edgecolor="white")

    xs           = np.logspace(np.log10(ia_min), np.log10(ia_max), 400)
    ys_weibull   = weibull_min.pdf(xs, c=k,         loc=0, scale=lam)
    ys_gamma     = gamma.pdf(xs,     a=a_g,          loc=0, scale=scale_g)
    ys_lognorm   = lognorm.pdf(xs,   s=s_l,          loc=0, scale=scale_l)
    ys_exp       = expon.pdf(xs,                      loc=0, scale=scale_e)
    ys_invgauss  = invgauss.pdf(xs,  mu=mu_ig,        loc=0, scale=scale_ig)
    ys_fisk      = fisk.pdf(xs,      c=c_fl,          loc=0, scale=scale_fl)
    ys_burr12    = burr12.pdf(xs,    c=c_b, d=d_b,   loc=0, scale=scale_b)
    ys_genpareto = genpareto.pdf(xs, c=c_gp,          loc=0, scale=scale_gp)
    ys_gengamma  = gengamma.pdf(xs,  a=a_gg, c=c_gg, loc=0, scale=scale_gg)
    ys_lomax     = np.where(xs > 0, lomax.pdf(xs, c=c_lx, loc=0, scale=scale_lx), np.nan)
    if ml_fit_ok:
        ys_ml = MittagLeffler.pdf(xs, ml_alpha, ml_scale)
        ys_ml = np.where(np.isfinite(ys_ml) & (ys_ml > 0), ys_ml, np.nan)
    else:
        ys_ml = np.full_like(xs, np.nan)

    ax1.plot(xs, ys_weibull,   color="#d62728", ls="--",  lw=2.0,
             label=f"Weibull(k={k:.2f}, lam={lam:.2f})")
    ax1.plot(xs, ys_gamma,     color="#9467bd", ls="-.",  lw=1.8,
             label=f"Gamma(a={a_g:.2f}, th={scale_g:.2f})")
    ax1.plot(xs, ys_lognorm,   color="#2ca02c", ls=":",   lw=2.2,
             label=f"Lognormal(s={s_l:.2f}, sc={scale_l:.2f})")
    ax1.plot(xs, ys_exp,       color="#ff7f0e", ls="-",   lw=1.8,
             label=f"Exponential(lam={1.0/scale_e:.2f})")
    ax1.plot(xs, ys_invgauss,  color="#1f77b4", ls="--",  lw=1.6,
             label=f"InvGaussian(shape={mu_ig:.2f}, sc={scale_ig:.2f})")
    ax1.plot(xs, ys_fisk,      color="#8c564b", ls="-.",  lw=1.6,
             label=f"LogLogistic(c={c_fl:.2f}, sc={scale_fl:.2f})")
    ax1.plot(xs, ys_burr12,    color="#e377c2", ls=":",   lw=1.8,
             label=f"Burr12(c={c_b:.2f}, d={d_b:.2f})")
    ax1.plot(xs, ys_genpareto, color="#7f7f7f", ls="-",   lw=1.6,
             label=f"GenPareto(c={c_gp:.2f}, sc={scale_gp:.2f})")
    ax1.plot(xs, ys_gengamma,  color="#bcbd22", ls="--",  lw=1.6,
             label=f"GenGamma(a={a_gg:.2f}, c={c_gg:.2f})")
    ax1.plot(xs, ys_lomax,     color="#17becf", ls="-",   lw=2.0,
             label=f"Lomax(c={c_lx:.2f}, sc={scale_lx:.2f}) [power-law]")
    ml_label = (f"MittagLeffler(alpha={ml_alpha:.2f}, sc={ml_scale:.2e}) [PL-kernel]"
                if ml_fit_ok else "MittagLeffler (fit failed)")
    ax1.plot(xs, ys_ml, color="#e41a1c", ls="-", lw=2.2, label=ml_label)

    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Inter-arrival time (s) - log")
    ax1.set_ylabel("Density")
    ax1.set_title("Market-order Inter-arrival Times")
    ax1.legend(fontsize=9)
    ax1.set_xlim(left=1e-6, right=5e2)
    ax1.set_ylim(bottom=1e-5, top=1e4)

    max_lag = min(30, len(X) - 2)
    if max_lag >= 1:
        acf_arr = _acf(X, max_lag)
        lags    = range(1, max_lag + 1)
        ax2.bar(lags, acf_arr, color=COLORS.get(ticker, "steelblue"), alpha=0.8)
        ax2.axhline(0, color="black", lw=0.8)
        ci = 1.96 / np.sqrt(len(X))
        ax2.axhline( ci, color="red", ls="--", lw=1, label="95% CI (i.i.d.)")
        ax2.axhline(-ci, color="red", ls="--", lw=1)
        ax2.legend(fontsize=9)
    else:
        ax2.text(0.5, 0.5, "Not enough data for ACF", ha="center", va="center",
                 transform=ax2.transAxes)
        ax2.axhline(0, color="black", lw=0.8)
        acf_arr = np.array([])

    ax2.set_xlabel("Lag"); ax2.set_ylabel("Autocorrelation")
    ax2.set_title("Signed Trade-size Autocorrelation")
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, f"stylised_facts_multi_dist_{ticker}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _advance("plot ✓")

    # ------------------------------------------------------------------
    # Stage 6 — AIC  (reuse fitted params; no extra minimisation)
    #
    # All log-likelihood sums filter non-finite values before summing.
    # A single -inf logpdf (e.g. from a degenerate scipy.fit fallback or a
    # data point falling outside a distribution's support) would otherwise
    # make the whole .sum() equal -inf and produce AIC = +inf.  We treat
    # such points as having zero contribution to the likelihood — equivalent
    # to excluding them — and report -inf only when no finite values exist.
    # ------------------------------------------------------------------
    def _safe_ll(lp_arr):
        """Sum finite logpdf values; return -inf if none are finite."""
        finite = lp_arr[np.isfinite(lp_arr)]
        return float(finite.sum()) if len(finite) > 0 else float("-inf")

    ll_weibull   = _safe_ll(weibull_min.logpdf(inter_arrival, c=k,    loc=0, scale=lam))
    ll_gamma     = _safe_ll(gamma.logpdf(    inter_arrival, a=a_g,   loc=0, scale=scale_g))
    ll_lognorm   = _safe_ll(lognorm.logpdf(  inter_arrival, s=s_l,   loc=0, scale=scale_l))
    ll_exp       = _safe_ll(expon.logpdf(    inter_arrival,           loc=0, scale=scale_e))
    ll_invgauss  = _safe_ll(invgauss.logpdf( inter_arrival, mu=mu_ig, loc=0, scale=scale_ig))
    ll_fisk      = _safe_ll(fisk.logpdf(     inter_arrival, c=c_fl,  loc=0, scale=scale_fl))
    ll_burr12    = (_safe_ll(burr12.logpdf(inter_arrival, c=c_b, d=d_b, loc=0, scale=scale_b))
                    if _burr12_ok else float("-inf"))
    ll_genpareto = _safe_ll(genpareto.logpdf(inter_arrival, c=c_gp,  loc=0, scale=scale_gp))
    ll_gengamma  = _safe_ll(gengamma.logpdf( inter_arrival, a=a_gg, c=c_gg, loc=0, scale=scale_gg))
    ll_lomax     = _safe_ll(lomax.logpdf(    inter_arrival, c=c_lx,  loc=0, scale=scale_lx))
    ll_ml        = (_safe_ll(MittagLeffler.logpdf(inter_arrival, ml_alpha, ml_scale))
                    if ml_fit_ok else float("-inf"))

    aic = {
        "Weibull":       2 * 2 - 2 * ll_weibull,
        "Gamma":         2 * 2 - 2 * ll_gamma,
        "Lognormal":     2 * 2 - 2 * ll_lognorm,
        "Exponential":   2 * 1 - 2 * ll_exp,
        "InvGaussian":   2 * 2 - 2 * ll_invgauss,
        "LogLogistic":   2 * 2 - 2 * ll_fisk,
        "Burr12":        2 * 3 - 2 * ll_burr12,
        "GenPareto":     2 * 2 - 2 * ll_genpareto,
        "GenGamma":      2 * 3 - 2 * ll_gengamma,
        "Lomax":         2 * 2 - 2 * ll_lomax,
        "MittagLeffler": 2 * 2 - 2 * ll_ml,
    }
    aic_winner = min(aic, key=aic.get)
    _advance(f"[green]done[/green] AIC->{aic_winner}")
    _log(f"  [{ticker}] ✓  AIC: {aic_winner}  |  ML alpha={ml_alpha:.2f}  |  "
         f"{len(inter_arrival):,} events")

    return {
        "ticker":         ticker,
        "inter_arrival":  inter_arrival,
        "acf":            acf_arr,
        "gamma_a":        float(a_g),
        "gamma_scale":    float(scale_g),
        "exp_scale":      float(scale_e),
        "lomax_c":        float(c_lx),
        "lomax_scale":    float(scale_lx),
        "lognorm_s":      float(s_l),
        "lognorm_scale":  float(scale_l),
        "gengamma_a":     float(a_gg),
        "gengamma_c":     float(c_gg),
        "gengamma_scale": float(scale_gg),
        "ml_alpha":       float(ml_alpha),
        "ml_scale":       float(ml_scale),
        "ml_fit_ok":      ml_fit_ok,
        "date":           str(df.Date.iloc[0]),
        "aic":            aic,
    }


# ===========================================================================
# AIC comparison plot + console table
# ===========================================================================

def plot_aic_comparison(results):
    if not results:
        return

    dist_names = ["Weibull", "Gamma", "Lognormal", "Exponential",
                  "InvGaussian", "LogLogistic", "Burr12", "GenPareto",
                  "GenGamma", "Lomax", "MittagLeffler"]
    tickers = list(results.keys())

    delta_aic = {}
    raw_aic   = {}
    winners   = {}

    for ticker, res in results.items():
        if "aic" not in res:
            continue
        row  = res["aic"]
        best = min(row.values())
        winners[ticker]   = min(row, key=row.get)
        raw_aic[ticker]   = row
        delta_aic[ticker] = {d: row[d] - best for d in dist_names}

    if not delta_aic:
        _console.print("  No AIC data to plot.")
        return

    # ── Raw AIC table ─────────────────────────────────────────────────────
    tbl = Table(
        title="AIC Comparison  —  raw AIC, lower is better",
        box=box.SIMPLE_HEAD, header_style="bold magenta", title_justify="left",
    )
    tbl.add_column("Ticker", style="bold")
    for d in dist_names:
        tbl.add_column(d[:11], justify="right")
    tbl.add_column("Winner", style="bold green", justify="right")
    for t in tickers:
        if t not in raw_aic:
            continue
        best_d = winners[t]
        cells  = [t]
        for d in dist_names:
            v    = raw_aic[t][d]
            star = " [bold green]★[/bold green]" if d == best_d else ""
            cells.append(f"{v:.1f}{star}")
        cells.append(best_d)
        tbl.add_row(*cells)
    _console.print(tbl)

    # ── ΔAIC table ────────────────────────────────────────────────────────
    dtbl = Table(
        title="ΔAIC  (raw AIC − best AIC for that ticker)  —  winner at 0",
        box=box.SIMPLE_HEAD, header_style="bold cyan", title_justify="left",
    )
    dtbl.add_column("Ticker", style="bold")
    for d in dist_names:
        dtbl.add_column(d[:11], justify="right")
    for t in tickers:
        if t not in delta_aic:
            continue
        cells = [t]
        for d in dist_names:
            v = delta_aic[t][d]
            cells.append(
                f"[bold green]{v:.1f}[/bold green]" if v == 0.0 else f"{v:.1f}"
            )
        dtbl.add_row(*cells)
    _console.print(dtbl)

    n_t     = len([t for t in tickers if t in delta_aic])
    n_d     = len(dist_names)
    x       = np.arange(n_t)
    dcolors = {
        "Weibull":       "#d62728", "Gamma":       "#9467bd",
        "Lognormal":     "#2ca02c", "Exponential": "#ff7f0e",
        "InvGaussian":   "#1f77b4", "LogLogistic": "#8c564b",
        "Burr12":        "#e377c2", "GenPareto":   "#7f7f7f",
        "GenGamma":      "#bcbd22", "Lomax":       "#17becf",
        "MittagLeffler": "#e41a1c",
    }
    _, ax   = plt.subplots(figsize=(max(14, 3 * n_t), 5))
    valid_t = [t for t in tickers if t in delta_aic]
    bar_w   = 0.07
    offsets = np.linspace(-(n_d - 1) / 2, (n_d - 1) / 2, n_d) * bar_w
    for i, dist in enumerate(dist_names):
        vals = [max(delta_aic[t][dist], 1e-2) for t in valid_t]
        ax.bar(x + offsets[i], vals, width=bar_w,
               label=dist, color=dcolors[dist], alpha=0.85, edgecolor="white")

    max_d = max(delta_aic[t][d] for t in valid_t for d in dist_names)
    if max_d > 100:
        ax.set_yscale("log"); ax.set_ylabel("dAIC (log scale)")
    else:
        ax.set_ylabel("dAIC")
    ax.set_xticks(x); ax.set_xticklabels(valid_t)
    ax.set_xlabel("Ticker")
    ax.set_title("AIC Model Comparison — dAIC per Ticker\n(winner at zero; lower is better)")
    ax.legend(title="Distribution", fontsize=9)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "aic_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _console.print(f"    [dim]→ {fname}[/dim]")




# ===========================================================================
# Raw AIC + AIC/N supplementary figure
# ===========================================================================

def plot_aic_raw_and_normalised(results):
    """
    Two-panel supplementary AIC figure.

    Left  — Raw AIC per distribution per ticker.
      Grouped bars, one cluster per ticker.  Shows absolute scale and the
      margin between distributions within each ticker.  Not directly
      comparable across tickers because log-likelihood scales with N.

    Right — AIC/N (per-observation AIC = AIC / n_events).
      Equivalent to -2 * mean_log_likelihood + 2k/N.  Puts all tickers on
      the same scale so cross-ticker fit quality is directly comparable.
      Lower (more negative) is better.

    Both panels share the same distribution colour scheme as the dAIC plot.
    Degenerate fits (AIC = +inf) are plotted at the panel ceiling and
    annotated with an infinity symbol rather than being silently clipped.
    """
    if not results:
        return

    dist_names = ["Weibull", "Gamma", "Lognormal", "Exponential",
                  "InvGaussian", "LogLogistic", "Burr12", "GenPareto",
                  "GenGamma", "Lomax", "MittagLeffler"]
    dcolors = {
        "Weibull":       "#d62728", "Gamma":       "#9467bd",
        "Lognormal":     "#2ca02c", "Exponential": "#ff7f0e",
        "InvGaussian":   "#1f77b4", "LogLogistic": "#8c564b",
        "Burr12":        "#e377c2", "GenPareto":   "#7f7f7f",
        "GenGamma":      "#bcbd22", "Lomax":       "#17becf",
        "MittagLeffler": "#e41a1c",
    }

    # Only tickers that have both AIC dict and event-count data.
    valid_tickers = [
        t for t, res in results.items()
        if "aic" in res and "inter_arrival" in res
    ]
    if not valid_tickers:
        _console.print("  No AIC/N data to plot.")
        return

    n_events = {t: len(results[t]["inter_arrival"]) for t in valid_tickers}
    raw_aic  = {t: results[t]["aic"]                for t in valid_tickers}
    norm_aic = {
        t: {d: raw_aic[t][d] / n_events[t] for d in dist_names}
        for t in valid_tickers
    }
    winners_raw  = {t: min(raw_aic[t],  key=raw_aic[t].get)  for t in valid_tickers}
    winners_norm = {t: min(norm_aic[t], key=norm_aic[t].get) for t in valid_tickers}

    # ── AIC/N table ───────────────────────────────────────────────────────
    ntbl = Table(
        title="AIC/N  —  per-observation AIC, cross-ticker comparable, lower is better",
        box=box.SIMPLE_HEAD, header_style="bold magenta", title_justify="left",
    )
    ntbl.add_column("Ticker", style="bold")
    ntbl.add_column("N", justify="right")
    for d in dist_names:
        ntbl.add_column(d[:11], justify="right")
    ntbl.add_column("Winner", style="bold green", justify="right")
    for t in valid_tickers:
        best_d = winners_norm[t]
        cells  = [t, f"{n_events[t]:,}"]
        for d in dist_names:
            v    = norm_aic[t][d]
            star = " [bold green]★[/bold green]" if d == best_d else ""
            cells.append(f"{v:.4f}{star}")
        cells.append(best_d)
        ntbl.add_row(*cells)
    _console.print(ntbl)

    # ── Plot setup ───────────────────────────────────────────────────────────
    n_t     = len(valid_tickers)
    n_d     = len(dist_names)
    x       = np.arange(n_t)
    bar_w   = 0.07
    offsets = np.linspace(-(n_d - 1) / 2, (n_d - 1) / 2, n_d) * bar_w

    fig, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(max(22, 5 * n_t), 6))
    fig.suptitle("AIC Supplementary Diagnostics", fontsize=13, fontweight="bold")

    def _draw_panel(ax, aic_dict, winners, ylabel, title):
        """Shared drawing logic for both panels."""
        finite_vals = [
            aic_dict[t][d]
            for t in valid_tickers for d in dist_names
            if np.isfinite(aic_dict[t][d])
        ]
        if not finite_vals:
            ax.text(0.5, 0.5, "No finite AIC values",
                    ha="center", va="center", transform=ax.transAxes)
            return

        y_min = min(finite_vals)
        y_max = max(finite_vals)
        pad   = (y_max - y_min) * 0.12 or abs(y_min) * 0.12 or 1.0
        # Reserve headroom above y_max for inf annotations and winner stars.
        ax_top  = y_max + pad * 5
        inf_bar = y_max + pad * 3   # height to draw inf bars at

        for i, dist in enumerate(dist_names):
            bar_vals = []
            for t in valid_tickers:
                v = aic_dict[t][dist]
                bar_vals.append(v if np.isfinite(v) else inf_bar)
            bars = ax.bar(x + offsets[i], bar_vals, width=bar_w,
                          label=dist, color=dcolors[dist],
                          alpha=0.85, edgecolor="white")
            # Annotate degenerate (inf) bars.
            for bar, t in zip(bars, valid_tickers):
                if not np.isfinite(aic_dict[t][dist]):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        inf_bar + pad * 0.15,
                        "\u221e",   # ∞
                        ha="center", va="bottom", fontsize=7,
                        color=dcolors[dist], fontweight="bold",
                    )

        # Mark winning distribution with a star below its bar.
        for j, t in enumerate(valid_tickers):
            w  = winners[t]
            wi = dist_names.index(w)
            wv = aic_dict[t][w]
            if np.isfinite(wv):
                ax.annotate(
                    "\u2605",   # ★
                    xy=(x[j] + offsets[wi], wv),
                    ha="center", va="top", fontsize=9,
                    color="black", fontweight="bold",
                )

        ax.set_ylim(y_min - pad, ax_top)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"{t}\n(N={n_events[t]:,})" for t in valid_tickers], fontsize=9
        )
        ax.set_xlabel("Ticker")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.4)
        ax.legend(title="Distribution", fontsize=8, ncol=2)

    _draw_panel(
        ax_raw, raw_aic, winners_raw,
        ylabel="AIC",
        title=(
            "Raw AIC per Ticker\n"
            "(lower is better  |  \u2605 = winner  |  \u221e = degenerate fit  |  "
            "not cross-ticker comparable)"
        ),
    )
    _draw_panel(
        ax_norm, norm_aic, winners_norm,
        ylabel="AIC / N",
        title=(
            "AIC/N — Per-observation AIC\n"
            "(lower is better  |  \u2605 = winner  |  cross-ticker comparable)"
        ),
    )

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "aic_raw_and_normalised.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _console.print(f"    [dim]→ {fname}[/dim]")

# ===========================================================================
# Cross-ticker comparison plot
# ===========================================================================

def plot_cross_ticker_stylised_comparison(results):
    if len(results) < 2:
        _console.print("  Skipping cross-ticker comparison — need at least 2 fitted tickers.")
        return

    dates      = sorted({res["date"] for res in results.values()})
    date_label = dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}"
    tickers    = list(results.keys())

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Cross-Ticker Stylised Facts Comparison ({date_label})",
                 fontsize=13, fontweight="bold")

    ccdf_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]
    acf_ax = axes[2, 1]

    for idx, ticker in enumerate(tickers[:5]):
        res   = results[ticker]
        color = COLORS.get(ticker, "steelblue")
        ax    = axes[ccdf_positions[idx]]

        ia             = res["inter_arrival"]
        xs_emp, ys_emp = _ccdf(ia)
        xw = np.logspace(np.log10(max(ia.min(), 1e-8)), np.log10(ia.max()), 400)

        ax.loglog(xs_emp, ys_emp, lw=2.0, color=color, alpha=0.9, label="Empirical")
        ax.loglog(xw, expon.sf(xw, loc=0, scale=res["exp_scale"]),
                  ls=":", lw=1.6, color="orange", alpha=0.85, label="Exponential")
        ax.loglog(xw, lognorm.sf(xw, s=res["lognorm_s"], loc=0,
                                 scale=res["lognorm_scale"]),
                  ls="--", lw=1.6, color="green", alpha=0.85, label="Lognormal")
        ygg = gengamma.sf(xw, a=res["gengamma_a"], c=res["gengamma_c"],
                          loc=0, scale=res["gengamma_scale"])
        ax.loglog(xw, np.where(ygg > 0, ygg, np.nan),
                  ls="-.", lw=1.6, color="purple", alpha=0.85, label="GenGamma")

        ax.set_xlim(left=1e-6, right=5e2)
        ax.set_ylim(bottom=1e-4, top=1.1e0)
        ax.set_title(ticker, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Inter-arrival time (s)")
        ax.set_ylabel("P(Dt > x)")
        ax.legend(fontsize=8)

    ci = None
    for ticker, res in results.items():
        acf = res["acf"]
        if len(acf) == 0:
            continue
        lags = np.arange(1, len(acf) + 1)
        n    = len(res["inter_arrival"])
        ci   = 1.96 / np.sqrt(n)
        acf_ax.plot(lags, acf, marker="o", ms=2.5, lw=1.2,
                    color=COLORS.get(ticker, None), label=ticker)
    acf_ax.axhline(0, color="black", lw=0.8)
    if ci is not None:
        acf_ax.axhline( ci, color="red", ls="--", lw=1.0, label="95% CI")
        acf_ax.axhline(-ci, color="red", ls="--", lw=1.0)
    acf_ax.set_xlabel("Lag")
    acf_ax.set_ylabel("Autocorrelation")
    acf_ax.set_title("Signed Trade-size ACF by Ticker", fontsize=11)
    acf_ax.legend(fontsize=9)

    for idx in range(len(tickers), 5):
        axes[ccdf_positions[idx]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(PLOTS_DIR, "stylised_facts_multi_dist_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _console.print(f"    [dim]→ {fname}[/dim]")


# ===========================================================================
# Main entry point
# ===========================================================================

def run_stylised_facts_multi_dist(tickers=None, start=START_DATE, end=END_DATE,
                                  data_path=DATA_PATH):
    """
    Two-level parallel pipeline.

    Level 1 (ThreadPoolExecutor, max_workers=MAX_TICKER_WORKERS):
      Tickers are loaded and fitted concurrently.  Threads handle I/O-bound
      loading cheaply (no pickling overhead) and schedule each ticker's pool.

    Level 2 (ProcessPoolExecutor per ticker, max_workers=N_WORKERS):
      Grid-level CPU work for Burr12, GenGamma, and Mittag-Leffler.  Each
      ticker owns its own pool for the duration of its fitting phase; workers
      are initialised once with that ticker's data via _pool_initializer so
      grid jobs carry only lightweight parameter tuples.

    No nesting: the thread pool does I/O and scheduling; the process pools do
    compute.  Total simultaneous worker processes <= MAX_TICKER_WORKERS * N_WORKERS.

    Coarse grids use scrambled Sobol sequences for better space-filling coverage
    than Cartesian products with fewer points.  Fine grids use a Cartesian
    neighbourhood around the coarse winner.

    NOTE: all distribution fitting is a multi-start LOCAL-SEARCH HEURISTIC.
    Global optima are NOT guaranteed.
    """
    t0 = time.perf_counter()
    if tickers is None:
        tickers = STOCKS

    _console.print(Panel(
        f"[bold]Tickers[/bold]         : {', '.join(tickers)}\n"
        f"[bold]Period[/bold]          : {start}  →  {end}\n"
        f"[bold]Ticker workers[/bold]  : {MAX_TICKER_WORKERS}  ·  "
        f"[bold]Grid workers/ticker[/bold] : {N_WORKERS}  ·  "
        f"[bold]Max processes[/bold] : {MAX_TICKER_WORKERS * N_WORKERS}\n"
        f"[bold]Numba[/bold]           : "
        f"{'[green]enabled[/green]' if _NUMBA else '[yellow]off[/yellow] (pip install numba)'}  ·  "
        f"[bold]Sobol seed[/bold] : {RANDOM_SEED}",
        title="[bold cyan]Stylised Facts  —  Multi-Distribution Inter-Arrival Fitting[/bold cyan]",
        border_style="cyan",
    ))

    progress = Progress(
        SpinnerColumn(spinner_name="dots"),
        TextColumn("[bold blue]{task.description:<28}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[status]}"),
        console=_console,
        refresh_per_second=10,
    )

    stage_tasks = {
        t: progress.add_task(t, total=6, status="waiting...", visible=True)
        for t in tickers
    }
    # Per-ticker grid sub-bar so concurrent tickers don't fight over one bar.
    grid_tasks = {
        t: progress.add_task(f"  grid {t}", total=1, status="", visible=False)
        for t in tickers
    }

    comparison_results = {}
    results_lock = threading.Lock()

    def _load_and_fit(ticker):
        """Thread worker: load, validate, spawn process pool, fit, collect result."""
        progress.update(stage_tasks[ticker], status="loading...")

        # ---- load ----
        try:
            loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
            daily  = loader.load()
            if not daily:
                progress.update(stage_tasks[ticker], status="[red]no data[/red]")
                return
            df_raw = daily[0]
            t0_    = df_raw["Time"].min() + 3600
            t1_    = df_raw["Time"].max() - 3600
            df     = df_raw[(df_raw["Time"] >= t0_) &
                            (df_raw["Time"] <= t1_)].copy()
        except Exception as exc:
            progress.update(stage_tasks[ticker],
                            status=f"[red]load error: {exc}[/red]")
            return

        # Pre-extract inter-arrivals to gate early before pool creation.
        mo = df[df["Type"] == 4]
        if len(mo) < 10:
            progress.update(stage_tasks[ticker],
                            status="[red]not enough MO[/red]")
            return
        ia_raw = np.diff(mo["Time"].values)
        ia     = ia_raw[ia_raw > 0].astype(np.float64)
        if len(ia) < 10:
            progress.update(stage_tasks[ticker],
                            status="[red]not enough arrivals[/red]")
            return

        progress.update(stage_tasks[ticker], status="fitting...")

        # ---- fit — dedicated pool initialised with this ticker's data ----
        try:
            with ProcessPoolExecutor(
                max_workers=N_WORKERS,
                initializer=_pool_initializer,
                initargs=(ia,),
            ) as pool:
                res = _compute_ticker(
                    df, ticker, pool,
                    progress=progress,
                    stage_task=stage_tasks[ticker],
                    grid_task=grid_tasks[ticker],
                )
            if res is not None:
                with results_lock:
                    comparison_results[ticker] = res
        except Exception as exc:
            progress.update(stage_tasks[ticker],
                            status=f"[red]ERROR: {exc}[/red]")

    # ---- concurrent ticker fan-out (non-nested: threads -> process pools) ----
    with progress:
        with ThreadPoolExecutor(max_workers=MAX_TICKER_WORKERS) as tex:
            futures = {tex.submit(_load_and_fit, t): t for t in tickers}
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as exc:
                    t = futures[fut]
                    _console.print(f"  [{t}] Unhandled error: {exc}")

    _console.rule("[dim]generating comparison plots[/dim]")
    plot_cross_ticker_stylised_comparison(comparison_results)
    plot_aic_comparison(comparison_results)
    plot_aic_raw_and_normalised(comparison_results)

    elapsed = time.perf_counter() - t0
    _console.print(Panel(
        f"[bold green]Done[/bold green] in [bold]{elapsed:.1f}s[/bold]  ·  "
        f"plots → [dim]{os.path.abspath(PLOTS_DIR)}[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    mp.freeze_support()   # required on Windows
    run_stylised_facts_multi_dist()
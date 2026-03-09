"""
Experiment 3 — Power-law vs Exponential Hawkes Kernel

Fits a Hawkes process with power-law kernel:
    φ(t) = c / (1 + t/τ)^η

and compares against the exponential kernel from main.py:
    φ(t) = α · exp(−β · t)

using AIC. Produces:
  • Per-stock intensity plots (power-law kernel)
  • Per-stock QQ residual diagnostics (both kernels side-by-side)
  • Cross-stock AIC comparison bar chart (ΔAIC)
  • Cross-stock parameter comparison
"""

import os
import sys
import time
import warnings
import math

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

# ---------------------------------------------------------------------------
# Numba acceleration with transparent fallback
# ---------------------------------------------------------------------------
try:
    from numba import njit
    HAVE_NUMBA = True
except ModuleNotFoundError:
    HAVE_NUMBA = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def _wrap(fn):
            return fn
        return _wrap

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main as main_module
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


def _savefig(fname, quiet=False):
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    _log(f"    [dim]→ {fname}[/dim]" if HAVE_RICH else f"Saved: {fname}", force=not quiet)


def _format_winner(aic_exp, aic_pl):
    return "Power-law" if aic_pl < aic_exp else "Exponential"


# =============================================================================
# POWER-LAW HAWKES — KERNEL HELPERS
# =============================================================================

def powerlaw_kernel(dt, c, tau, eta):
    """φ(t) = c / (1 + t/τ)^η"""
    return c / (1.0 + dt / tau) ** eta


def powerlaw_kernel_integral(dt, c, tau, eta):
    """Φ(t) = ∫₀ᵗ φ(s) ds = c·τ/(η−1) · [1 − (1 + t/τ)^(1−η)]"""
    return c * tau / (eta - 1.0) * (1.0 - (1.0 + dt / tau) ** (1.0 - eta))


def powerlaw_branching_ratio(c, tau, eta):
    """n = ∫₀^∞ φ(s) ds = c·τ/(η−1).  Requires η > 1."""
    return c * tau / (eta - 1.0)


def _truncation_horizon(tau, eta, tol=1e-6, max_H=600.0):
    """
    Consistent truncation horizon H.  Capped at max_H (default 600s)
    to keep the per-event lookback bounded during optimisation.
    """
    H = tau * (tol ** (1.0 / (1.0 - eta)) - 1.0)
    return min(H, max_H)


# =============================================================================
# REPARAMETERIZATION
# =============================================================================

# Maximum η: beyond ~8 the power-law kernel decays so fast it's
# indistinguishable from an exponential, making the comparison meaningless.
ETA_MAX = 8.0
_ETA_M1_CAP = ETA_MAX - 1.0  # = 7.0
DEFAULT_TAU_FLOOR = 1e-3



def _theta_to_params(theta):
    """Map unconstrained θ → physical (μ, c, τ, η, n).
    η = 1 + min(exp(θ₃), 7), so η ∈ (1, 8]."""
    log_mu, log_n, log_tau, log_eta_m1 = theta
    mu = math.exp(log_mu)
    n = math.exp(log_n)
    tau = math.exp(log_tau)
    eta_m1 = min(math.exp(log_eta_m1), _ETA_M1_CAP)
    eta = 1.0 + eta_m1
    c = n * eta_m1 / tau
    return mu, c, tau, eta, n


def _params_to_theta(mu, c, tau, eta):
    """Map physical parameters → unconstrained θ."""
    eta_m1 = min(eta - 1.0, _ETA_M1_CAP)
    n = c * tau / eta_m1 if eta_m1 > 0 else 1e-10
    return np.array([
        math.log(mu),
        math.log(max(n, 1e-10)),
        math.log(tau),
        math.log(max(eta_m1, 1e-10)),
    ])




def infer_time_resolution(T):
    """Infer the effective timestamp resolution from sorted event times."""
    T = np.asarray(T, dtype=np.float64)
    if T.size < 2:
        return 1.0
    diffs = np.diff(np.sort(T))
    pos = diffs[diffs > 1e-12]
    if pos.size == 0:
        return 1.0
    # For LOBSTER-style integer-second timestamps this will typically be 1.0.
    # Use a small quantile instead of the absolute min for robustness.
    return float(max(np.quantile(pos, 0.05), 1e-6))


def preprocess_event_times(T):
    """
    Stabilise event times for continuous-time Hawkes fitting.

    If the data contain tied timestamps, the continuous-time likelihood becomes
    pathological because the model can drive tau -> 0 and exploit dt = 0 pairs.
    We therefore de-tie each timestamp bucket with a tiny deterministic jitter
    that stays well below the observed clock resolution.
    """
    T = np.sort(np.asarray(T, dtype=np.float64))
    T = T[np.isfinite(T)]
    if T.size < 2:
        return T, 1.0, 0

    resolution = infer_time_resolution(T)
    unique_vals, counts = np.unique(T, return_counts=True)
    n_ties = int(np.sum(np.maximum(counts - 1, 0)))
    if n_ties == 0:
        return T, resolution, 0

    jitter_span = min(0.49 * resolution, 1e-3 * max(resolution, 1.0))
    T_adj = T.copy()
    start = 0
    while start < T_adj.size:
        end = start + 1
        while end < T_adj.size and T_adj[end] == T_adj[start]:
            end += 1
        m = end - start
        if m > 1:
            offsets = np.linspace(0.0, jitter_span, m, endpoint=False)
            T_adj[start:end] += offsets
        start = end
    return T_adj, resolution, n_ties

# =============================================================================
# NUMBA JIT HOT PATHS (used when numba is available)
# =============================================================================

@njit(cache=True)
def _nll_inner_jit(T, N, T_end, mu, c, tau, eta, H):
    """Core NLL via scalar loops — fast under Numba."""
    inv_tau = 1.0 / tau
    one_m_eta = 1.0 - eta
    eta_m1 = eta - 1.0
    c_tau_over_eta_m1 = c * tau / eta_m1

    log_lam_sum = 0.0
    lo = 0
    for i in range(N):
        ti = T[i]
        cutoff = ti - H
        while lo < i and T[lo] < cutoff:
            lo += 1
        lam_i = mu
        for j in range(lo, i):
            lam_i += c * (1.0 + (ti - T[j]) * inv_tau) ** (-eta)
        if lam_i <= 0.0:
            return -np.inf, 0.0
        log_lam_sum += math.log(lam_i)

    compensator = mu * (T_end - T[0])
    for i in range(N):
        rem = T_end - T[i]
        if rem > H:
            rem = H
        compensator += c_tau_over_eta_m1 * (1.0 - (1.0 + rem * inv_tau) ** one_m_eta)

    return log_lam_sum, compensator


@njit(cache=True)
def _residuals_pl_jit(T, n, mu, c, tau, eta, H):
    """Power-law compensator increments — fast under Numba."""
    inv_tau = 1.0 / tau
    one_m_eta = 1.0 - eta
    coeff = c * tau / (eta - 1.0)
    residuals = np.empty(n - 1)
    lo = 0  # forward-running lookback pointer
    for i in range(1, n):
        ti = T[i]
        ti_prev = T[i - 1]
        val = mu * (ti - ti_prev)
        # Advance lo forward: oldest ancestor within H of ti
        while lo < i and T[lo] < ti - H:
            lo += 1
        for j in range(lo, i):
            dt_new = min(ti - T[j], H)
            dt_old = ti_prev - T[j]
            phi_new = coeff * (1.0 - (1.0 + dt_new * inv_tau) ** one_m_eta)
            if dt_old <= 0.0:
                phi_old = 0.0
            else:
                dt_old = min(dt_old, H)
                phi_old = coeff * (1.0 - (1.0 + dt_old * inv_tau) ** one_m_eta)
            val += phi_new - phi_old
        residuals[i - 1] = val
    return residuals


@njit(cache=True)
def _residuals_exp_jit(T, n, mu, alpha, beta):
    """Exponential compensator increments — fast under Numba."""
    aob = alpha / beta
    Lambda_cum = np.zeros(n)
    A = 0.0
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            en = math.exp(-beta * dt)
            A *= en
            Lambda_cum[i] = Lambda_cum[i - 1] + mu * dt + aob * (1.0 - en) * A
        A += 1.0
    result = np.empty(n - 1)
    for i in range(n - 1):
        result[i] = Lambda_cum[i + 1] - Lambda_cum[i]
    return result


@njit(cache=True)
def _intensity_grid_jit(T, N, t_grid, n_grid, mu, c, tau, eta, H):
    """Evaluate λ(t) on a grid — fast under Numba."""
    inv_tau = 1.0 / tau
    intensities = np.empty(n_grid)
    ev_ptr = 0
    for j in range(n_grid):
        t = t_grid[j]
        while ev_ptr < N and T[ev_ptr] < t:
            ev_ptr += 1
        lam = mu
        for k in range(ev_ptr - 1, -1, -1):
            dt = t - T[k]
            if dt > H:
                break
            lam += c * (1.0 + dt * inv_tau) ** (-eta)
        intensities[j] = lam
    return intensities


# =============================================================================
# VECTORISED NUMPY FALLBACK (used when numba is NOT available)
# =============================================================================

def _nll_inner_numpy(T, N, T_end, mu, c, tau, eta, H):
    """Core NLL via vectorised NumPy — fast without Numba."""
    inv_tau = 1.0 / tau
    one_m_eta = 1.0 - eta
    eta_m1 = eta - 1.0

    log_lam_sum = 0.0
    for i in range(N):
        if i == 0:
            lam_i = mu
        else:
            lo = np.searchsorted(T, T[i] - H, side="left")
            if lo < i:
                dts = T[i] - T[lo:i]
                lam_i = mu + np.sum(c * (1.0 + dts * inv_tau) ** (-eta))
            else:
                lam_i = mu
        if lam_i <= 0.0:
            return -np.inf, 0.0
        log_lam_sum += math.log(lam_i)

    remaining = np.minimum(T_end - T, H)
    c_tau_over_eta_m1 = c * tau / eta_m1
    compensator = mu * (T_end - T[0]) + np.sum(
        c_tau_over_eta_m1 * (1.0 - (1.0 + remaining * inv_tau) ** one_m_eta)
    )
    return log_lam_sum, compensator


def _residuals_pl_numpy(T, n, mu, c, tau, eta, H):
    """Power-law compensator increments via vectorised NumPy."""
    inv_tau = 1.0 / tau
    one_m_eta = 1.0 - eta
    coeff = c * tau / (eta - 1.0)
    residuals = np.zeros(n - 1)
    for i in range(1, n):
        dt_base = T[i] - T[i - 1]
        lo = max(0, np.searchsorted(T[:i], T[i] - H))
        past = T[lo:i]
        if len(past) > 0:
            dt_new = np.minimum(T[i] - past, H)
            dt_old = T[i - 1] - past
            phi_new = coeff * (1.0 - (1.0 + dt_new * inv_tau) ** one_m_eta)
            phi_old = np.where(
                dt_old <= 0.0,
                0.0,
                coeff * (1.0 - (1.0 + np.minimum(dt_old, H) * inv_tau) ** one_m_eta),
            )
            residuals[i - 1] = mu * dt_base + np.sum(phi_new - phi_old)
        else:
            residuals[i - 1] = mu * dt_base
    return residuals


def _residuals_exp_numpy(T, n, mu, alpha, beta):
    """Exponential compensator increments via O(n) recursion."""
    aob = alpha / beta
    Lambda_cum = np.zeros(n)
    A = 0.0
    for i in range(n):
        if i > 0:
            dt = T[i] - T[i - 1]
            en = math.exp(-beta * dt)
            A *= en
            Lambda_cum[i] = Lambda_cum[i - 1] + mu * dt + aob * (1.0 - en) * A
        A += 1.0
    return np.diff(Lambda_cum)


def _intensity_grid_numpy(T, N, t_grid, n_grid, mu, c, tau, eta, H):
    """Evaluate λ(t) on a grid via vectorised NumPy."""
    inv_tau = 1.0 / tau
    intensities = np.full(n_grid, mu)
    for j in range(n_grid):
        idx_end = np.searchsorted(T, t_grid[j], side="left")
        if idx_end > 0:
            lo = np.searchsorted(T[:idx_end], t_grid[j] - H)
            dts = t_grid[j] - T[lo:idx_end]
            if len(dts) > 0:
                intensities[j] = mu + np.sum(c * (1.0 + dts * inv_tau) ** (-eta))
    return intensities


# =============================================================================
# DISPATCH — pick JIT or NumPy path at call time
# =============================================================================

def _nll_inner(T, N, T_end, mu, c, tau, eta, H):
    if HAVE_NUMBA:
        return _nll_inner_jit(T, N, T_end, mu, c, tau, eta, H)
    return _nll_inner_numpy(T, N, T_end, mu, c, tau, eta, H)


def _residuals_pl(T, n, mu, c, tau, eta, H):
    if HAVE_NUMBA:
        return _residuals_pl_jit(T, n, mu, c, tau, eta, H)
    return _residuals_pl_numpy(T, n, mu, c, tau, eta, H)


def _residuals_exp(T, n, mu, alpha, beta):
    if HAVE_NUMBA:
        return _residuals_exp_jit(T, n, mu, alpha, beta)
    return _residuals_exp_numpy(T, n, mu, alpha, beta)


def _intensity_grid(T, N, t_grid, n_grid, mu, c, tau, eta, H):
    if HAVE_NUMBA:
        return _intensity_grid_jit(T, N, t_grid, n_grid, mu, c, tau, eta, H)
    return _intensity_grid_numpy(T, N, t_grid, n_grid, mu, c, tau, eta, H)


# =============================================================================
# NUMBA WARM-UP
# =============================================================================

def _warmup_jit():
    if not HAVE_NUMBA:
        return
    _t = np.array([0.0, 1.0, 2.0])
    try:
        _nll_inner_jit(_t, 3, 2.0, 0.1, 0.5, 0.1, 2.0, 10.0)
        _residuals_pl_jit(_t, 3, 0.1, 0.5, 0.1, 2.0, 10.0)
        _residuals_exp_jit(_t, 3, 0.1, 0.5, 1.0)
        _g = np.array([0.0, 1.0, 2.0])
        _intensity_grid_jit(_t, 3, _g, 3, 0.1, 0.5, 0.1, 2.0, 10.0)
    except Exception:
        pass

_warmup_jit()


# =============================================================================
# LIKELIHOOD WRAPPER
# =============================================================================

def hawkes_nll_powerlaw_theta(theta, T, T_end, tau_floor=DEFAULT_TAU_FLOOR):
    """Negative log-likelihood for (consistently truncated) power-law Hawkes."""
    mu, c, tau, eta, n = _theta_to_params(theta)

    # Continuous-time Hawkes fit is not identifiable below the timestamp resolution.
    if tau < tau_floor:
        return 1e18

    # Hard stationarity bound: n must be < 1
    if n >= 1.0 or n <= 0:
        return 1e18
    # Soft barrier near n=1 to keep optimizer away from the boundary
    # while still allowing high branching ratios (e.g. n=0.95)
    barrier = 0.0
    if n > 0.95:
        barrier = -10.0 * math.log(1.0 - n)
    if eta > ETA_MAX - 0.25:
        barrier += 25.0 * ((eta - (ETA_MAX - 0.25)) / 0.25) ** 2

    N = len(T)
    H = _truncation_horizon(tau, eta, tol=1e-6)
    H = min(H, T_end - T[0])

    log_lam_sum, compensator = _nll_inner(T, N, T_end, mu, c, tau, eta, H)

    if log_lam_sum == -np.inf:
        return 1e18

    nll = -(log_lam_sum - compensator) + barrier
    if not np.isfinite(nll):
        return 1e18
    return nll


# =============================================================================
# SOBOL + WARM-START INITIALISATION
# =============================================================================

def _generate_inits(T, exp_params=None, tau_floor=DEFAULT_TAU_FLOOR):
    """
    Generate ~28 diverse initial θ-vectors using Sobol low-discrepancy
    sampling (16 pts) + exponential warm-starts (8 pts) + random (4 pts).
    """
    mean_rate = len(T) / (T[-1] - T[0])
    mean_ia = np.mean(np.diff(T))

    tau_mult_floor = max(1.15, tau_floor / max(mean_ia, 1e-9))
    lo_bounds = np.array([0.10, 0.10, tau_mult_floor, 0.3])
    hi_bounds = np.array([0.85, 0.90, 5.00, 20.0])

    sobol = Sobol(d=4, scramble=True, seed=42)
    unit_pts = sobol.random(16)
    phys_pts = lo_bounds + unit_pts * (hi_bounds - lo_bounds)

    inits = []
    for row in phys_pts:
        mu_frac, n_target, eta_val, tau_mult = row
        mu = mean_rate * mu_frac
        tau = mean_ia * tau_mult
        c = n_target * (eta_val - 1.0) / tau
        if c > 0 and mu > 0:
            inits.append(_params_to_theta(mu, c, tau, eta_val))

    # Warm-starts from exponential fit
    if exp_params is not None:
        mu_e, alpha_e, beta_e = exp_params
        n_e = float(np.clip(alpha_e / beta_e, 0.05, 0.95)) if beta_e > 0 else 0.5
        tau_e = 1.0 / beta_e if beta_e > 0 else mean_ia
        for eta_val in [1.3, 1.8, 2.5, 4.0]:
            c_w = n_e * (eta_val - 1.0) / tau_e
            inits.append(_params_to_theta(mu_e, c_w, tau_e, eta_val))
            c_alt = n_e * (eta_val - 1.0) / (tau_e * 5.0)
            inits.append(_params_to_theta(mu_e, c_alt, tau_e * 5.0, eta_val))

    # Random perturbations
    rng = np.random.default_rng(42)
    base = np.array(inits)
    for _ in range(4):
        idx = rng.integers(0, len(base))
        inits.append(base[idx] + rng.normal(0, 0.4, size=4))

    return inits


# =============================================================================
# FITTING — COARSE-TO-FINE MULTI-START
# =============================================================================

def fit_hawkes_powerlaw(T, label="", quiet=False, exp_params=None, tau_floor=DEFAULT_TAU_FLOOR):
    """
    Fit Hawkes process with power-law kernel.

    Coarse-to-fine:
      1. Screen ~28 inits with short L-BFGS-B (≤50 iters)
      2. Polish top 3 with long L-BFGS-B (≤500 iters)
      3. Final Nelder-Mead refinement on the best

    Returns ((mu, c, tau, eta), nll) or (None, inf) if fitting fails.
    """
    T = np.ascontiguousarray(T, dtype=np.float64)
    T = T[np.isfinite(T)]
    if len(T) < 20:
        _log(
            f"  [yellow]⚠[/yellow] Not enough events to fit power-law Hawkes ({label})."
            if HAVE_RICH else
            f"  Not enough events to fit power-law Hawkes ({label}).",
            force=not quiet,
        )
        return None, np.inf

    T = T - T[0]
    T_end = float(T[-1])
    inits = _generate_inits(T, exp_params=exp_params, tau_floor=tau_floor)

    # --- Phase 1: coarse screen ---
    coarse = []
    n_tried = 0
    n_infeasible = 0
    for theta0 in inits:
        n_tried += 1
        try:
            res = minimize(
                hawkes_nll_powerlaw_theta, theta0, args=(T, T_end, tau_floor),
                method="L-BFGS-B",
                options={"ftol": 1e-10, "gtol": 1e-6, "maxiter": 50},
            )
            _, _, _, _, n_chk = _theta_to_params(res.x)
            if np.isfinite(res.fun) and 0 < n_chk < 1.0:
                coarse.append((res.fun, res.x))
            else:
                n_infeasible += 1
        except Exception:
            continue

    # Fallback: Nelder-Mead
    if not coarse:
        for theta0 in inits[:10]:
            try:
                res = minimize(
                    hawkes_nll_powerlaw_theta, theta0, args=(T, T_end, tau_floor),
                    method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-8, "fatol": 1e-8},
                )
                _, _, _, _, n_chk = _theta_to_params(res.x)
                if np.isfinite(res.fun) and 0 < n_chk < 1.0:
                    coarse.append((res.fun, res.x))
            except Exception:
                continue

    if not coarse:
        _log(
            f"  [yellow]⚠[/yellow] Power-law fitting failed for {label} "
            f"({n_tried} inits tried, {n_infeasible} hit boundary)."
            if HAVE_RICH else
            f"  Power-law fitting failed for {label} "
            f"({n_tried} inits, {n_infeasible} hit boundary).",
            force=not quiet,
        )
        return None, np.inf

    # --- Phase 2: polish top 3 ---
    coarse.sort(key=lambda x: x[0])
    best_val = np.inf
    best_theta = None

    for _, theta_start in coarse[:3]:
        try:
            res = minimize(
                hawkes_nll_powerlaw_theta, theta_start, args=(T, T_end, tau_floor),
                method="L-BFGS-B",
                options={"ftol": 1e-14, "gtol": 1e-9, "maxiter": 500},
            )
            _, _, _, _, n_chk = _theta_to_params(res.x)
            if res.fun < best_val and 0 < n_chk < 1.0:
                best_val = res.fun
                best_theta = res.x
        except Exception:
            continue

    # --- Phase 3: Nelder-Mead refinement ---
    if best_theta is not None:
        try:
            res = minimize(
                hawkes_nll_powerlaw_theta, best_theta, args=(T, T_end, tau_floor),
                method="Nelder-Mead",
                options={"maxiter": 2000, "xatol": 1e-12, "fatol": 1e-12},
            )
            _, _, _, _, n_chk = _theta_to_params(res.x)
            if res.fun < best_val and 0 < n_chk < 1.0:
                best_val = res.fun
                best_theta = res.x
        except Exception:
            pass

    if best_theta is None:
        _log(
            f"  [yellow]⚠[/yellow] Power-law fitting failed completely for {label}."
            if HAVE_RICH else
            f"  Power-law fitting failed completely for {label}.",
            force=not quiet,
        )
        return None, np.inf

    mu, c, tau, eta, n_br = _theta_to_params(best_theta)
    nll = float(best_val)

    if not quiet:
        if HAVE_RICH and console is not None:
            tbl = Table(title=f"Power-law Hawkes fit — {label}", box=box.SIMPLE_HEAVY)
            tbl.add_column("Parameter", style="cyan")
            tbl.add_column("Value", justify="right")
            tbl.add_row("μ (baseline)", f"{mu:.5f} events/sec")
            tbl.add_row("c (kernel scale)", f"{c:.5f}")
            tbl.add_row("τ (time scale)", f"{tau:.5f} sec")
            tbl.add_row("η (decay exponent)", f"{eta:.4f}")
            tbl.add_row("Branching ratio", f"{n_br:.4f}")
            console.print(tbl)
            if n_br >= 1:
                console.print("  [yellow]⚠[/yellow] Branching ratio >= 1 -> non-stationary.")
            else:
                console.print(f"  [green]↳[/green] ~{n_br * 100:.1f}% triggered.")
        else:
            _log(f"\n{'─' * 55}", force=True)
            _log(f"  Power-law Hawkes fit — {label}", force=True)
            _log(f"  μ = {mu:.5f}  c = {c:.5f}  τ = {tau:.5f}  η = {eta:.4f}  n = {n_br:.4f}", force=True)
            _log(f"{'─' * 55}\n", force=True)

    return (mu, c, tau, eta), nll


# =============================================================================
# DIAGNOSTICS — INTENSITY PLOT
# =============================================================================

def plot_powerlaw_intensity(T, mu, c, tau, eta, ticker, n_grid=2000, quiet=False):
    T = np.ascontiguousarray(T, dtype=np.float64)
    t_grid = np.linspace(T[0], T[-1], n_grid)
    H = _truncation_horizon(tau, eta, tol=1e-6)

    intensities = _intensity_grid(T, len(T), t_grid, n_grid, mu, c, tau, eta, H)
    finite = intensities[np.isfinite(intensities)]
    y_cap = np.quantile(finite, 0.995) if finite.size else mu
    y_cap = max(y_cap, mu * 2.0, 1e-12)
    n_clipped = int(np.sum(intensities > y_cap))
    intensities_plot = np.minimum(intensities, y_cap)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), height_ratios=[3, 1])
    fig.suptitle(f"{ticker} — Power-law Hawkes Intensity", fontweight="bold")
    ax1.plot(t_grid, intensities_plot, color=COLORS.get(ticker, "steelblue"),
             lw=1.0, alpha=0.9, label="λ(t-) power-law")
    ax1.axhline(mu, color="red", ls="--", lw=1, alpha=0.7, label=f"μ = {mu:.4f}")
    ax1.set_ylabel("Intensity λ(t)")
    ax1.set_title("Conditional Intensity (left-limit, robust y-scale)")
    if n_clipped > 0:
        ax1.text(0.01, 0.98, f"Clipped {n_clipped} / {len(intensities)} grid points above 99.5% quantile",
                 transform=ax1.transAxes, va="top", ha="left", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85))
    ax1.legend(fontsize=9)
    ax2.eventplot([T], lineoffsets=0, linelengths=1,
                  color=COLORS.get(ticker, "steelblue"), alpha=0.3)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Events")
    ax2.set_yticks([])
    plt.tight_layout()
    _savefig(os.path.join(PLOTS_DIR, f"powerlaw_intensity_{ticker}.png"), quiet=quiet)


# =============================================================================
# DIAGNOSTICS — QQ PLOT
# =============================================================================

def _compute_residuals_exponential(T, mu, alpha, beta):
    T = np.ascontiguousarray(T, dtype=np.float64)
    return _residuals_exp(T, len(T), mu, alpha, beta)


def _compute_residuals_powerlaw(T, mu, c, tau, eta):
    T = np.ascontiguousarray(T, dtype=np.float64)
    H = _truncation_horizon(tau, eta, tol=1e-6)
    return _residuals_pl(T, len(T), mu, c, tau, eta, H)


def plot_qq_comparison(T_raw, exp_params, pl_params, ticker, quiet=False):
    T_zeroed = T_raw - T_raw[0]
    mu_e, alpha_e, beta_e = exp_params
    mu_p, c_p, tau_p, eta_p = pl_params

    resid_exp = _compute_residuals_exponential(T_zeroed, mu_e, alpha_e, beta_e)
    resid_pl = _compute_residuals_powerlaw(T_zeroed, mu_p, c_p, tau_p, eta_p)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(f"{ticker} — Residual QQ Plots (Exp vs Power-law)", fontweight="bold")

    for ax, resid, label, color in [
        (axes[0], resid_exp, "Exponential kernel", "#2ca02c"),
        (axes[1], resid_pl, "Power-law kernel", "#d62728"),
    ]:
        resid = resid[resid > 0]
        n_pts = len(resid)
        if n_pts < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            ax.set_title(label)
            continue
        quantiles_emp = np.sort(resid)
        probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts
        quantiles_th = -np.log(1.0 - probs)

        ax.plot(quantiles_th, quantiles_emp, ".", alpha=0.4, color=color, ms=3)
        lim = max(quantiles_th.max(), quantiles_emp.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=1.5, alpha=0.6, label="Perfect fit")
        ax.set_xlabel("Theoretical Exp(1) quantiles")
        ax.set_ylabel("Empirical quantiles")
        ax.set_title(label)

        # KS statistic: max |F_emp(x) - F_theo(x)| using the sorted residuals
        ecdf = np.arange(1, n_pts + 1) / n_pts
        theo_cdf = 1.0 - np.exp(-quantiles_emp)
        ks_stat = np.max(np.abs(ecdf - theo_cdf))

        ax.text(0.05, 0.92, f"n = {n_pts}\nKS = {ks_stat:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.legend(fontsize=9)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

    plt.tight_layout()
    _savefig(os.path.join(PLOTS_DIR, f"qq_comparison_{ticker}.png"), quiet=quiet)


# =============================================================================
# AIC
# =============================================================================

def compute_aic(neg_loglik, k):
    return 2 * k + 2 * neg_loglik


def plot_aic_comparison(results, quiet=False):
    tickers = list(results.keys())
    aic_exp = [results[t]["aic_exp"] for t in tickers]
    aic_pl = [results[t]["aic_pl"] for t in tickers]

    delta_exp, delta_pl, winners = [], [], []
    for i in range(len(tickers)):
        best = min(aic_exp[i], aic_pl[i])
        delta_exp.append(aic_exp[i] - best)
        delta_pl.append(aic_pl[i] - best)
        winners.append(_format_winner(aic_exp[i], aic_pl[i]))

    x = np.arange(len(tickers))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.bar(x - w / 2, delta_exp, w, label="Exponential (k=3)", color="#2ca02c", alpha=0.85, edgecolor="white")
    ax.bar(x + w / 2, delta_pl, w, label="Power-law (k=4)", color="#d62728", alpha=0.85, edgecolor="white")
    for i in range(len(tickers)):
        ax.text(x[i], max(delta_exp[i], delta_pl[i]) * 1.05 + 0.5,
                f"Winner: {winners[i]}", ha="center", va="bottom", fontsize=8,
                fontweight="bold", color="#333333")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("ΔAIC (lower is better)")
    ax.set_title("Hawkes Kernel Comparison — ΔAIC per Ticker\n(Exponential vs Power-law)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend(fontsize=10)
    plt.tight_layout()
    _savefig(os.path.join(PLOTS_DIR, "aic_kernel_comparison.png"), quiet=quiet)

    if quiet:
        return
    if HAVE_RICH and console is not None:
        tbl = Table(title="AIC Kernel Comparison Summary", box=box.SIMPLE_HEAVY)
        tbl.add_column("Ticker", style="cyan")
        tbl.add_column("AIC (Exp)", justify="right")
        tbl.add_column("AIC (PL)", justify="right")
        tbl.add_column("ΔAIC (Exp)", justify="right")
        tbl.add_column("ΔAIC (PL)", justify="right")
        tbl.add_column("Winner", justify="right")
        for i, t in enumerate(tickers):
            tbl.add_row(t, f"{aic_exp[i]:.2f}", f"{aic_pl[i]:.2f}",
                        f"{delta_exp[i]:.2f}", f"{delta_pl[i]:.2f}", winners[i])
        console.print(tbl)
    else:
        _log(f"\n{'=' * 70}", force=True)
        _log("  AIC Kernel Comparison Summary", force=True)
        _log(f"{'=' * 70}", force=True)
        _log(f"  {'Ticker':<8} {'AIC(Exp)':>14} {'AIC(PL)':>14} {'ΔAIC(Exp)':>12} {'ΔAIC(PL)':>12} {'Winner':>12}", force=True)
        _log(f"  {'─' * 66}", force=True)
        for i, t in enumerate(tickers):
            _log(f"  {t:<8} {aic_exp[i]:>14.2f} {aic_pl[i]:>14.2f} {delta_exp[i]:>12.2f} {delta_pl[i]:>12.2f} {winners[i]:>12}", force=True)
        _log(f"{'=' * 70}\n", force=True)


# =============================================================================
# CROSS-STOCK PARAMETER COMPARISON
# =============================================================================

def plot_parameter_comparison(results, quiet=False):
    tickers = [t for t in results if results[t]["pl_params"] is not None]
    if len(tickers) < 2:
        _log("  [yellow]⚠[/yellow] Not enough stocks for parameter comparison." if HAVE_RICH else "  Not enough stocks.", force=not quiet)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Experiment 3 — Cross-stock Parameter Comparison", fontweight="bold", fontsize=13)

    br_exp = [results[t]["exp_params"][1] / results[t]["exp_params"][2] for t in tickers]
    br_pl = [powerlaw_branching_ratio(*results[t]["pl_params"][1:]) for t in tickers]
    x = np.arange(len(tickers))
    w = 0.35

    ax = axes[0, 0]
    ax.bar(x - w/2, br_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
    ax.bar(x + w/2, br_pl, w, label="Power-law", color="#d62728", alpha=0.85)
    ax.axhline(1, color="black", ls="--", lw=1, alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("Branching ratio"); ax.set_title("Branching Ratio"); ax.legend(fontsize=9)

    ax = axes[0, 1]
    mu_exp = [results[t]["exp_params"][0] for t in tickers]
    mu_pl = [results[t]["pl_params"][0] for t in tickers]
    ax.bar(x - w/2, mu_exp, w, label="Exponential", color="#2ca02c", alpha=0.85)
    ax.bar(x + w/2, mu_pl, w, label="Power-law", color="#d62728", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(tickers)
    ax.set_ylabel("μ (events/sec)"); ax.set_title("Baseline Rate μ"); ax.legend(fontsize=9)

    ax = axes[1, 0]
    etas = [results[t]["pl_params"][3] for t in tickers]
    colors = [COLORS.get(t, "grey") for t in tickers]
    ax.bar(tickers, etas, color=colors, alpha=0.85)
    ax.axhline(2, color="grey", ls=":", lw=1, alpha=0.7, label="η=2")
    ax.set_ylabel("η"); ax.set_title("Power-law Decay Exponent η"); ax.legend(fontsize=9)
    for i, e in enumerate(etas):
        ax.text(i, e + 0.02, f"{e:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    taus = [results[t]["pl_params"][2] for t in tickers]
    ax.bar(tickers, taus, color=colors, alpha=0.85)
    ax.set_ylabel("τ (seconds)"); ax.set_title("Power-law Time Scale τ")
    for i, v in enumerate(taus):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _savefig(os.path.join(PLOTS_DIR, "parameter_comparison.png"), quiet=quiet)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_experiment3(tickers=None, start=START_DATE, end=END_DATE, data_path=DATA_PATH):
    t0 = time.perf_counter()
    if tickers is None:
        tickers = STOCKS
    results = {}
    fitted_tickers = []

    if HAVE_RICH and console is not None:
        jit_tag = "[green]Numba JIT[/green]" if HAVE_NUMBA else "[yellow]NumPy fallback[/yellow]"
        console.print(Panel(
            f"[bold]Tickers[/bold] : {', '.join(tickers)}\n"
            f"[bold]Period[/bold]  : {start} -> {end}\n"
            f"[bold]Backend[/bold] : {jit_tag}",
            title="[bold cyan]Experiment 3 - Power-law vs Exponential[/bold cyan]",
            border_style="cyan",
        ))

    def _run_one_ticker(ticker, progress=None, stage_task=None):
        stage_names = ["loaded", "filtered MOs", "fit exponential",
                       "fit power-law", "AIC computed", "intensity plot",
                       "QQ plot", "done"]

        def _advance(ix):
            if progress is not None and stage_task is not None:
                progress.advance(stage_task)
                progress.update(stage_task, status=stage_names[ix])

        t_stock = time.perf_counter()
        loader = Loader(ticker, start, end, dataPath=data_path, nlevels=10)
        daily = loader.load()
        if not daily:
            _log(f"  [yellow]⚠[/yellow] No data for {ticker}." if HAVE_RICH else f"  No data for {ticker}.", force=True)
            return

        df = daily[0]
        t_lo = df["Time"].min() + 3600
        t_hi = df["Time"].max() - 3600
        df = df[(df["Time"] >= t_lo) & (df["Time"] <= t_hi)].copy()
        _advance(0)

        mo = df[df["Type"] == 4]
        T = np.sort(mo["Time"].values.astype(np.float64))
        T = T[np.isfinite(T)]
        _advance(1)

        if len(T) < 20:
            _log(f"  [yellow]⚠[/yellow] Too few MOs for {ticker} (n={len(T)})." if HAVE_RICH else f"  Too few MOs for {ticker}.", force=True)
            return

        T_model, time_resolution, n_ties = preprocess_event_times(T)
        if n_ties > 0:
            _log(f"  [yellow]ℹ[/yellow] {ticker}: de-tied {n_ties} duplicated timestamps at ~{time_resolution:g}s resolution." if HAVE_RICH else f"  {ticker}: de-tied {n_ties} duplicated timestamps at ~{time_resolution:g}s resolution.", force=True)
        T_zeroed = T_model - T_model[0]
        exp_params = fit_hawkes(T_model, label=f"{ticker} exponential", quiet=True)
        _advance(2)
        if exp_params is None:
            _log(f"  [yellow]⚠[/yellow] Exp fit failed for {ticker}." if HAVE_RICH else f"  Exp fit failed for {ticker}.", force=True)
            return

        mu_e, alpha_e, beta_e = exp_params
        nll_exp = float(hawkes_loglik([mu_e, alpha_e, beta_e], T_zeroed))

        tau_floor = max(0.5 * time_resolution, 1e-3)
        pl_params, nll_pl = fit_hawkes_powerlaw(T_model, label=f"{ticker} power-law", quiet=True, exp_params=exp_params, tau_floor=tau_floor)
        _advance(3)
        if pl_params is None:
            _log(f"  [yellow]⚠[/yellow] PL fit failed for {ticker}." if HAVE_RICH else f"  PL fit failed for {ticker}.", force=True)
            return

        aic_exp = float(compute_aic(nll_exp, k=3))
        aic_pl = float(compute_aic(nll_pl, k=4))
        winner = _format_winner(aic_exp, aic_pl)
        delta_aic = abs(aic_exp - aic_pl)
        _advance(4)

        results[ticker] = {
            "exp_params": exp_params, "pl_params": pl_params,
            "nll_exp": nll_exp, "nll_pl": nll_pl,
            "aic_exp": aic_exp, "aic_pl": aic_pl, "T": T_model, "raw_T": T, "time_resolution": time_resolution,
        }
        fitted_tickers.append(ticker)

        mu_p, c_p, tau_p, eta_p = pl_params
        plot_powerlaw_intensity(T_model, mu_p, c_p, tau_p, eta_p, ticker, quiet=True)
        _advance(5)
        plot_qq_comparison(T_model, exp_params, pl_params, ticker, quiet=True)
        _advance(6)

        br_exp = alpha_e / beta_e if beta_e > 0 else np.nan
        br_pl = powerlaw_branching_ratio(c_p, tau_p, eta_p)
        _log(
            f"  [bold]{ticker}[/bold]  winner={winner}  ΔAIC={delta_aic:.2f}  "
            f"BR_exp={br_exp:.3f}  BR_pl={br_pl:.3f}  η={eta_p:.3f}  "
            f"({time.perf_counter() - t_stock:.1f}s)"
            if HAVE_RICH else
            f"  {ticker}  winner={winner}  ΔAIC={delta_aic:.2f}  "
            f"BR_exp={br_exp:.3f}  BR_pl={br_pl:.3f}  η={eta_p:.3f}  "
            f"({time.perf_counter() - t_stock:.1f}s)",
            force=True,
        )
        _advance(7)

    if HAVE_RICH and console is not None:
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
                console=console, refresh_per_second=10,
            ) as progress:
                outer = progress.add_task("tickers", total=len(tickers), status="")
                for ticker in tickers:
                    stask = progress.add_task(f"[cyan]{ticker}[/cyan]", total=8, status="starting")
                    _run_one_ticker(ticker, progress=progress, stage_task=stask)
                    progress.advance(outer)
        finally:
            set_quiet_logs(False)
            if hasattr(main_module, "set_quiet_logs"):
                main_module.set_quiet_logs(False)
    else:
        for ticker in tickers:
            _log("\n" + "=" * 60, force=True)
            _log(f"  Experiment 3 [{ticker}]", force=True)
            _log("=" * 60, force=True)
            _run_one_ticker(ticker)

    if len(fitted_tickers) > 1:
        _log("\n[bold cyan]Cross-stock comparison[/bold cyan]" if HAVE_RICH else "\nCross-stock comparison", force=True)
        plot_aic_comparison(results, quiet=False)
        plot_parameter_comparison(results, quiet=False)
    elif len(fitted_tickers) == 1:
        _log("\n  Only one stock fitted — skipping cross-stock plots.", force=True)

    elapsed = time.perf_counter() - t0
    if HAVE_RICH and console is not None:
        console.print(Panel(
            f"[bold green]Done[/bold green] in [bold]{elapsed:.1f}s[/bold] · plots -> [dim]{os.path.abspath(PLOTS_DIR)}[/dim]",
            border_style="green",
        ))
    else:
        _log(f"\nExperiment 3 complete ({elapsed:.1f}s).  Plots in: {os.path.abspath(PLOTS_DIR)}", force=True)
    return results


if __name__ == "__main__":
    run_experiment3()
import os
import warnings
import hashlib
from datetime import datetime

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# CONFIG
# =============================================================================
STOCKS = ["AMZN", "AAPL", "GOOG", "MSFT", "INTC"]
COLORS = {
    "AMZN": "#FF9900",
    "AAPL": "#555555",
    "GOOG": "#4285F4",
    "MSFT": "#00A4EF",
    "INTC": "#CC0000",
}

DATA_PATH = "data/"
START_DATE = "2012-06-21"
END_DATE = "2012-06-21"
PLOT_DIR = "plot_powerlaw_2d"
os.makedirs(PLOT_DIR, exist_ok=True)

# Robust-selection defaults (adapted from the original 2D file)
HEALTHY_RHO_MAX = 0.95
HEALTHY_GRAD_MAX = 2.5e5
HEALTHY_MU_MIN = 1e-4
HEALTHY_MU_MAX = 1.0
HEALTHY_TAU_MIN = 1e-4
HEALTHY_TAU_MAX = 2e4
HEALTHY_ETA_MIN = 1.01
HEALTHY_ETA_MAX = 25.0
RAW_STABILITY_RHO_MAX = 0.995
PENALTY_WEIGHT = 5e5

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


# =============================================================================
# FILE HELPERS
# =============================================================================

def _stable_seed_from_ticker(ticker, base_seed=42):
    h = hashlib.md5(f"{ticker}_{base_seed}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


# =============================================================================
# LOBSTER LOADER
# =============================================================================
class Loader:
    def __init__(self, ric, sDate, eDate, **kwargs):
        self.ric = ric.split(".")[0]
        self.sDate = sDate
        self.eDate = eDate
        self.nlevels = kwargs.get("nlevels", 10)
        self.dataPath = kwargs.get("dataPath", DATA_PATH)

    def _col_names(self):
        sides = ["Ask Price", "Ask Size", "Bid Price", "Bid Size"]
        all_names, keep = [], []
        for lvl in range(1, 11):
            for s in sides:
                col = f"{s} {lvl}"
                all_names.append(col)
                if lvl <= self.nlevels:
                    keep.append(col)
        return all_names, keep

    def _find_file(self, date_str, kind):
        for levels_tag in ["10", "5"]:
            fname = f"{self.ric}_{date_str}_34200000_57600000_{kind}_{levels_tag}.csv"
            full = os.path.join(self.dataPath, fname)
            if os.path.exists(full):
                return full
        return None

    def load(self):
        data = []
        all_names, keep_cols = self._col_names()

        for d in pd.date_range(self.sDate, self.eDate, freq="B"):
            date_str = d.strftime("%Y-%m-%d")
            msg_path = self._find_file(date_str, "message")
            ob_path = self._find_file(date_str, "orderbook")

            if msg_path is None or ob_path is None:
                continue

            print(f"  Loading {self.ric}  {date_str} …")

            msg = pd.read_csv(
                msg_path,
                names=["Time", "Type", "OrderID", "Size", "Price", "TradeDirection", "tmp"],
            )

            t_open = 9.5 * 3600
            t_close = 16.0 * 3600
            msg = msg[(msg.Time >= t_open) & (msg.Time <= t_close)].copy()

            type6 = msg[msg.Type == 6]
            if type6.empty:
                t_start, t_end = t_open, t_close
            else:
                t_start = type6.iloc[0].Time
                t_end = type6.iloc[1].Time if len(type6) > 1 else t_close

            msg = msg[(msg.Time >= t_start) & (msg.Time <= t_end)].copy()

            ob = pd.read_csv(ob_path, names=all_names)[keep_cols]
            ob_idx = msg.index
            ob_aligned = ob.loc[ob_idx].copy()

            price_cols = [c for c in ob_aligned.columns if "Price" in c]
            ob_aligned[price_cols] = ob_aligned[price_cols] / 10_000.0
            msg["Price"] = msg["Price"] / 10_000.0

            combined = pd.concat(
                [msg.reset_index(drop=True), ob_aligned.reset_index(drop=True)], axis=1
            )
            combined["Date"] = date_str
            combined["Ticker"] = self.ric
            combined["TimeSinceOpen"] = combined["Time"] - t_start
            data.append(combined)

        if not data:
            print(f"  ⚠  No data found for {self.ric} between {self.sDate} and {self.eDate}.")
            print(f"     Expected files in: {os.path.abspath(self.dataPath)}")

        return data


# =============================================================================
# 2D DIRECTIONAL EVENTS
# =============================================================================

def extract_directional_events(df, use_time_since_open=False):
    time_col = "TimeSinceOpen" if use_time_since_open and "TimeSinceOpen" in df.columns else "Time"

    mo_buy = df[(df["Type"] == 4) & (df["TradeDirection"] == 1)][time_col].values
    mo_sell = df[(df["Type"] == 4) & (df["TradeDirection"] == -1)][time_col].values

    mo_buy = np.sort(np.asarray(mo_buy, dtype=float))
    mo_sell = np.sort(np.asarray(mo_sell, dtype=float))
    mo_buy = mo_buy[np.isfinite(mo_buy)]
    mo_sell = mo_sell[np.isfinite(mo_sell)]
    return mo_buy, mo_sell


# =============================================================================
# POWER-LAW 2D HAWKES CORE
# =============================================================================

def _stack_events(times_list):
    all_t, all_k = [], []
    for k, arr in enumerate(times_list):
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        for t in a:
            all_t.append(float(t))
            all_k.append(int(k))
    if not all_t:
        return np.array([]), np.array([], dtype=int)
    idx = np.argsort(all_t)
    return np.asarray(all_t)[idx], np.asarray(all_k, dtype=int)[idx]


def powerlaw_kernel(dt, c, tau, eta):
    return c / np.power(1.0 + dt / tau, eta)


def powerlaw_kernel_integral(dt, c, tau, eta):
    # Integral from 0 to dt of c / (1 + s/tau)^eta ds, valid for eta > 1.
    return c * tau / (eta - 1.0) * (1.0 - np.power(1.0 + dt / tau, 1.0 - eta))


def _branching_matrix(c, tau, eta):
    d = c.shape[0]
    A = np.zeros((d, d), dtype=float)
    for source in range(d):
        for target in range(d):
            A[target, source] = c[source, target] * tau[source, target] / (eta[source, target] - 1.0)
    return A


def _spectral_radius(M):
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def _pack_theta(mu, c, tau, eta):
    return np.concatenate([mu, c.reshape(-1), tau.reshape(-1), eta.reshape(-1)])


def _unpack_theta(params, d):
    params = np.asarray(params, dtype=float)
    n = d * d
    mu = params[:d]
    c = params[d:d + n].reshape(d, d)
    tau = params[d + n:d + 2 * n].reshape(d, d)
    eta = params[d + 2 * n:d + 3 * n].reshape(d, d)
    return mu, c, tau, eta


def _to_unconstrained(params, d=2):
    mu, c, tau, eta = _unpack_theta(params, d)
    mu = np.log(np.maximum(mu, 1e-12))
    c = np.log(np.maximum(c, 1e-12))
    tau = np.log(np.maximum(tau, 1e-12))
    eta_minus_1 = np.log(np.maximum(eta - 1.0, 1e-12))
    return _pack_theta(mu, c, tau, eta_minus_1)


def _from_unconstrained(theta, d=2):
    mu, c, tau, eta_minus_1 = _unpack_theta(theta, d)
    mu = np.exp(mu)
    c = np.exp(c)
    tau = np.exp(tau)
    eta = 1.0 + np.exp(eta_minus_1)
    return _pack_theta(mu, c, tau, eta)


def mv_hawkes_negloglik(params, times_list, d=2, stability_rho_max=0.999, cutoff_factor=1e8):
    params = np.asarray(params, dtype=float)
    mu, c, tau, eta = _unpack_theta(params, d)

    if np.any(mu <= 0) or np.any(c < 0) or np.any(tau <= 0) or np.any(eta <= 1.0):
        return np.inf

    A = _branching_matrix(c, tau, eta)
    rho = _spectral_radius(A)
    if (not np.isfinite(rho)) or rho >= stability_rho_max:
        return np.inf

    t_all, k_all = _stack_events(times_list)
    if len(t_all) < 10:
        return np.inf

    t0 = t_all[0]
    t_all = t_all - t0
    T_end = t_all[-1]

    shifted = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = np.sort(a[np.isfinite(a)])
        shifted.append(a - t0 if len(a) else np.array([]))

    ll = 0.0

    for i, (t, k) in enumerate(zip(t_all, k_all)):
        lam_k = mu[k]
        if i > 0:
            past_t = t_all[:i]
            past_k = k_all[:i]
            dts = t - past_t
            for source in range(d):
                mask = (past_k == source)
                if np.any(mask):
                    for target in range(d):
                        if target == k:
                            lam_k += np.sum(powerlaw_kernel(dts[mask], c[source, target], tau[source, target], eta[source, target]))

        if lam_k <= 0 or not np.isfinite(lam_k):
            return np.inf
        ll += np.log(lam_k)

    compensator = float(np.sum(mu) * T_end)
    for source in range(d):
        ts = shifted[source]
        if len(ts) == 0:
            continue
        delta = T_end - ts
        for target in range(d):
            compensator += np.sum(powerlaw_kernel_integral(delta, c[source, target], tau[source, target], eta[source, target]))

    val = -(ll - compensator)
    return float(val) if np.isfinite(val) else np.inf


def mv_hawkes_negloglik_and_grad(params, times_list, d=2, stability_rho_max=0.999):
    params = np.asarray(params, dtype=float)
    base = mv_hawkes_negloglik(params, times_list, d=d, stability_rho_max=stability_rho_max)
    grad = np.zeros_like(params)
    if not np.isfinite(base):
        return np.inf, grad

    for i in range(len(params)):
        h = 1e-4 * max(1.0, abs(params[i]))
        p_plus = params.copy()
        p_minus = params.copy()
        p_plus[i] += h
        p_minus[i] = max(p_minus[i] - h, 1e-12)
        v_plus = mv_hawkes_negloglik(p_plus, times_list, d=d, stability_rho_max=stability_rho_max)
        v_minus = mv_hawkes_negloglik(p_minus, times_list, d=d, stability_rho_max=stability_rho_max)
        if np.isfinite(v_plus) and np.isfinite(v_minus) and p_plus[i] != p_minus[i]:
            grad[i] = (v_plus - v_minus) / (p_plus[i] - p_minus[i])
        elif np.isfinite(v_plus):
            grad[i] = (v_plus - base) / h
        else:
            grad[i] = 0.0
    return base, grad


def _robust_penalty_from_params(params, d=2, rho_soft_max=HEALTHY_RHO_MAX, penalty_weight=PENALTY_WEIGHT):
    mu, c, tau, eta = _unpack_theta(params, d)
    A = _branching_matrix(c, tau, eta)
    rho = _spectral_radius(A)

    pen = 0.0

    def sq_pos(x):
        return float(max(0.0, x)) ** 2

    for x in mu:
        pen += sq_pos(HEALTHY_MU_MIN - x) / (HEALTHY_MU_MIN ** 2)
        pen += sq_pos(x - HEALTHY_MU_MAX) / (HEALTHY_MU_MAX ** 2)

    for x in tau.reshape(-1):
        pen += sq_pos(HEALTHY_TAU_MIN - x) / (HEALTHY_TAU_MIN ** 2)
        pen += sq_pos(x - HEALTHY_TAU_MAX) / (HEALTHY_TAU_MAX ** 2)

    for x in eta.reshape(-1):
        pen += sq_pos(HEALTHY_ETA_MIN - x) / (HEALTHY_ETA_MIN ** 2)
        pen += sq_pos(x - HEALTHY_ETA_MAX) / (HEALTHY_ETA_MAX ** 2)

    pen += 25.0 * sq_pos(rho - rho_soft_max) / (max(1e-6, 1.0 - rho_soft_max) ** 2)

    if np.any(~np.isfinite(A)) or not np.isfinite(rho):
        return 1e100
    return penalty_weight * pen


def _objective_unconstrained(theta, times_list, d=2, stability_rho_max=RAW_STABILITY_RHO_MAX):
    params = _from_unconstrained(theta, d=d)
    val = mv_hawkes_negloglik(params, times_list, d=d, stability_rho_max=stability_rho_max)
    if not np.isfinite(val):
        return 1e100
    pen = _robust_penalty_from_params(params, d=d)
    if not np.isfinite(pen):
        return 1e100
    return float(val + pen)


def _objective_unconstrained_with_grad(theta, times_list, d=2, stability_rho_max=RAW_STABILITY_RHO_MAX):
    base = _objective_unconstrained(theta, times_list, d=d, stability_rho_max=stability_rho_max)
    if not np.isfinite(base):
        return 1e100, np.zeros_like(theta)

    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        h = 1e-4 * max(1.0, abs(theta[i]))
        th_plus = theta.copy()
        th_minus = theta.copy()
        th_plus[i] += h
        th_minus[i] -= h
        v_plus = _objective_unconstrained(th_plus, times_list, d=d, stability_rho_max=stability_rho_max)
        v_minus = _objective_unconstrained(th_minus, times_list, d=d, stability_rho_max=stability_rho_max)
        if np.isfinite(v_plus) and np.isfinite(v_minus):
            grad[i] = (v_plus - v_minus) / (2.0 * h)
        elif np.isfinite(v_plus):
            grad[i] = (v_plus - base) / h
        elif np.isfinite(v_minus):
            grad[i] = (base - v_minus) / h
        else:
            grad[i] = 0.0
    grad[~np.isfinite(grad)] = 0.0
    return float(base), grad


def _mv_make_inits(times_list, d, n_starts=20, seed=42):
    t_all, _ = _stack_events(times_list)
    t_all = np.sort(t_all[np.isfinite(t_all)])
    if len(t_all) < 30:
        return []

    T_end = max(t_all[-1] - t_all[0], 1e-6)
    rates = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        rates.append(len(a) / T_end)
    rates = np.maximum(np.asarray(rates, dtype=float), 1e-4)

    ia = np.diff(t_all)
    ia = ia[ia > 0]
    mean_ia = float(np.mean(ia)) if len(ia) else 1.0

    rng = np.random.default_rng(seed)
    inits = []

    # Conservative 2D initialization in the spirit of 1D main.py:
    # keep the initial branching ratios moderate rather than starting near instability.
    # Here A0[target, source] is the initial branching matrix. The diagonal entries
    # are self-excitation guesses and are kept in a conservative range around ~0.1-0.35,
    # while off-diagonal cross-excitation entries are smaller.
    anchor_As = [
        np.array([[0.08, 0.01], [0.01, 0.08]], dtype=float),
        np.array([[0.16, 0.02], [0.02, 0.16]], dtype=float),
        np.array([[0.24, 0.03], [0.03, 0.24]], dtype=float),
        np.array([[0.32, 0.05], [0.05, 0.32]], dtype=float),
    ]
    # Same spirit as 1D: multiple conservative seeds from smaller to larger baseline share.
    anchor_scales = [0.20, 0.40, 0.65, 0.90]

    for scale, A0 in zip(anchor_scales, anchor_As):
        # Use the diagonal of A0 to split the observed rate into baseline vs endogenous part,
        # matching the 1D philosophy: stronger assumed self-excitation -> smaller mu0.
        mu0 = np.maximum(rates * (1.0 - np.diag(A0)) * scale, rates * 0.05)
        tau0 = mean_ia * np.array([[5.0, 8.0], [8.0, 5.0]], dtype=float)
        eta0 = np.array([[2.2, 2.6], [2.6, 2.2]], dtype=float)
        c0 = A0 * (eta0 - 1.0) / tau0
        inits.append(_pack_theta(mu0, c0, tau0, eta0))

    while len(inits) < n_starts:
        # Conservative random seeds: keep branching ratios in a moderate range.
        s = rng.uniform(0.20, 0.90)

        A0 = rng.uniform(0.005, 0.08, size=(d, d))
        A0[np.diag_indices(d)] = rng.uniform(0.08, 0.35, size=d)
        rho = _spectral_radius(A0.T)
        if rho >= 0.75:
            A0 *= (0.75 / max(rho, 1e-12))

        mu0 = np.maximum(rates * (1.0 - np.diag(A0)) * s, rates * 0.05)

        tau0 = mean_ia * rng.uniform(2.0, 15.0, size=(d, d))
        tau0 = np.clip(tau0, 1e-4, 2e4)
        eta0 = rng.uniform(1.6, 4.0, size=(d, d))
        c0 = A0 * (eta0 - 1.0) / tau0
        inits.append(_pack_theta(mu0, c0, tau0, eta0))

    return inits[:n_starts]


def _gradient_norm_at_params(params, times_list, d=2, stability_rho_max=0.999):
    _, grad = mv_hawkes_negloglik_and_grad(params, times_list, d=d, stability_rho_max=stability_rho_max)
    return float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else np.inf


def _record_from_result(start_id, seed, init_params, final_params, res, times_list, d=2):
    mu_i, c_i, tau_i, eta_i = _unpack_theta(init_params, d)
    A_i = _branching_matrix(c_i, tau_i, eta_i)
    rho_i = _spectral_radius(A_i)

    mu_f, c_f, tau_f, eta_f = _unpack_theta(final_params, d)
    A_f = _branching_matrix(c_f, tau_f, eta_f)
    rho_f = _spectral_radius(A_f)

    grad_init = _gradient_norm_at_params(init_params, times_list, d=d)
    grad_final = _gradient_norm_at_params(final_params, times_list, d=d)

    msg = str(getattr(res, "message", "")).replace("\n", " | ")

    return {
        "start_id": int(start_id),
        "seed": int(seed),
        "success": bool(getattr(res, "success", False)),
        "fun": float(getattr(res, "fun", np.inf)),
        "nit": int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else -1,
        "nfev": int(getattr(res, "nfev", -1)) if getattr(res, "nfev", None) is not None else -1,
        "njev": int(getattr(res, "njev", -1)) if getattr(res, "njev", None) is not None else -1,
        "grad_norm_init": grad_init,
        "grad_norm_final": grad_final,
        "rho_init": float(rho_i),
        "rho": float(rho_f),
        "mu_buy_init": float(mu_i[0]),
        "mu_sell_init": float(mu_i[1]),
        "mu_buy": float(mu_f[0]),
        "mu_sell": float(mu_f[1]),
        "A_bb_init": float(A_i[0, 0]),
        "A_bs_init": float(A_i[0, 1]),
        "A_sb_init": float(A_i[1, 0]),
        "A_ss_init": float(A_i[1, 1]),
        "A_bb": float(A_f[0, 0]),
        "A_bs": float(A_f[0, 1]),
        "A_sb": float(A_f[1, 0]),
        "A_ss": float(A_f[1, 1]),
        "message": msg,
    }


def _is_healthy_record(rec, rho_max=HEALTHY_RHO_MAX, grad_max=HEALTHY_GRAD_MAX):
    if not bool(rec.get("success", False)):
        return False
    if not np.isfinite(rec.get("fun", np.inf)):
        return False
    if not np.isfinite(rec.get("rho", np.inf)) or rec["rho"] >= rho_max:
        return False
    if not np.isfinite(rec.get("grad_norm_final", np.inf)) or rec["grad_norm_final"] > grad_max:
        return False
    if min(rec.get("mu_buy", 0.0), rec.get("mu_sell", 0.0)) < HEALTHY_MU_MIN:
        return False
    if max(rec.get("mu_buy", 0.0), rec.get("mu_sell", 0.0)) > HEALTHY_MU_MAX:
        return False
    return True


def _summary_from_params(params, times_list, d=2, record=None):
    mu, c, tau, eta = _unpack_theta(params, d)
    A = _branching_matrix(c, tau, eta)
    rho = _spectral_radius(A)
    grad_norm = _gradient_norm_at_params(params, times_list, d=d, stability_rho_max=RAW_STABILITY_RHO_MAX)
    out = {
        "mu": mu, "c": c, "tau": tau, "eta": eta, "A": A, "rho": rho, "grad_norm": grad_norm
    }
    if record is not None:
        out.update({
            "fun": record["fun"],
            "start_id": int(record["start_id"]),
            "success": bool(record["success"]),
            "message": record["message"],
        })
    return out


def fit_2d_hawkes_powerlaw(
    mo_buy,
    mo_sell,
    label="2D Directional Asymmetry (power-law)",
    n_starts=12,
    verbose=True,
    seed=42,
    stability_rho_max=RAW_STABILITY_RHO_MAX,
    accept_non_success_best=True,
):
    times_list = [np.asarray(mo_buy, dtype=float), np.asarray(mo_sell, dtype=float)]
    d = 2

    total_events = sum(len(np.asarray(x)[np.isfinite(x)]) for x in times_list)
    if total_events < 10:
        raise ValueError("Not enough events to fit the 2D Hawkes model.")

    inits = _mv_make_inits(times_list, d, n_starts=n_starts, seed=seed)
    if not inits:
        raise ValueError("Could not create valid initial guesses for 2D Hawkes.")

    records = []
    final_params_by_start = {}
    best_success = None
    best_success_val = np.inf
    best_any = None
    best_any_val = np.inf

    print("\n" + "=" * 70)
    print(f"ROBUST DEBUG OPTIMIZATION TRACE — {label}")
    print(f"n_starts   = {n_starts}")
    print(f"seed       = {seed}")
    print(f"n_events   = {total_events}")
    print(f"accept_non_success_best = {accept_non_success_best}")
    print("=" * 70)

    for i, x0 in enumerate(inits, start=1):
        theta0 = _to_unconstrained(x0, d=d)
        mu0, c0, tau0, eta0 = _unpack_theta(x0, d)
        A0 = _branching_matrix(c0, tau0, eta0)
        rho0 = _spectral_radius(A0)
        grad0 = _gradient_norm_at_params(x0, times_list, d=d, stability_rho_max=stability_rho_max)

        print("\n" + "." * 70)
        print(f"Start {i:02d}/{len(inits)}")
        print(f"  clock      = {datetime.now().strftime('%H:%M:%S')}")
        print(f"  init mu_buy  = {mu0[0]:.10f}")
        print(f"  init mu_sell = {mu0[1]:.10f}")
        print(f"  init rho(A)  = {rho0:.10f}")
        print(f"  ||grad||_init= {grad0:.10f}")
        print("  init A:")
        print(pd.DataFrame(A0, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))

        fun = lambda th: _objective_unconstrained(th, times_list, d=d, stability_rho_max=stability_rho_max)
        jac = lambda th: _objective_unconstrained_with_grad(th, times_list, d=d, stability_rho_max=stability_rho_max)[1]

        res = minimize(
            fun=fun,
            x0=theta0,
            method="L-BFGS-B",
            jac=jac,
            options={
                "maxiter": 800,
                "maxfun": 5000,
                "ftol": 1e-12,
                "gtol": 1e-7,
                "maxls": 50,
            },
        )

        p_final = _from_unconstrained(res.x, d=d)
        mu_f, c_f, tau_f, eta_f = _unpack_theta(p_final, d)
        A_f = _branching_matrix(c_f, tau_f, eta_f)
        rho_f = _spectral_radius(A_f)
        grad_f = _gradient_norm_at_params(p_final, times_list, d=d, stability_rho_max=stability_rho_max)

        print(f"  done success = {bool(res.success)}")
        print(f"  done fun     = {float(res.fun):.10f}")
        print(f"  done nit     = {getattr(res, 'nit', -1)}")
        print(f"  done nfev    = {getattr(res, 'nfev', -1)}")
        print(f"  done njev    = {getattr(res, 'njev', -1)}")
        print(f"  done rho(A)  = {rho_f:.10f}")
        print(f"  ||grad||_final= {grad_f:.10f}")
        print(f"  message      = {str(res.message).replace(chr(10), ' | ')}")
        print("  done A:")
        print(pd.DataFrame(A_f, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))

        rec = _record_from_result(i, seed, x0, p_final, res, times_list, d=d)
        records.append(rec)
        final_params_by_start[int(i)] = p_final

        if np.isfinite(res.fun) and res.fun < best_any_val:
            best_any_val = float(res.fun)
            best_any = (res, p_final)

        if bool(res.success) and np.isfinite(res.fun) and res.fun < best_success_val:
            best_success_val = float(res.fun)
            best_success = (res, p_final)

    if best_success is None and best_any is None:
        raise RuntimeError(f"Optimization failed for {label}.")

    records_df = pd.DataFrame(records).sort_values(["fun", "nit", "start_id"]).reset_index(drop=True)
    records_df["healthy"] = records_df.apply(lambda r: _is_healthy_record(r.to_dict()), axis=1)
    healthy_df = records_df[records_df["healthy"]].copy().sort_values(["fun", "nit", "start_id"]).reset_index(drop=True)

    if best_success is None:
        selected_res, p = best_any
        selection_mode = "best run overall (no successful run available)"
    elif accept_non_success_best and best_any is not None and best_any_val + 1e-8 < best_success_val:
        selected_res, p = best_any
        selection_mode = "best run overall (better than best successful run)"
    else:
        selected_res, p = best_success
        selection_mode = "best successful run"

    raw_row = records_df.iloc[0].to_dict()
    raw_summary = _summary_from_params(final_params_by_start[int(raw_row["start_id"])], times_list, d=d, record=raw_row)

    if len(healthy_df):
        healthy_row = healthy_df.iloc[0].to_dict()
        healthy_summary = _summary_from_params(final_params_by_start[int(healthy_row["start_id"])], times_list, d=d, record=healthy_row)
        healthy_mode = "best healthy run"
    else:
        healthy_summary = None
        healthy_mode = "no healthy run"

    mu, c, tau, eta = _unpack_theta(p, d)
    A = _branching_matrix(c, tau, eta)
    rho = _spectral_radius(A)
    grad_norm = _gradient_norm_at_params(p, times_list, d=d, stability_rho_max=stability_rho_max)

    if verbose:
        print("\n" + "=" * 70)
        print(f"2D power-law Hawkes fit — {label}")
        print(f"selection_mode = {selection_mode}")
        print(f"mu_buy  = {mu[0]:.10f}")
        print(f"mu_sell = {mu[1]:.10f}")
        print(f"||grad|| = {grad_norm:.10f}")
        print("\nc matrix:")
        print(pd.DataFrame(c, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))
        print("\ntau matrix:")
        print(pd.DataFrame(tau, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))
        print("\neta matrix:")
        print(pd.DataFrame(eta, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))
        print("\nBranching matrix A (rows=target, cols=source):")
        print(pd.DataFrame(A, index=["BUY_MO", "SELL_MO"], columns=["BUY_MO", "SELL_MO"]).to_string(float_format=lambda x: f"{x:.10f}"))
        print(f"\nSpectral radius rho(A) = {rho:.10f}")
        print("=" * 70)

    return {
        "mu": mu,
        "c": c,
        "tau": tau,
        "eta": eta,
        "A": A,
        "rho": rho,
        "success": bool(getattr(selected_res, "success", False)),
        "selection_mode": selection_mode,
        "healthy_selection_mode": healthy_mode,
        "mo_buy": np.asarray(mo_buy, dtype=float),
        "mo_sell": np.asarray(mo_sell, dtype=float),
        "opt_result": selected_res,
        "records_df": records_df,
        "healthy_df": healthy_df,
        "grad_norm": grad_norm,
        "seed": seed,
        "best_success_fun": best_success_val if best_success is not None else np.nan,
        "best_any_fun": best_any_val if best_any is not None else np.nan,
        "raw_best": raw_summary,
        "healthy_best": healthy_summary,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def _mv_intensity_on_grid(times_list, mu, c, tau, eta, t_grid):
    d = len(times_list)
    t_grid = np.asarray(t_grid, dtype=float)
    lam = np.zeros((len(t_grid), d), dtype=float)

    ts = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        a = np.sort(a)
        ts.append(a)

    for g_idx, t in enumerate(t_grid):
        lam[g_idx, :] = mu.copy()
        for source in range(d):
            src_times = ts[source]
            if len(src_times) == 0:
                continue
            past = src_times[src_times < t]
            if len(past) == 0:
                continue
            dts = t - past
            for target in range(d):
                lam[g_idx, target] += np.sum(powerlaw_kernel(dts, c[source, target], tau[source, target], eta[source, target]))
    return lam


def plot_healthy_heatmap(summary, ticker, outdir=PLOT_DIR, vmax=None):
    M = np.asarray(summary["A"], dtype=float)
    vmax = float(np.nanmax(M)) if vmax is None else float(vmax)
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), constrained_layout=True)
    im = ax.imshow(M, aspect="equal", cmap="Reds", vmin=0.0, vmax=vmax)
    ax.set_title(f"{ticker} healthy best (power-law)")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["BUY_MO", "SELL_MO"], rotation=45, ha="right")
    ax.set_yticklabels(["BUY_MO", "SELL_MO"])
    for i in range(2):
        for j in range(2):
            txt = "NA" if not np.isfinite(M[i, j]) else f"{M[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=10, color="black")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04, shrink=0.95)
    path = os.path.join(outdir, f"powerlaw_healthy_{ticker}.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    print(f"    Saved: {path}")
    plt.close(fig)


# =============================================================================
# RUNNER
# =============================================================================

def run_directional_asymmetry_2d(df, ticker="AMZN", make_plots=False, verbose=True, use_time_since_open=False, n_starts=12, base_seed=42):
    mo_buy, mo_sell = extract_directional_events(df, use_time_since_open=use_time_since_open)

    print("\n" + "=" * 80)
    print(f"Directional asymmetry study — {ticker} (2D power-law Hawkes)")
    print(f"BUY_MO events : {len(mo_buy)}")
    print(f"SELL_MO events: {len(mo_sell)}")
    print("=" * 80)

    if len(mo_buy) < 10 or len(mo_sell) < 10:
        print(f"  ⚠  Not enough buy/sell market-order events for {ticker}.")
        return None

    seed = _stable_seed_from_ticker(ticker, base_seed=base_seed)
    print(f"  Using ticker-specific seed = {seed}")

    res2 = fit_2d_hawkes_powerlaw(
        mo_buy,
        mo_sell,
        label=f"{ticker} 2D directional asymmetry",
        n_starts=n_starts,
        verbose=verbose,
        seed=seed,
        stability_rho_max=RAW_STABILITY_RHO_MAX,
        accept_non_success_best=True,
    )

    chosen = res2.get("healthy_best") or res2.get("raw_best")
    A = chosen["A"]
    print("\n  Interpretation")
    print(f"    BUY <- BUY   = {A[0, 0]:.6f}")
    print(f"    BUY <- SELL  = {A[0, 1]:.6f}")
    print(f"    SELL <- BUY  = {A[1, 0]:.6f}")
    print(f"    SELL <- SELL = {A[1, 1]:.6f}")

    if abs(A[0, 0] - A[1, 1]) < 0.05:
        print("    → Buy and sell flows look similarly self-exciting.")
    elif A[0, 0] > A[1, 1]:
        print("    → Buy flow appears MORE self-exciting than sell flow.")
    else:
        print("    → Sell flow appears MORE self-exciting than buy flow.")

    if make_plots:
        plot_healthy_heatmap(chosen, ticker, outdir=PLOT_DIR)

    return res2


def run_all_stocks_directional_asymmetry(stocks=None, data_path=DATA_PATH, start_date=START_DATE, end_date=END_DATE, make_plots=False, n_starts=12, base_seed=42):
    if stocks is None:
        stocks = STOCKS

    results = {}
    print("\n" + "=" * 80)
    print("STEP — 2D Directional Asymmetry for all stocks (POWER-LAW)")
    print("=" * 80)

    for ticker in stocks:
        print(f"\n{'-' * 70}")
        print(f"Processing {ticker}")
        print(f"{'-' * 70}")

        loader = Loader(ticker, start_date, end_date, dataPath=data_path)
        data_list = loader.load()
        if not data_list:
            results[ticker] = None
            continue

        df = data_list[0].copy()

        try:
            res = run_directional_asymmetry_2d(
                df,
                ticker=ticker,
                make_plots=make_plots,
                verbose=True,
                use_time_since_open=True,
                n_starts=n_starts,
                base_seed=base_seed,
            )
            results[ticker] = res
        except Exception as e:
            print(f"  ✗ Failed for {ticker}: {e}")
            results[ticker] = None

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        print("\n" + "=" * 80)
        print("FINAL CROSS-STOCK DIRECTIONAL SUMMARY (POWER-LAW)")
        print("=" * 80)
        rows = []
        for ticker, res in valid.items():
            chosen = res.get("healthy_best") or res.get("raw_best")
            A = chosen["A"]
            rows.append({
                "Ticker": ticker,
                "BUY<-BUY": A[0, 0],
                "BUY<-SELL": A[0, 1],
                "SELL<-BUY": A[1, 0],
                "SELL<-SELL": A[1, 1],
                "rho(A)": chosen["rho"],
                "||grad||": chosen.get("grad_norm", np.nan),
                "seed": res["seed"],
                "best_success_fun": res["best_success_fun"],
                "best_any_fun": res["best_any_fun"],
                "selection_mode": "best healthy run" if res.get("healthy_best") is not None else res["selection_mode"],
            })
        summary_df = pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    return results


if __name__ == "__main__":
    run_all_stocks_directional_asymmetry(
        stocks=STOCKS,
        data_path=DATA_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
        make_plots=True,
        n_starts=12,
        base_seed=42,
    )

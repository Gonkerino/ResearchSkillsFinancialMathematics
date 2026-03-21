import os
import hashlib
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# =============================================================================
# CONFIG
# =============================================================================
STOCKS = ["AMZN", "AAPL", "GOOG", "MSFT", "INTC"]
DATA_PATH = "data/"
START_DATE = "2012-06-21"
END_DATE = "2012-06-21"
PLOT_DIR = "plot"
os.makedirs(PLOT_DIR, exist_ok=True)

DIM_NAMES = ["UP", "DOWN", "BUY_MO", "SELL_MO"]
D = 4

RAW_STABILITY_RHO_MAX = 0.99
HEALTHY_RHO_MAX = 0.95
HEALTHY_GRAD_MAX = 2.5e5
HEALTHY_MU_MIN = 1e-4
HEALTHY_MU_MAX = 5.0

plt.rcParams.update({
    "figure.figsize": (12, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})


# =============================================================================
# HELPERS
# =============================================================================
def _save_current_fig(filename):
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {path}")


def _ticker_seed(ticker, base_seed=42):
    x = f"{ticker}_{base_seed}".encode("utf-8")
    return int(hashlib.md5(x).hexdigest()[:8], 16)


# =============================================================================
# LOADER
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

            combined = pd.concat([msg.reset_index(drop=True), ob_aligned.reset_index(drop=True)], axis=1)
            combined["Date"] = date_str
            combined["Ticker"] = self.ric
            combined["TimeSinceOpen"] = combined["Time"] - t_start
            data.append(combined)

        if not data:
            print(f"  ⚠  No data found for {self.ric} between {self.sDate} and {self.eDate}.")
            print(f"     Expected files in: {os.path.abspath(self.dataPath)}")

        return data


# =============================================================================
# EVENT EXTRACTION
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


def extract_up_down_midprice_times(df, use_time_since_open=False):
    ask_candidates = ["AskPrice1", "Ask Price 1", "ask_price_1", "AskPrice_1", "ASKPRICE1"]
    bid_candidates = ["BidPrice1", "Bid Price 1", "bid_price_1", "BidPrice_1", "BIDPRICE1"]

    def _pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        lower_map = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    ask_col = _pick(ask_candidates)
    bid_col = _pick(bid_candidates)
    if ask_col is None or bid_col is None:
        raise KeyError("Cannot find best bid/ask columns for mid-price extraction.")

    mid = (df[ask_col].values.astype(float) + df[bid_col].values.astype(float)) / 2.0
    dmid = np.diff(mid, prepend=mid[0])
    tcol = "TimeSinceOpen" if use_time_since_open and "TimeSinceOpen" in df.columns else "Time"
    t = df[tcol].values.astype(float)
    up_t = np.sort(t[dmid > 0])
    dn_t = np.sort(t[dmid < 0])
    return up_t, dn_t


# =============================================================================
# MULTIVARIATE HAWKES CORE
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


def _branching_matrix(alpha, beta):
    d = alpha.shape[0]
    A = np.zeros((d, d), dtype=float)
    for source in range(d):
        for target in range(d):
            A[target, source] = alpha[source, target] / beta[source, target]
    return A


def _spectral_radius(M):
    eigvals = np.linalg.eigvals(M)
    return float(np.max(np.abs(eigvals)))


def mv_hawkes_negloglik(params, times_list, d=4, stability_rho_max=RAW_STABILITY_RHO_MAX):
    params = np.asarray(params, dtype=float)
    mu = params[:d]
    alpha = params[d:d + d * d].reshape(d, d)
    beta = params[d + d * d:].reshape(d, d)

    if np.any(mu <= 0) or np.any(alpha < 0) or np.any(beta <= 0):
        return np.inf

    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)
    if (not np.isfinite(rho)) or rho >= stability_rho_max:
        return np.inf

    t_all, k_all = _stack_events(times_list)
    if len(t_all) < 20:
        return np.inf

    t0 = t_all[0]
    t_all = t_all - t0
    T_end = t_all[-1]

    shifted = []
    for arr in times_list:
        a = np.asarray(arr, dtype=float)
        a = np.sort(a[np.isfinite(a)])
        shifted.append(a - t0 if len(a) else np.array([]))

    R = np.zeros((d, d), dtype=float)
    ll = 0.0
    last_t = 0.0

    for t, k in zip(t_all, k_all):
        dt = t - last_t
        if dt < 0:
            return np.inf
        R *= np.exp(-beta * dt)
        lam_k = mu[k] + np.sum(alpha[:, k] * R[:, k])
        if lam_k <= 0 or not np.isfinite(lam_k):
            return np.inf
        ll += np.log(lam_k)
        R[k, :] += 1.0
        last_t = t

    comp = float(np.sum(mu) * T_end)
    for source in range(d):
        ts = shifted[source]
        if len(ts) == 0:
            continue
        delta = T_end - ts
        for target in range(d):
            a = alpha[source, target]
            b = beta[source, target]
            comp += (a / b) * np.sum(1.0 - np.exp(-b * delta))

    return -(ll - comp)


def _pack_theta(mu, alpha, beta):
    return np.concatenate([mu, alpha.reshape(-1), beta.reshape(-1)])


def _unpack_theta(params, d=4):
    params = np.asarray(params, dtype=float)
    mu = params[:d]
    alpha = params[d:d + d * d].reshape(d, d)
    beta = params[d + d * d:].reshape(d, d)
    return mu, alpha, beta


def _to_unconstrained(params, d=4):
    mu, alpha, beta = _unpack_theta(params, d)
    mu_u = np.log(np.maximum(mu, 1e-12))
    alpha_u = np.log(np.maximum(alpha + 1e-12, 1e-12))
    beta_u = np.log(np.maximum(beta, 1e-12))
    return _pack_theta(mu_u, alpha_u, beta_u)


def _from_unconstrained(theta, d=4):
    mu_u, alpha_u, beta_u = _unpack_theta(theta, d)
    mu = np.exp(mu_u)
    alpha = np.exp(alpha_u)
    beta = np.exp(beta_u)
    alpha = np.maximum(alpha - 1e-12, 0.0)
    return _pack_theta(mu, alpha, beta)


def _soft_penalty(mu, alpha, beta, A, rho):
    pen = 0.0
    pen += 25.0 * max(0.0, rho - 0.92) ** 2
    pen += 0.1 * np.sum(np.maximum(0.0, 0.01 - mu) ** 2)
    pen += 0.001 * np.sum(np.maximum(0.0, beta - 2500.0) ** 2)
    pen += 0.001 * np.sum(np.maximum(0.0, 0.02 - beta) ** 2)
    pen += 0.05 * np.sum(np.maximum(0.0, A - 0.95) ** 2)
    return float(pen)


def _objective_unconstrained(theta, times_list, d=4, stability_rho_max=RAW_STABILITY_RHO_MAX):
    params = _from_unconstrained(theta, d=d)
    mu, alpha, beta = _unpack_theta(params, d)
    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)
    if (not np.isfinite(rho)) or rho >= stability_rho_max:
        return 1e100
    base = mv_hawkes_negloglik(params, times_list, d=d, stability_rho_max=stability_rho_max)
    if not np.isfinite(base):
        return 1e100
    return float(base + _soft_penalty(mu, alpha, beta, A, rho))


def _objective_unconstrained_with_grad(theta, times_list, d=4, stability_rho_max=RAW_STABILITY_RHO_MAX):
    base = _objective_unconstrained(theta, times_list, d=d, stability_rho_max=stability_rho_max)
    if not np.isfinite(base):
        return 1e100, np.zeros_like(theta)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        h = 1e-4 * max(1.0, abs(theta[i]))
        th_plus = theta.copy(); th_plus[i] += h
        th_minus = theta.copy(); th_minus[i] -= h
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


def mv_hawkes_negloglik_and_grad(params, times_list, d=4, stability_rho_max=RAW_STABILITY_RHO_MAX):
    params = np.asarray(params, dtype=float)
    base = mv_hawkes_negloglik(params, times_list, d=d, stability_rho_max=stability_rho_max)
    grad = np.zeros_like(params)
    if not np.isfinite(base):
        return np.inf, grad
    for i in range(len(params)):
        h = 1e-4 * max(1.0, abs(params[i]))
        p_plus = params.copy(); p_plus[i] += h
        p_minus = params.copy(); p_minus[i] -= h
        v_plus = mv_hawkes_negloglik(p_plus, times_list, d=d, stability_rho_max=stability_rho_max)
        v_minus = mv_hawkes_negloglik(p_minus, times_list, d=d, stability_rho_max=stability_rho_max)
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


def _mv_make_inits(times_list, d=4, n_starts=12, seed=42):
    t_all, _ = _stack_events(times_list)
    t_all = np.sort(t_all[np.isfinite(t_all)])
    if len(t_all) < 40:
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
    beta0 = float(np.clip(1.0 / (np.mean(ia) if len(ia) else 1.0), 0.05, 300.0))
    rng = np.random.default_rng(seed)
    inits = []

    # Unbiased anchor templates: neutral, self-dominant, cross-dominant, and mixed.
    anchors = [
        np.array([
            [0.08, 0.08, 0.04, 0.04],
            [0.08, 0.08, 0.04, 0.04],
            [0.04, 0.04, 0.08, 0.08],
            [0.04, 0.04, 0.08, 0.08],
        ], dtype=float),
        np.array([
            [0.22, 0.08, 0.03, 0.03],
            [0.08, 0.22, 0.03, 0.03],
            [0.03, 0.03, 0.22, 0.08],
            [0.03, 0.03, 0.08, 0.22],
        ], dtype=float),
        np.array([
            [0.10, 0.24, 0.03, 0.03],
            [0.24, 0.10, 0.03, 0.03],
            [0.03, 0.03, 0.10, 0.24],
            [0.03, 0.03, 0.24, 0.10],
        ], dtype=float),
        np.array([
            [0.18, 0.16, 0.05, 0.05],
            [0.16, 0.18, 0.05, 0.05],
            [0.05, 0.05, 0.18, 0.16],
            [0.05, 0.05, 0.16, 0.18],
        ], dtype=float),
    ]

    beta_patterns = [
        np.ones((d, d), dtype=float),
        np.array([
            [1.0, 0.9, 0.7, 0.7],
            [0.9, 1.0, 0.7, 0.7],
            [0.7, 0.7, 1.0, 0.9],
            [0.7, 0.7, 0.9, 1.0],
        ], dtype=float),
    ]

    for idx, A0 in enumerate(anchors):
        rho = _spectral_radius(A0.T)
        if rho >= 0.92:
            A0 = A0 * (0.92 / max(rho, 1e-12))
        scale = 0.30 + 0.12 * idx
        avg_excitation = np.clip(np.mean(A0, axis=1), 0.05, 0.85)
        mu0 = np.maximum(rates * np.maximum(1.0 - avg_excitation, 0.18) * scale, rates * 0.03)
        beta0m = beta0 * beta_patterns[idx % len(beta_patterns)]
        alpha0 = A0 * beta0m
        inits.append(_pack_theta(mu0, alpha0, beta0m))

    # Random starts without diagonal bias: sample all entries on the same scale,
    # then explicitly mix regimes across self-dominant / cross-dominant / neutral.
    regimes = ["neutral", "self", "cross"]
    while len(inits) < n_starts:
        regime = regimes[(len(inits) - len(anchors)) % len(regimes)]
        mu0 = np.maximum(rates * rng.uniform(0.2, 0.9, size=d), 1e-5)

        A0 = rng.uniform(0.03, 0.28, size=(d, d))

        if regime == "self":
            bump = rng.uniform(0.06, 0.22, size=d)
            for i in range(d):
                A0[i, i] += bump[i]
        elif regime == "cross":
            pairs = [(0, 1), (1, 0), (2, 3), (3, 2)]
            for i, j in pairs:
                A0[i, j] += rng.uniform(0.08, 0.24)
        else:  # neutral
            A0 += rng.uniform(-0.02, 0.02, size=(d, d))
            A0 = np.clip(A0, 0.01, None)

        rho = _spectral_radius(A0.T)
        if rho >= 0.92:
            A0 *= 0.92 / max(rho, 1e-12)

        beta0m = beta0 * rng.uniform(0.3, 2.0, size=(d, d))
        beta0m = np.clip(beta0m, 0.03, 2500.0)
        alpha0 = A0 * beta0m
        inits.append(_pack_theta(mu0, alpha0, beta0m))

    return inits[:n_starts]


def _gradient_norm_at_params(params, times_list, d=4, stability_rho_max=RAW_STABILITY_RHO_MAX):
    _, grad = mv_hawkes_negloglik_and_grad(params, times_list, d=d, stability_rho_max=stability_rho_max)
    return float(np.linalg.norm(grad)) if np.all(np.isfinite(grad)) else np.inf


def _record_from_result(start_id, seed, init_params, final_params, res, times_list, d=4, dim_names=None):
    dim_names = dim_names or [f"dim{i}" for i in range(d)]
    mu_i, alpha_i, beta_i = _unpack_theta(init_params, d)
    A_i = _branching_matrix(alpha_i, beta_i)
    rho_i = _spectral_radius(A_i)
    mu_f, alpha_f, beta_f = _unpack_theta(final_params, d)
    A_f = _branching_matrix(alpha_f, beta_f)
    rho_f = _spectral_radius(A_f)
    grad_init = _gradient_norm_at_params(init_params, times_list, d=d)
    grad_final = _gradient_norm_at_params(final_params, times_list, d=d)
    rec = {
        "start_id": int(start_id), "seed": int(seed), "success": bool(getattr(res, "success", False)),
        "fun": float(getattr(res, "fun", np.inf)), "nit": int(getattr(res, "nit", -1) or -1),
        "nfev": int(getattr(res, "nfev", -1) or -1), "njev": int(getattr(res, "njev", -1) or -1),
        "grad_norm_init": grad_init, "grad_norm_final": grad_final, "rho_init": rho_i, "rho": rho_f,
        "message": str(getattr(res, "message", "")).replace("\n", " | "),
    }
    for i, name in enumerate(dim_names):
        rec[f"mu_{name}"] = float(mu_f[i])
    for i, ti in enumerate(dim_names):
        for j, sj in enumerate(dim_names):
            rec[f"A_{ti}_{sj}"] = float(A_f[i, j])
    return rec


def _is_healthy_record(rec, dim_names=None, rho_max=HEALTHY_RHO_MAX, grad_max=HEALTHY_GRAD_MAX):
    dim_names = dim_names or DIM_NAMES
    if not bool(rec.get("success", False)):
        return False
    if not np.isfinite(rec.get("fun", np.inf)):
        return False
    if not np.isfinite(rec.get("rho", np.inf)) or rec["rho"] >= rho_max:
        return False
    if not np.isfinite(rec.get("grad_norm_final", np.inf)) or rec["grad_norm_final"] > grad_max:
        return False
    mus = [rec.get(f"mu_{name}", 0.0) for name in dim_names]
    if min(mus) < HEALTHY_MU_MIN:
        return False
    if max(mus) > HEALTHY_MU_MAX:
        return False
    return True


def _summary_from_params(params, times_list, d=4, record=None):
    mu, alpha, beta = _unpack_theta(params, d)
    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)
    grad_norm = _gradient_norm_at_params(params, times_list, d=d, stability_rho_max=RAW_STABILITY_RHO_MAX)
    out = {"mu": mu, "alpha": alpha, "beta": beta, "A": A, "rho": rho, "grad_norm": grad_norm}
    if record is not None:
        out.update({
            "fun": record["fun"], "start_id": int(record["start_id"]),
            "success": bool(record["success"]), "message": record["message"],
        })
    return out


def fit_4d_hawkes(times_list, label="4D robust Hawkes", dim_names=None, n_starts=12, verbose=True, seed=42,
                  stability_rho_max=RAW_STABILITY_RHO_MAX, accept_non_success_best=True):
    dim_names = dim_names or DIM_NAMES
    d = len(times_list)
    total_events = sum(len(np.asarray(x)[np.isfinite(x)]) for x in times_list)
    if total_events < 30:
        raise ValueError("Not enough events to fit the 4D Hawkes model.")

    inits = _mv_make_inits(times_list, d=d, n_starts=n_starts, seed=seed)
    if not inits:
        raise ValueError("Could not create valid initial guesses for 4D Hawkes.")

    records, final_params_by_start = [], {}
    best_success, best_any = None, None
    best_success_val, best_any_val = np.inf, np.inf

    print("\n" + "=" * 70)
    print(f"ROBUST DEBUG OPTIMIZATION TRACE — {label}")
    print(f"n_starts   = {n_starts}")
    print(f"seed       = {seed}")
    print(f"n_events   = {total_events}")
    print(f"accept_non_success_best = {accept_non_success_best}")
    print("=" * 70)

    for i, x0 in enumerate(inits, start=1):
        try:
            theta0 = _to_unconstrained(x0, d=d)
            mu0, alpha0, beta0 = _unpack_theta(x0, d)
            A0 = _branching_matrix(alpha0, beta0)
            rho0 = _spectral_radius(A0)
            grad0 = _gradient_norm_at_params(x0, times_list, d=d, stability_rho_max=stability_rho_max)

            print("\n" + "." * 70)
            print(f"Start {i:02d}/{len(inits)}")
            print(f"  clock      = {datetime.now().strftime('%H:%M:%S')}")
            print(f"  init rho(A)  = {rho0:.10f}")
            print(f"  ||grad||_init= {grad0:.10f}")
            print("  init A:")
            print(pd.DataFrame(A0, index=dim_names, columns=dim_names).to_string(float_format=lambda x: f"{x:.6f}"))

            fun = lambda th: _objective_unconstrained(th, times_list, d=d, stability_rho_max=stability_rho_max)
            jac = lambda th: _objective_unconstrained_with_grad(th, times_list, d=d, stability_rho_max=stability_rho_max)[1]

            res = minimize(
                fun=fun, x0=theta0, method="L-BFGS-B", jac=jac,
                options={"maxiter": 800, "maxfun": 8000, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
            )

            p_final = _from_unconstrained(res.x, d=d)
            if (not np.all(np.isfinite(p_final))):
                raise FloatingPointError("non-finite final parameters")
            mu_f, alpha_f, beta_f = _unpack_theta(p_final, d)
            A_f = _branching_matrix(alpha_f, beta_f)
            rho_f = _spectral_radius(A_f)
            grad_f = _gradient_norm_at_params(p_final, times_list, d=d, stability_rho_max=stability_rho_max)
            if not (np.isfinite(res.fun) and np.isfinite(rho_f) and np.isfinite(grad_f) and np.all(np.isfinite(A_f))):
                raise FloatingPointError("non-finite objective/summary values")

            print(f"  done success = {bool(res.success)}")
            print(f"  done fun     = {float(res.fun):.10f}")
            print(f"  done nit     = {getattr(res, 'nit', -1)}")
            print(f"  done nfev    = {getattr(res, 'nfev', -1)}")
            print(f"  done njev    = {getattr(res, 'njev', -1)}")
            print(f"  done rho(A)  = {rho_f:.10f}")
            print(f"  ||grad||_final= {grad_f:.10f}")
            print(f"  message      = {str(res.message).replace(chr(10), ' | ')}")

            rec = _record_from_result(i, seed, x0, p_final, res, times_list, d=d, dim_names=dim_names)
            records.append(rec)
            final_params_by_start[int(i)] = p_final

            if np.isfinite(res.fun) and res.fun < best_any_val:
                best_any_val = float(res.fun); best_any = (res, p_final)
            if bool(res.success) and np.isfinite(res.fun) and res.fun < best_success_val:
                best_success_val = float(res.fun); best_success = (res, p_final)
        except Exception as e:
            print(f"  start failed : {e}")
            continue

    if best_success is None and best_any is None:
        raise RuntimeError(f"Optimization failed for {label}: no valid starts produced finite results.")

    records_df = pd.DataFrame(records).sort_values(["fun", "nit", "start_id"]).reset_index(drop=True)
    records_df["healthy"] = records_df.apply(lambda r: _is_healthy_record(r.to_dict(), dim_names=dim_names), axis=1)
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

    mu, alpha, beta = _unpack_theta(p, d)
    A = _branching_matrix(alpha, beta)
    rho = _spectral_radius(A)
    grad_norm = _gradient_norm_at_params(p, times_list, d=d, stability_rho_max=stability_rho_max)

    if verbose:
        print("\n" + "=" * 70)
        print(f"4D Hawkes fit — {label}")
        print(f"selection_mode = {selection_mode}")
        print(f"||grad|| = {grad_norm:.10f}")
        print("\nBranching matrix A (rows=target, cols=source):")
        print(pd.DataFrame(A, index=dim_names, columns=dim_names).to_string(float_format=lambda x: f"{x:.6f}"))
        print(f"\nSpectral radius rho(A) = {rho:.10f}")
        print("=" * 70)
        show_cols = ["start_id", "success", "healthy", "fun", "nit", "nfev", "njev", "grad_norm_final", "rho"]
        print("\nTop optimization runs:")
        print(records_df[show_cols].head(min(12, len(records_df))).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print(f"\nHealthy selection: {healthy_mode}")
        if healthy_summary is not None:
            print(pd.DataFrame(healthy_summary["A"], index=dim_names, columns=dim_names).to_string(float_format=lambda x: f"{x:.6f}"))

    return {
        "mu": mu, "alpha": alpha, "beta": beta, "A": A, "rho": rho,
        "success": bool(getattr(selected_res, "success", False)),
        "selection_mode": selection_mode,
        "healthy_selection_mode": healthy_mode,
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
def _annot_color(v, vmax):
    return "white" if vmax > 0 and v > 0.62 * vmax else "black"


def plot_branching_heatmap(A, dim_names=None, title="Branching matrix heatmap", filename=None):
    A = np.asarray(A, dtype=float)
    dim_names = dim_names or DIM_NAMES
    d = A.shape[0]
    vmax = max(np.nanmax(A), 1e-8)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(A, aspect="equal", cmap="Reds", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Source")
    ax.set_ylabel("Target")
    ax.set_xticks(range(d))
    ax.set_yticks(range(d))
    ax.set_xticklabels(dim_names, rotation=45, ha="right")
    ax.set_yticklabels(dim_names)

    for i in range(d):
        for j in range(d):
            val = A[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9,
                    color=_annot_color(val, vmax))

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    if filename is not None:
        _save_current_fig(filename)


def plot_cross_stock_healthy_summary(results_dict, outdir=PLOT_DIR):
    valid = {k: v for k, v in results_dict.items() if v is not None}
    if not valid:
        return

    tickers = [t for t in STOCKS if t in valid]
    vmax = max(float(np.nanmax(valid[t]["healthy_best"]["A"])) for t in tickers)
    vmax = max(vmax, 1e-8)

    fig, axes = plt.subplots(1, len(tickers), figsize=(22, 5.8))
    if len(tickers) == 1:
        axes = [axes]

    im = None
    for ax, ticker in zip(axes, tickers):
        A = np.asarray(valid[ticker]["healthy_best"]["A"], dtype=float)
        im = ax.imshow(A, aspect="equal", cmap="Reds", vmin=0.0, vmax=vmax)
        ax.set_title(f"{ticker}\nhealthy", fontweight="bold")
        ax.set_xticks(range(D))
        ax.set_yticks(range(D))
        ax.set_xticklabels(DIM_NAMES, rotation=45, ha="right")
        ax.set_yticklabels(DIM_NAMES)
        for i in range(D):
            for j in range(D):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color=_annot_color(A[i, j], vmax))
        ax.grid(False)

    fig.subplots_adjust(right=0.92, wspace=0.32)
    cax = fig.add_axes([0.935, 0.18, 0.012, 0.64])
    fig.colorbar(im, cax=cax)
    fig.suptitle("4D Hawkes branching matrices — healthy best", fontweight="bold", y=0.98)
    path = os.path.join(outdir, "robust_cross_stock_4d_healthy_1x5.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# =============================================================================
# RUNNER
# =============================================================================
def run_directional_price_order_4d(df, ticker="AMZN", make_plots=True, verbose=True, use_time_since_open=True,
                                   n_starts=12, base_seed=42):
    up_t, dn_t = extract_up_down_midprice_times(df, use_time_since_open=use_time_since_open)
    mo_buy, mo_sell = extract_directional_events(df, use_time_since_open=use_time_since_open)

    print("\n" + "=" * 80)
    print(f"4D price+order study — {ticker}")
    print(f"UP events      : {len(up_t)}")
    print(f"DOWN events    : {len(dn_t)}")
    print(f"BUY_MO events  : {len(mo_buy)}")
    print(f"SELL_MO events : {len(mo_sell)}")
    print("=" * 80)

    min_count = min(len(up_t), len(dn_t), len(mo_buy), len(mo_sell))
    if min_count < 10:
        print(f"  ⚠  Not enough events across the 4 dimensions for {ticker}.")
        return None

    seed = _ticker_seed(ticker, base_seed=base_seed)
    print(f"  Using ticker-specific seed = {seed}")

    res4 = fit_4d_hawkes(
        [up_t, dn_t, mo_buy, mo_sell],
        label=f"{ticker} 4D (UP,DOWN,BUY_MO,SELL_MO)",
        dim_names=DIM_NAMES,
        n_starts=n_starts,
        verbose=verbose,
        seed=seed,
    )

    chosen = res4["healthy_best"] if res4.get("healthy_best") is not None else res4["raw_best"]
    A = chosen["A"]
    print("\n  Interpretation (healthy best priority)")
    for i, target in enumerate(DIM_NAMES):
        strongest_j = int(np.argmax(A[i, :]))
        print(f"    strongest source for {target:<7s} is {DIM_NAMES[strongest_j]:<7s}  A={A[i, strongest_j]:.6f}")

    if make_plots:
        plot_branching_heatmap(
            A,
            dim_names=DIM_NAMES,
            title=f"{ticker} 4D branching matrix A\nhealthy best",
            filename=f"robust_healthy_4d_{ticker}.png",
        )

    res4["chosen_best"] = chosen
    return res4


def run_all_stocks_directional_asymmetry(stocks=None, data_path=DATA_PATH, start_date=START_DATE, end_date=END_DATE,
                                         make_plots=True, n_starts=12, base_seed=42):
    if stocks is None:
        stocks = STOCKS

    results = {}
    print("\n" + "=" * 80)
    print("STEP — 4D Price Moves + Directional Market Orders")
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
            res = run_directional_price_order_4d(
                df, ticker=ticker, make_plots=make_plots, verbose=True,
                use_time_since_open=True, n_starts=n_starts, base_seed=base_seed,
            )
            results[ticker] = res
        except Exception as e:
            print(f"  ✗ Failed for {ticker}: {e}")
            results[ticker] = None

    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        print("\n" + "=" * 80)
        print("FINAL CROSS-STOCK 4D SUMMARY (healthy best priority)")
        print("=" * 80)
        rows = []
        for ticker in STOCKS:
            if ticker in valid:
                chosen = valid[ticker]["chosen_best"]
                A = chosen["A"]
                rows.append({
                    "Ticker": ticker,
                    "rho(A)": float(chosen["rho"]),
                    "UP<-UP": float(A[0, 0]),
                    "DOWN<-DOWN": float(A[1, 1]),
                    "BUY<-BUY": float(A[2, 2]),
                    "SELL<-SELL": float(A[3, 3]),
                })
            else:
                rows.append({
                    "Ticker": ticker,
                    "rho(A)": np.nan,
                    "UP<-UP": np.nan,
                    "DOWN<-DOWN": np.nan,
                    "BUY<-BUY": np.nan,
                    "SELL<-SELL": np.nan,
                })
        summary_df = pd.DataFrame(rows)
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        if make_plots:
            plot_cross_stock_healthy_summary(valid, outdir=PLOT_DIR)

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

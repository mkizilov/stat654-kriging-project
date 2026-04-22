"""Spatio-temporal kriging support library.

Implements (from scratch, using only numpy/scipy):
  * Empirical semivariogram and empirical space-time semivariogram.
  * Four theoretical variogram models: nugget, spherical, exponential, gaussian.
  * Sum-metric space-time variogram fitting following Kuo et al. (2021)
    gamma(h,u) = nugget * 1{h>0 OR u>0}
               + gamma_s(h) + gamma_t(u)
               + gamma_joint( sqrt(h^2 + (k*u)^2) )
  * Ordinary kriging and ordinary space-time kriging via the standard
    augmented linear system.

The library is used by the project notebook; docstrings document inputs/outputs
because those functions are called from outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km for (lat, lon) pairs in degrees."""
    R = 6371.0088
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat = lat2r - lat1r
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def pairwise_haversine_km(lats, lons):
    """n x n Haversine distance matrix in km."""
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    return haversine_km(lats[:, None], lons[:, None], lats[None, :], lons[None, :])


def spherical(h, sill, rng, nugget=0.0):
    h = np.asarray(h, dtype=float)
    x = np.clip(h / max(rng, 1e-12), 0.0, 1.0)
    out = np.where(h <= rng, sill * (1.5 * x - 0.5 * x ** 3), sill)
    out = np.where(h > 0, out + nugget, out)
    return out


def exponential(h, sill, rng, nugget=0.0):
    h = np.asarray(h, dtype=float)
    out = sill * (1.0 - np.exp(-h / max(rng, 1e-12)))
    out = np.where(h > 0, out + nugget, out)
    return out


def gaussian(h, sill, rng, nugget=0.0):
    h = np.asarray(h, dtype=float)
    out = sill * (1.0 - np.exp(-(h / max(rng, 1e-12)) ** 2))
    out = np.where(h > 0, out + nugget, out)
    return out


MODELS: Dict[str, Callable] = {
    "spherical": spherical,
    "exponential": exponential,
    "gaussian": gaussian,
}


def empirical_variogram_1d(h_pairs, diff_pairs, n_bins=15):
    """Classical Matheron estimator of gamma(h) on a 1D distance axis.

    Parameters
    ----------
    h_pairs : (P,) array of pairwise distances (>0).
    diff_pairs : (P,) array of squared differences  (z_i - z_j)**2.
    n_bins : number of equal-count bins.

    Returns
    -------
    h_center, gamma, n_per_bin : arrays of length <= n_bins.
    """
    h = np.asarray(h_pairs, dtype=float)
    d = np.asarray(diff_pairs, dtype=float)
    mask = h > 0
    h = h[mask]
    d = d[mask]
    if h.size == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(h, qs)
    edges[0] = 0.0

    h_c, g_c, n_c = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (h > a) & (h <= b)
        if m.sum() < 2:
            continue
        h_c.append(h[m].mean())
        g_c.append(0.5 * d[m].mean())
        n_c.append(int(m.sum()))
    return np.asarray(h_c), np.asarray(g_c), np.asarray(n_c)


def empirical_st_variogram(values, dist_matrix, t_axis=None,
                           h_bins=None, u_bins=None,
                           max_pairs=200_000, rng=None):
    """Space-time empirical semivariogram on a station x time matrix.

    Parameters
    ----------
    values : (n_stations, n_times) observation matrix. NaNs are allowed.
    dist_matrix : (n_stations, n_stations) pairwise spatial distance.
    t_axis : optional (n_times,) time index (e.g. hours since t0). If None we use 0..n_times-1.
    h_bins : 1D array of spatial lag edges. If None, 12 quantile bins.
    u_bins : 1D array of temporal lag edges. If None, 10 quantile bins.
    max_pairs : cap on number of random pairs sampled (for speed).
    rng : np.random.Generator for reproducible sampling.

    Returns
    -------
    H, U, Gamma, Nc : 2D arrays with shape (len(h_bins)-1, len(u_bins)-1).
        Gamma holds semivariance estimates; Nc holds pair counts per bin.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    V = np.asarray(values, dtype=float)
    D = np.asarray(dist_matrix, dtype=float)
    n_stations, n_times = V.shape
    if t_axis is None:
        t_axis = np.arange(n_times, dtype=float)
    else:
        t_axis = np.asarray(t_axis, dtype=float)

    total_pairs = n_stations * (n_stations + 1) // 2 * (n_times * (n_times + 1) // 2)
    if total_pairs > max_pairs:
        rows_i = rng.integers(0, n_stations, size=max_pairs)
        rows_j = rng.integers(0, n_stations, size=max_pairs)
        cols_p = rng.integers(0, n_times, size=max_pairs)
        cols_q = rng.integers(0, n_times, size=max_pairs)
    else:
        idx = np.indices((n_stations, n_stations, n_times, n_times)).reshape(4, -1)
        rows_i, rows_j, cols_p, cols_q = idx

    mask = ~(np.isnan(V[rows_i, cols_p]) | np.isnan(V[rows_j, cols_q]))
    rows_i, rows_j = rows_i[mask], rows_j[mask]
    cols_p, cols_q = cols_p[mask], cols_q[mask]

    h_pairs = D[rows_i, rows_j]
    u_pairs = np.abs(t_axis[cols_p] - t_axis[cols_q])
    d_pairs = (V[rows_i, cols_p] - V[rows_j, cols_q]) ** 2
    keep = (h_pairs >= 0) & (u_pairs >= 0)
    h_pairs, u_pairs, d_pairs = h_pairs[keep], u_pairs[keep], d_pairs[keep]

    if h_bins is None:
        positive_h = h_pairs[h_pairs > 0]
        if positive_h.size == 0:
            h_bins = np.linspace(0, 1, 13)
        else:
            qs = np.linspace(0.0, 0.95, 13)
            h_bins = np.quantile(positive_h, qs)
            h_bins = np.concatenate([[0.0], h_bins[1:]])
    if u_bins is None:
        positive_u = u_pairs[u_pairs > 0]
        if positive_u.size == 0:
            u_bins = np.linspace(0, 1, 11)
        else:
            qs = np.linspace(0.0, 0.95, 11)
            u_bins = np.quantile(positive_u, qs)
            u_bins = np.concatenate([[0.0], u_bins[1:]])

    nh, nu = len(h_bins) - 1, len(u_bins) - 1
    H = np.full((nh, nu), np.nan)
    U = np.full((nh, nu), np.nan)
    Gamma = np.full((nh, nu), np.nan)
    Nc = np.zeros((nh, nu), dtype=int)
    for a in range(nh):
        if a == 0:
            h_lo = h_bins[0]
            h_mask = (h_pairs >= h_lo) & (h_pairs <= h_bins[1])
        else:
            h_mask = (h_pairs > h_bins[a]) & (h_pairs <= h_bins[a + 1])
        for b in range(nu):
            if b == 0:
                u_lo = u_bins[0]
                u_mask = (u_pairs >= u_lo) & (u_pairs <= u_bins[1])
            else:
                u_mask = (u_pairs > u_bins[b]) & (u_pairs <= u_bins[b + 1])
            m = h_mask & u_mask
            k = int(m.sum())
            if k < 3:
                continue
            H[a, b] = 0.5 * (h_bins[a] + h_bins[a + 1])
            U[a, b] = 0.5 * (u_bins[b] + u_bins[b + 1])
            Gamma[a, b] = 0.5 * d_pairs[m].mean()
            Nc[a, b] = k
    return H, U, Gamma, Nc


@dataclass
class SumMetricParams:
    """Sum-metric variogram parameters."""
    nugget: float
    s_sill: float
    s_range: float
    t_sill: float
    t_range: float
    j_sill: float
    j_range: float
    k: float
    s_model: str = "spherical"
    t_model: str = "exponential"
    j_model: str = "spherical"

    def to_vector(self):
        return np.array([self.nugget, self.s_sill, self.s_range,
                         self.t_sill, self.t_range,
                         self.j_sill, self.j_range, self.k], dtype=float)

    @staticmethod
    def from_vector(theta, s_model="spherical", t_model="exponential",
                    j_model="spherical"):
        return SumMetricParams(
            nugget=float(theta[0]),
            s_sill=float(theta[1]),
            s_range=float(theta[2]),
            t_sill=float(theta[3]),
            t_range=float(theta[4]),
            j_sill=float(theta[5]),
            j_range=float(theta[6]),
            k=float(theta[7]),
            s_model=s_model, t_model=t_model, j_model=j_model,
        )


def gamma_sum_metric(h, u, p: SumMetricParams):
    """Sum-metric semivariogram gamma(h, u).

    gamma = nugget*1{h>0 or u>0} + gamma_s(h) + gamma_t(u) + gamma_j(sqrt(h^2 + (k u)^2))
    """
    h = np.asarray(h, dtype=float)
    u = np.asarray(u, dtype=float)
    gs = MODELS[p.s_model](h, p.s_sill, p.s_range)
    gt = MODELS[p.t_model](u, p.t_sill, p.t_range)
    r = np.sqrt(h ** 2 + (p.k * u) ** 2)
    gj = MODELS[p.j_model](r, p.j_sill, p.j_range)
    nug = p.nugget * ((h > 0) | (u > 0)).astype(float)
    return nug + gs + gt + gj


def fit_sum_metric(H, U, Gamma, Nc,
                   s_model="spherical",
                   t_model="exponential",
                   j_model="spherical",
                   theta0=None,
                   max_h=None, max_u=None):
    """Fit sum-metric variogram by weighted least squares on binned data.

    Returns
    -------
    SumMetricParams, scipy.optimize.OptimizeResult
    """
    mask = ~np.isnan(Gamma)
    if max_h is not None:
        mask &= (H <= max_h)
    if max_u is not None:
        mask &= (U <= max_u)
    h = H[mask].ravel()
    u = U[mask].ravel()
    g = Gamma[mask].ravel()
    w = np.sqrt(np.clip(Nc[mask].astype(float), 1, None)).ravel()

    if theta0 is None:
        s_sill_0 = float(np.nanquantile(g, 0.9)) * 0.35
        j_sill_0 = float(np.nanquantile(g, 0.9)) * 0.35
        t_sill_0 = float(np.nanquantile(g, 0.9)) * 0.25
        theta0 = np.array([
            float(np.nanmin(g)) * 0.05 if np.nanmin(g) > 0 else 0.01,
            max(s_sill_0, 0.1), float(np.nanquantile(h, 0.5)) + 1e-6,
            max(t_sill_0, 0.1), float(np.nanquantile(u, 0.5)) + 1e-6,
            max(j_sill_0, 0.1), float(np.nanquantile(h, 0.7)) + 1e-6,
            float(np.nanquantile(h, 0.5) / max(np.nanquantile(u, 0.5), 1e-6)) + 1e-6,
        ])

    lower = np.array([0.0, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4, 1e-3, 1e-4])
    upper = np.array([np.inf] * 8)

    def residuals(theta):
        p = SumMetricParams.from_vector(theta, s_model, t_model, j_model)
        ghat = gamma_sum_metric(h, u, p)
        return (ghat - g) * w

    fit = least_squares(
        residuals, theta0, bounds=(lower, upper),
        xtol=1e-10, ftol=1e-10, max_nfev=8000,
    )
    return SumMetricParams.from_vector(fit.x, s_model, t_model, j_model), fit


def ordinary_kriging(x_obs, y_obs, x_pred, gamma_func, total_sill):
    """Ordinary (spatial) kriging with a semivariogram model.

    Solves the classic augmented system:
        | Gamma  1 | |w|   |gamma_0|
        | 1^T    0 | |mu| = |1      |
    where Gamma is the n x n matrix of semivariances between observed points,
    and gamma_0 is the length-n vector of semivariances between observed and
    target points.

    Parameters
    ----------
    x_obs : (n, d) observed coordinates
    y_obs : (n,)  observed values
    x_pred : (m, d) prediction coordinates
    gamma_func : callable g(h) giving semivariance at scalar / array distance h
    total_sill : float used to convert semivariance to covariance if needed
        (unused by this function but required by the variance formula below).

    Returns
    -------
    preds : (m,) predictions
    kvar  : (m,) kriging variance
    """
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)
    n = len(y_obs)

    D = cdist(x_obs, x_obs)
    G = gamma_func(D)

    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0
    A[:n, :n] += 1e-10 * np.eye(n)

    m = len(x_pred)
    preds = np.empty(m)
    kvar = np.empty(m)

    d = cdist(x_pred, x_obs)
    g0 = gamma_func(d)
    B = np.ones((n + 1, m), dtype=float)
    B[:n, :] = g0.T
    sol = np.linalg.lstsq(A, B, rcond=None)[0]
    w = sol[:n, :]
    mu = sol[n, :]
    preds = w.T @ y_obs
    kvar = (w * g0.T).sum(axis=0) + mu
    return preds, kvar


def ordinary_st_kriging_local(X_obs, t_obs, y_obs, X_pred, t_pred,
                              p: SumMetricParams, n_neighbors=120):
    """Local moving-neighborhood ordinary space-time kriging.

    For each prediction point, use only the n_neighbors closest observations
    in the metric distance sqrt(h^2 + (k*u)^2). Solves a small (n_neighbors+1)
    system per prediction point. This is standard practice when the full
    kriging system is too large.
    """
    X_obs = np.asarray(X_obs, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    n = len(y_obs)
    m = len(t_pred)
    k_aniso = max(p.k, 1e-6)

    H_all = cdist(X_pred, X_obs)
    U_all = np.abs(t_pred[:, None] - t_obs[None, :])
    joint_dist = np.sqrt(H_all ** 2 + (k_aniso * U_all) ** 2)

    n_nb = min(n_neighbors, n)
    neighbors = np.argpartition(joint_dist, n_nb - 1, axis=1)[:, :n_nb]

    preds = np.empty(m)
    kvar = np.empty(m)

    for i in range(m):
        idx = neighbors[i]
        X_i = X_obs[idx]
        t_i = t_obs[idx]
        y_i = y_obs[idx]

        H_ii = cdist(X_i, X_i)
        U_ii = np.abs(t_i[:, None] - t_i[None, :])
        G = gamma_sum_metric(H_ii, U_ii, p)

        ni = n_nb
        A = np.zeros((ni + 1, ni + 1), dtype=float)
        A[:ni, :ni] = G + 1e-10 * np.eye(ni)
        A[:ni, ni] = 1.0
        A[ni, :ni] = 1.0

        h0 = H_all[i, idx]
        u0 = U_all[i, idx]
        g0 = gamma_sum_metric(h0, u0, p)
        b = np.ones(ni + 1)
        b[:ni] = g0

        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        w = sol[:ni]
        mu = sol[ni]
        preds[i] = float(w @ y_i)
        kvar[i] = float(w @ g0 + mu)
    return preds, kvar


def ordinary_st_kriging(X_obs, t_obs, y_obs, X_pred, t_pred, p: SumMetricParams,
                        total_sill=None):
    """Ordinary space-time kriging with a sum-metric variogram.

    Parameters
    ----------
    X_obs : (n, 2) spatial coordinates of observations (e.g. [x_km, y_km]).
    t_obs : (n,) time index of observations (hours).
    y_obs : (n,) observed values.
    X_pred, t_pred : analogous for prediction points (m of them).
    p : SumMetricParams fitted variogram.

    Returns
    -------
    preds : (m,) predictions
    kvar  : (m,) kriging variance
    """
    X_obs = np.asarray(X_obs, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    t_pred = np.asarray(t_pred, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)

    n = len(y_obs)
    m = len(t_pred)

    H_obs = cdist(X_obs, X_obs)
    U_obs = np.abs(t_obs[:, None] - t_obs[None, :])
    G = gamma_sum_metric(H_obs, U_obs, p)

    A = np.zeros((n + 1, n + 1), dtype=float)
    A[:n, :n] = G
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    A[n, n] = 0.0
    A[:n, :n] += 1e-10 * np.eye(n)

    H0 = cdist(X_pred, X_obs)
    U0 = np.abs(t_pred[:, None] - t_obs[None, :])
    g0 = gamma_sum_metric(H0, U0, p)

    B = np.ones((n + 1, m), dtype=float)
    B[:n, :] = g0.T
    sol = np.linalg.lstsq(A, B, rcond=None)[0]
    w = sol[:n, :]
    mu = sol[n, :]
    preds = w.T @ y_obs
    kvar = (w * g0.T).sum(axis=0) + mu
    return preds, kvar


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def leave_one_station_out(X, t_hours, Y, p: SumMetricParams):
    """Leave-one-station-out CV: for each station, predict all its hourly temperatures
    using the other stations.

    X : (n_stations, 2) spatial coords
    t_hours : (n_times,) time axis in hours
    Y : (n_stations, n_times) observations (NaNs allowed)

    Returns
    -------
    Y_hat : (n_stations, n_times) predicted
    """
    n, T = Y.shape
    Y_hat = np.full_like(Y, fill_value=np.nan)
    for i in range(n):
        keep = np.arange(n) != i
        Xk = X[keep]
        Yk = Y[keep]

        obs_rows = []
        obs_vals = []
        obs_t = []
        for j, t in enumerate(t_hours):
            v = Yk[:, j]
            mok = ~np.isnan(v)
            if mok.any():
                obs_rows.append(Xk[mok])
                obs_vals.append(v[mok])
                obs_t.append(np.full(mok.sum(), t))
        if not obs_rows:
            continue
        X_obs = np.vstack(obs_rows)
        y_obs = np.concatenate(obs_vals)
        t_obs = np.concatenate(obs_t)

        X_pred = np.tile(X[i:i + 1], (T, 1))
        t_pred = np.asarray(t_hours, dtype=float)
        preds, _ = ordinary_st_kriging(X_obs, t_obs, y_obs, X_pred, t_pred, p)
        Y_hat[i, :] = preds
    return Y_hat

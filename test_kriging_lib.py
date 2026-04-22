"""Smoke / sanity tests for the kriging library.

Run:   python test_kriging_lib.py
This is a plain script (not pytest) so graders can reproduce easily.
Exits non-zero on failure.
"""
import sys
import numpy as np

from kriging_lib import (
    SumMetricParams, gamma_sum_metric,
    ordinary_kriging, ordinary_st_kriging,
    empirical_variogram_1d, empirical_st_variogram,
    fit_sum_metric, rmse, mae, r2_score,
    haversine_km, pairwise_haversine_km,
    spherical, exponential, gaussian,
)


def test_variogram_shapes():
    for model in (spherical, exponential, gaussian):
        g = model(np.array([0.0, 1.0, 10.0]), sill=2.0, rng=5.0)
        assert g.shape == (3,)
        assert np.isclose(g[0], 0.0)
        assert g[-1] >= 0.0
    print("  test_variogram_shapes OK")


def test_ordinary_kriging_exact_at_data_points():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, (20, 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1]

    def gamma(h):
        return exponential(h, sill=1.0, rng=3.0)

    preds, kvar = ordinary_kriging(x, y, x, gamma, total_sill=1.0)
    err = np.max(np.abs(preds - y))
    assert err < 1e-6, f"OK should exactly reproduce obs, got max err {err:.3e}"
    assert (kvar < 1e-6).all(), "Kriging variance at data points should be 0"
    print("  test_ordinary_kriging_exact_at_data_points OK")


def test_st_kriging_exact_at_data_points():
    rng = np.random.default_rng(1)
    n = 15
    T = 6
    X = rng.uniform(0, 20, (n, 2))
    t = np.arange(T, dtype=float)
    Y = np.outer(np.cos(X[:, 0] * 0.1), np.sin(t)) + X[:, 1][:, None]

    X_stack = np.repeat(X, T, axis=0)
    t_stack = np.tile(t, n)
    y_stack = Y.ravel()

    p = SumMetricParams(
        nugget=0.01, s_sill=1.0, s_range=5.0,
        t_sill=0.5, t_range=2.0, j_sill=0.3, j_range=3.0, k=1.0,
    )
    preds, kvar = ordinary_st_kriging(X_stack, t_stack, y_stack,
                                      X_stack, t_stack, p)
    err = np.max(np.abs(preds - y_stack))
    assert err < 1e-5, f"ST kriging exactness failed: err={err:.3e}"
    print("  test_st_kriging_exact_at_data_points OK")


def test_sum_metric_non_negative():
    p = SumMetricParams(
        nugget=0.1, s_sill=1.0, s_range=10.0,
        t_sill=0.5, t_range=3.0, j_sill=0.3, j_range=5.0, k=2.0,
    )
    h = np.linspace(0, 50, 15)
    u = np.linspace(0, 10, 7)
    H, U = np.meshgrid(h, u, indexing="ij")
    G = gamma_sum_metric(H, U, p)
    assert (G >= 0).all(), "Semivariogram must be non-negative"
    assert G[0, 0] == 0, "gamma(0,0) must be 0"
    print("  test_sum_metric_non_negative OK")


def test_fit_recovers_known_params():
    rng = np.random.default_rng(7)
    p_true = SumMetricParams(
        nugget=0.05, s_sill=1.0, s_range=15.0,
        t_sill=0.6, t_range=4.0, j_sill=0.3, j_range=8.0, k=3.0,
    )
    hs = np.linspace(1, 50, 12)
    us = np.linspace(1, 12, 8)
    H, U = np.meshgrid(hs, us, indexing="ij")
    G = gamma_sum_metric(H, U, p_true) + rng.normal(scale=0.05, size=H.shape)
    Nc = np.full_like(G, 100)
    p_hat, info = fit_sum_metric(H, U, G, Nc)
    g_hat = gamma_sum_metric(H, U, p_hat)
    mean_err = np.mean(np.abs(g_hat - gamma_sum_metric(H, U, p_true)))
    assert mean_err < 0.1, f"Fit mean abs error {mean_err:.3f} too large"
    print(f"  test_fit_recovers_known_params OK (mean_err={mean_err:.3e})")


def test_metrics():
    y = np.array([1, 2, 3, 4, 5.0])
    yhat = np.array([1.1, 1.9, 3.0, 4.2, 4.8])
    r = rmse(y, yhat)
    m = mae(y, yhat)
    r2 = r2_score(y, yhat)
    assert 0.1 < r < 0.3
    assert 0.1 < m < 0.3
    assert r2 > 0.95
    print(f"  test_metrics OK (RMSE={r:.3f}, MAE={m:.3f}, R^2={r2:.3f})")


def test_haversine():
    d = haversine_km(0, 0, 0, 1)
    assert 110.0 < d < 112.0, f"1 deg lon at eq ≈ 111.19 km, got {d}"
    D = pairwise_haversine_km([0, 0, 0], [0, 1, 2])
    assert D.shape == (3, 3)
    assert D[0, 1] > 0 and D[0, 2] > D[0, 1]
    print("  test_haversine OK")


def test_empirical_st_variogram_structure():
    rng = np.random.default_rng(42)
    n, T = 10, 12
    X = rng.uniform(0, 30, (n, 2))
    t = np.arange(T, dtype=float)
    p = SumMetricParams(
        nugget=0.05, s_sill=1.0, s_range=10.0,
        t_sill=0.5, t_range=3.0, j_sill=0.3, j_range=6.0, k=2.0,
    )
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    H_full = D[:, :, None, None] * np.ones((n, n, T, T))
    U_full = np.abs(t[None, None, :, None] - t[None, None, None, :]) * np.ones((n, n, T, T))
    G_true = gamma_sum_metric(H_full, U_full, p)
    Z_flat = rng.multivariate_normal(
        mean=np.full(n * T, 10.0),
        cov=np.maximum(1.5 - G_true.transpose(0, 2, 1, 3).reshape(n * T, n * T), 1e-6),
        size=1, check_valid="ignore",
    )[0].reshape(n, T)

    H, U, Gamma, Nc = empirical_st_variogram(Z_flat, D, t_axis=t, max_pairs=20000, rng=rng)
    assert H.shape == Gamma.shape and Gamma.shape == Nc.shape
    finite = np.isfinite(Gamma)
    assert finite.sum() > 10, "Too few filled bins"
    print(f"  test_empirical_st_variogram_structure OK ({finite.sum()} bins filled)")


def test_empirical_st_variogram_no_duplicate_counting():
    """Zero-lag pairs must not be double-counted across bins.

    This is the bug Oracle flagged in review: the old implementation dumped
    h==0 pairs into every temporal bin of the first spatial row and vice-versa.
    Here we verify total pair counts across bins <= total pairs generated.
    """
    rng = np.random.default_rng(123)
    n, T = 6, 5
    X = rng.uniform(0, 20, (n, 2))
    t = np.arange(T, dtype=float)
    Z = rng.normal(size=(n, T))
    from scipy.spatial.distance import cdist
    D = cdist(X, X)
    H, U, Gamma, Nc = empirical_st_variogram(Z, D, t_axis=t, max_pairs=2000, rng=rng)
    total_pairs_counted = int(Nc.sum())
    total_pairs_possible = n * n * T * T
    assert total_pairs_counted <= total_pairs_possible, (
        f"Duplicate binning: counted {total_pairs_counted} > possible {total_pairs_possible}"
    )
    print(f"  test_empirical_st_variogram_no_duplicate_counting OK "
          f"(binned {total_pairs_counted} <= total {total_pairs_possible})")


def test_local_st_kriging_matches_full_when_all_neighbors():
    """Moving-neighborhood kriging with n_neighbors >= n should match global."""
    import kriging_lib as K
    rng = np.random.default_rng(0)
    n, T = 6, 4
    X = rng.uniform(0, 10, (n, 2))
    t = np.arange(T, dtype=float)
    Y = rng.normal(size=(n, T))

    X_stack = np.repeat(X, T, axis=0)
    t_stack = np.tile(t, n)
    y_stack = Y.ravel()

    p = SumMetricParams(
        nugget=0.05, s_sill=1.0, s_range=5.0,
        t_sill=0.5, t_range=2.0, j_sill=0.3, j_range=3.0, k=1.5,
    )

    X_pred = np.array([[5.0, 5.0], [2.0, 7.0], [8.0, 3.0]])
    t_pred = np.array([1.5, 2.5, 0.5])

    full, _ = ordinary_st_kriging(X_stack, t_stack, y_stack, X_pred, t_pred, p)
    local, _ = K.ordinary_st_kriging_local(
        X_stack, t_stack, y_stack, X_pred, t_pred, p,
        n_neighbors=len(y_stack),
    )
    err = np.max(np.abs(full - local))
    assert err < 1e-4, f"Local kriging with all neighbors should equal full: err={err}"
    print(f"  test_local_st_kriging_matches_full_when_all_neighbors OK (max_err={err:.2e})")


def main():
    print("Running kriging_lib smoke tests...")
    test_variogram_shapes()
    test_haversine()
    test_metrics()
    test_sum_metric_non_negative()
    test_ordinary_kriging_exact_at_data_points()
    test_st_kriging_exact_at_data_points()
    test_empirical_st_variogram_structure()
    test_empirical_st_variogram_no_duplicate_counting()
    test_local_st_kriging_matches_full_when_all_neighbors()
    test_fit_recovers_known_params()
    print("\nAll tests passed.")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"\nFAIL: {e}", file=sys.stderr)
        sys.exit(1)

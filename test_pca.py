"""Standalone validation of pcaAnalysis.run_pca() against sklearn.decomposition.PCA.

Not part of the app; run directly:

    python test_pca.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pcaAnalysis import prices_to_returns, run_pca

PRICES_CSV = Path(__file__).resolve().parent / "portfolio_prices.csv"
RTOL = 1e-8
ATOL = 1e-8


def load_returns_df():
    if PRICES_CSV.exists():
        print(f"Using real data from {PRICES_CSV.name}")
        prices = pd.read_csv(PRICES_CSV, index_col="Date", parse_dates=True)
        return prices_to_returns(prices)

    print("No portfolio_prices.csv found, synthesizing random log-return data")
    rng = np.random.default_rng(0)
    tickers = [f"TICK{i}" for i in range(5)]
    dates = pd.bdate_range("2024-01-01", periods=250)
    data = rng.normal(loc=0.0003, scale=0.01, size=(len(dates), len(tickers)))
    return pd.DataFrame(data, index=dates, columns=tickers)


def main():
    returns_df = load_returns_df()
    tickers = returns_df.columns.tolist()
    R = returns_df.values
    T, N = R.shape
    print(f"Data shape: {T} observations x {N} tickers ({tickers})")

    ours = run_pca(returns_df)

    # Reference: sklearn PCA on the same (uncentered) return matrix -- PCA
    # centers internally, matching pcaAnalysis's own centering step.
    skpca = PCA(n_components=N)
    skpca.fit(R)
    sk_eigvals = skpca.explained_variance_
    sk_evr = skpca.explained_variance_ratio_
    sk_components = skpca.components_  # (N, N), row k = eigenvector for PCk

    # Reconstruct our eigenvalues from evr * total variance (trace of the
    # covariance matrix), since run_pca only returns the ratio.
    Rc = R - R.mean(axis=0)
    stock_var = Rc.var(axis=0, ddof=1)
    total_var = stock_var.sum()
    our_eigvals = ours["evr"] * total_var

    failures = []

    if not np.allclose(our_eigvals, sk_eigvals, rtol=RTOL, atol=ATOL):
        failures.append(
            f"Eigenvalues mismatch:\n  ours = {our_eigvals}\n  sklearn = {sk_eigvals}"
        )

    if not np.allclose(ours["evr"], sk_evr, rtol=RTOL, atol=ATOL):
        failures.append(
            f"Explained variance ratio mismatch:\n  ours = {ours['evr']}\n  sklearn = {sk_evr}"
        )

    our_loadings = ours["loading_matrix"]
    for k in range(N):
        sk_loading_k = sk_components[k, :] * np.sqrt(sk_eigvals[k] / stock_var)
        sign = np.sign(np.dot(our_loadings[:, k], sk_loading_k)) or 1.0
        sk_loading_k_aligned = sk_loading_k * sign
        if not np.allclose(our_loadings[:, k], sk_loading_k_aligned, rtol=RTOL, atol=ATOL):
            failures.append(
                f"Loading mismatch on PC{k + 1} (after sign alignment):\n"
                f"  ours = {our_loadings[:, k]}\n  sklearn = {sk_loading_k_aligned}"
            )

    print()
    if failures:
        print("FAIL: from-scratch PCA does not match sklearn")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("PASS: eigenvalues and loadings (up to sign) match sklearn.decomposition.PCA")
        sys.exit(0)


if __name__ == "__main__":
    main()

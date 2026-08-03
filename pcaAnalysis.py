import pandas as pd
import numpy as np


def run_pca(returns_df=None, csv_path="portfolio_prices.csv"):
    """Run PCA on a returns DataFrame.

    If returns_df is None, prices are read from csv_path, NaN dates are
    dropped, and log returns are computed to build the returns DataFrame.
    If returns_df is passed in directly, it is used as-is (it must already
    be a returns DataFrame, not raw prices) and the file read / log-return
    step is skipped entirely.
    """
    if returns_df is None:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        df = df.dropna() #remove empty dates

        #Logarithmic Returns, R_tj = ln(P_tj / P(t-1)j)
        returns_df = np.log(df / df.shift(1)).dropna()

    tickers = returns_df.columns.tolist()
    R = returns_df.values
    T, N = R.shape

    #Centering data
    R_bar = R.mean(axis=0)
    Rc = R - R_bar

    #Covariance Matrix Σ = 1/(T-1) * Rc^T Rc
    Sigma = (Rc.T @ Rc) / (T-1)

    #Eigendecomposition
    eigvals, eigvects = np.linalg.eigh(Sigma)

    #Sort Eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    Q = eigvects[:, idx]

    #flip eigenvector so largest laoding is positive
    for k in range(N):
        max_idx = np.argmax(np.abs(Q[:, k]))
        if Q[max_idx, k] < 0:
            Q[:, k] *= -1


    Z = Rc @ Q
    evr = eigvals / eigvals.sum()

    #Loading Matrix corr(Rj, PCk)
    stock_var = Rc.var(axis=0, ddof = 1)

    loading_matrix = Q * np.sqrt(eigvals / stock_var[:, None])

    return {
        "tickers": tickers,
        "evr": evr,
        "loading_matrix": loading_matrix,
    }

if __name__ == "__main__":
    results = run_pca()
    print("Tickers:", results["tickers"])
    print("Explained variance ratio:", results["evr"])
    print("Loading matrix (tickers x PCs):")
    print(results["loading_matrix"])

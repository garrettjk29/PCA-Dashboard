import pandas as pd
import numpy as np


def prices_to_returns(prices_df):
    
    prices_df = prices_df.dropna() #keep only dates every ticker has data for

    #Logarithmic Returns, R_tj = ln(P_tj / P(t-1)j)
    return np.log(prices_df / prices_df.shift(1)).dropna()


def build_returns_df(prices):
    
    combined = pd.DataFrame(prices)
    combined.index.name = "Date"
    return prices_to_returns(combined)


def run_pca(returns_df=None, csv_path="portfolio_prices.csv"):
   
    if returns_df is None:
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        returns_df = prices_to_returns(df)

    tickers = returns_df.columns.tolist()
    R = returns_df.values
    T, N = R.shape

    if T <= N:
        raise ValueError(
            f"Not enough observations for PCA: {T} observations for {N} tickers. "
            f"Need more observations than tickers (T > N) -- widen the date window "
            f"or use fewer tickers."
        )

    #Centering data
    R_bar = R.mean(axis=0)
    Rc = R - R_bar

    #Guard against near-zero-variance tickers, which would produce a divide-by-zero
    #(NaN) in the loading matrix below.
    stock_var = Rc.var(axis=0, ddof=1)
    zero_var = stock_var < 1e-12
    if zero_var.any():
        offenders = ", ".join(t for t, bad in zip(tickers, zero_var) if bad)
        raise ValueError(
            f"Near-zero variance in the selected window for: {offenders}. "
            f"PCA loadings are undefined when a ticker's returns don't vary."
        )

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
    loading_matrix = Q * np.sqrt(eigvals / stock_var[:, None])

    return {
        "tickers": tickers,
        "evr": evr,
        "loading_matrix": loading_matrix,
        "pc_scores": Z,
    }

if __name__ == "__main__":
    results = run_pca()
    print("Tickers:", results["tickers"])
    print("Explained variance ratio:", results["evr"])
    print("Loading matrix (tickers x PCs):")
    print(results["loading_matrix"])

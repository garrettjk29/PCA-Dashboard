# PCA Dashboard

A desktop tool for exploring the principal component structure of a portfolio's daily returns. Fetches price data for a set of tickers, computes PCA on log returns **from scratch** (no `sklearn.decomposition.PCA`), and visualizes the loading matrix and explained variance in real time.

## Why this exists

Off-the-shelf PCA (`sklearn.PCA().fit()`) hides the linear algebra. This project implements the eigendecomposition manually — covariance construction, eigenvector sign convention, and correlation-based loadings — to demonstrate the underlying math. 

## What it does

1. User enters a ticker and a timeframe in the dashboard.
2. `getDataFunction.py` fetches daily price history for that ticker.
3. Prices are cached to disk (`storage.py`) so the portfolio persists across sessions.
4. Once 2+ tickers are loaded, `pcaAnalysis.py` runs PCA on the log-return matrix and returns the loading matrix and explained variance ratios.
5. `dashboard.py` renders both as live-updating matplotlib figures embedded in a Tkinter GUI, and re-fetches data on a background thread so the UI never blocks.

## The math

**1. Log returns.** For price series $P_{t,j}$ of asset $j$:

$$R_{t,j} = \ln\left(\frac{P_{t,j}}{P_{t-1,j}}\right)$$

**2. Centering.** Subtract the sample mean of each asset's return series:

$$R^c = R - \bar R$$

**3. Covariance matrix.** With $T$ observations:

$$\Sigma = \frac{1}{T-1} (R^c)^\top R^c$$

**4. Eigendecomposition.** Solve $\Sigma Q = Q \Lambda$, sort eigenvalues descending, and fix sign ambiguity by flipping each eigenvector so its largest-magnitude loading is positive (eigenvectors are only defined up to sign, and an arbitrary sign flip between runs makes loadings hard to compare — this convention keeps them stable).

**5. Principal component scores.**

$$Z = R^c Q$$

**6. Explained variance ratio.**

$$\text{EVR}_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

**7. Loading matrix.** The raw eigenvectors $Q$ tell you the *linear combination* that forms each component, but not how strongly each asset actually correlates with it. The loading reported here is the correlation between asset $j$'s returns and principal component $k$:

$$\text{loading}_{j,k} = \text{corr}(R_j, \text{PC}_k) = Q_{j,k} \sqrt{\frac{\lambda_k}{\text{Var}(R_j)}}$$

This is the standard PCA loading definition (as opposed to raw eigenvector weights), and it's what's plotted in the dashboard's heatmap — values are bounded in $[-1, 1]$ and directly interpretable as correlations.


## Project structure

| File | Responsibility |
|---|---|
| `dashboard.py` | Tkinter GUI: search/add tickers, async fetch via background thread + queue, renders loading-matrix heatmap and explained-variance bar chart |
| `getDataFunction.py` | Fetches daily price history for a given ticker/timeframe |
| `pcaAnalysis.py` | Pure PCA logic — covariance, eigendecomposition, loadings. No UI dependencies, so it can be run or tested standalone |
| `storage.py` | Persists fetched prices to `portfolio_prices.csv` so the portfolio survives restarts |

Data flows one direction: `dashboard.py` → `getDataFunction.py` → `storage.py` (cache) → `pcaAnalysis.py` (reads the cache) → back to `dashboard.py` for rendering. Keeping `pcaAnalysis.py` free of UI code means the math can be verified independently of the GUI.

## Running it

```bash
pip install -r requirements.txt
python dashboard.py
```

Add at least two tickers via the search bar to trigger a PCA run — the loading matrix and variance chart update automatically as tickers are added or removed.

## Known limitations

- No handling yet for tickers with mismatched date ranges beyond a basic dropna, partial-history assets currently just shrink the usable date window.
- If the selected date window has fewer observations than tickers, PCA is underdetermined and `run_pca()` raises a `ValueError` instead of running.
- If any ticker has near-zero return variance in the selected window, `run_pca()` raises a `ValueError` naming that ticker instead of producing NaN loadings.


# Nifty 50 one-day-ahead 95% VaR backtest

Backtests two one-day-ahead 95% Value-at-Risk models on Nifty 50 daily log returns (2 Jan 2015 to 30 Dec 2025):

- **Historical VaR**: the empirical 95% quantile of the previous 250 daily losses.
- **Gaussian VaR**: mean + z(0.95) x standard deviation of the previous 250 daily losses (rolling, sample std).

Each day's VaR uses only the 250 losses *before* that day, so there is no look-ahead. Coverage is evaluated with the **Kupiec proportion-of-failures (POF) test**. A Jarque-Bera test on the training-period losses motivates the comparison between a non-parametric and a normal model.

## Method

1. Compute log returns from the closing prices and define losses as the negative returns.
2. Split chronologically: the first 70% is the training period, the rest is the test period (test starts 2022-09-16, 812 days).
3. For every test day, compute both VaR forecasts from the preceding 250-day window.
4. Count exceedances (loss > VaR) and apply the Kupiec test against the expected 5% rate.

## Results

| Method | Observations | Exceedances | Exceedance rate | Kupiec LR | p-value |
|---|---|---|---|---|---|
| Historical VaR | 812 | 36 | 4.43% | 0.5694 | 0.4505 |
| Gaussian VaR | 812 | 30 | 3.69% | 3.1908 | 0.0741 |

- Both models are not rejected by the Kupiec test at the 5% level.
- The Gaussian model's exceedance rate (3.69%) is below the 5% target, so it is conservative in this sample. Its p-value (0.074) is close to the rejection threshold.
- Training-period losses are strongly non-normal (Jarque-Bera p < 0.001, skewness 1.42, excess kurtosis 19.72), which is why a Gaussian assumption is a natural thing to test.

## Reproducibility

The closing prices are frozen in `nifty50_close_snapshot.csv`, so the script does not download anything and gives identical results on every run. It prints the file's SHA-256 on each run; the snapshot's hash is:

`245aac3ab0d9e18bb5c7660048394e744223dd0b392d1c96676dc7c42fd81296`

```bash
pip install -r requirements.txt
python main.py
```

`Backtesting_Notes.docx` contains the methodology notes. The results were independently reproduced by a faculty member at IIT Madras.

## Files

- `main.py`: the backtest.
- `nifty50_close_snapshot.csv`: frozen Nifty 50 closing prices.
- `requirements.txt`: pinned dependencies.
- `Backtesting_Notes.docx`: methodology notes.

## Author

Sudeep Gangula, B.Tech Mathematics & Computing, IIIT Bhopal.

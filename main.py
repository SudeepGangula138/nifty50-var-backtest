import hashlib
import platform
import numpy as np
import pandas as pd
import scipy
from scipy import stats
data_file="nifty50_close_snapshot.csv"
train_fac=0.70
window=250
p=0.05

def load_returns():
    with open(data_file,"rb") as f:
        digest=hashlib.sha256(f.read()).hexdigest()
    print("Input:",data_file,"| SHA-256:",digest)
    close=pd.read_csv(data_file,index_col="Date",parse_dates=True,float_precision="round_trip")["Close"]
    log_returns=np.log(close/close.shift(1))
    returns=log_returns.dropna()
    print("Prices:",len(close),"| returns:",len(returns))
    return returns

def historical_var(losses,t,window):
    if t<window:
        raise ValueError("Not enough historical observations.")
    past=np.asarray(losses[t-window:t],dtype=float)
    return float(np.quantile(past,1-p))

def gaussian_var(losses,t,window):
    if t<window:
        raise ValueError("Not enough historical observations.")
    historical_losses=np.asarray(losses[t-window:t],dtype=float)
    if not np.all(np.isfinite(historical_losses)):
        raise ValueError("Historical losses must be finite.")
    mu=np.mean(historical_losses)
    sigma=np.std(historical_losses,ddof=1)
    z=stats.norm.ppf(1-p)
    return float(mu+z*sigma)

def kupiec_test(exceed,p=p):
    exceed=np.asarray(exceed,dtype=bool)
    T=len(exceed)
    x=int(np.sum(exceed))
    if T==0:
        raise ValueError("Exceedance array cannot be empty")
    if not 0<p<1:
        raise ValueError("p must be between 0 and 1")
    rate=x/T
    log_likelihood_null=(x*np.log(p)+(T-x)*np.log1p(-p))
    if x==0 or x==T:
        log_likelihood_alt=0.0
    else:
        log_likelihood_alt=(x*np.log(rate)+(T-x)*np.log1p(-rate))
    LR=-2*(log_likelihood_null-log_likelihood_alt)
    p_value=float(stats.chi2.sf(LR,df=1))
    return T,x,rate,float(LR),p_value

def main():
    print("Python",platform.python_version(),"| numpy",np.__version__,"| pandas",pd.__version__,"| scipy",scipy.__version__)
    r=load_returns()
    losses=-np.asarray(r.values,dtype=float).reshape(-1)
    n=len(losses)
    split=int(n*train_fac)
    if split<window:
        raise ValueError("Training period is shorter than WINDOW. Reduce WINDOW or use more data")
    print("First date:",r.index[0].date())
    print("Last date:",r.index[-1].date())
    print("Split date (first test day):",r.index[split].date())
    train_losses=losses[:split]
    jb_stat,jb_p_value=stats.jarque_bera(train_losses)
    print("\nJarque-Bera Normality Test (Training Losses)")
    print(f"JB statistic: {jb_stat:.4f}")
    print(f"p-value: {jb_p_value:.6f}")
    if jb_p_value<0.05:
        print("Conclusion: Reject normality at the 5% significance level.")
    else:
        print("Conclusion: Insufficient evidence to reject normality at the 5% significance level.")
    hist_var,gauss_var=[],[]
    for t in range(split,n):
        hist_var.append(historical_var(losses,t,window))
        gauss_var.append(gaussian_var(losses,t,window))
    test_losses=losses[split:]
    skewness=stats.skew(train_losses)
    excess_kurtosis=stats.kurtosis(train_losses)
    print(f"Skewness (training losses): {skewness:.4f}")
    print(f"Excess kurtosis (training losses): {excess_kurtosis:.4f}")
    hist_var=np.asarray(hist_var)
    gauss_var=np.asarray(gauss_var)
    hist_exceed=test_losses>hist_var
    gauss_exceed=test_losses>gauss_var
    hist_result=kupiec_test(hist_exceed)
    gauss_result=kupiec_test(gauss_exceed)
    results=pd.DataFrame([
        {"Method":"Historical VaR","Observations":hist_result[0],"Exceedances":hist_result[1],"Exceedance Rate":hist_result[2],"Kupiec LR":hist_result[3],"p-value":hist_result[4]},
        {"Method":"Gaussian VaR","Observations":gauss_result[0],"Exceedances":gauss_result[1],"Exceedance Rate":gauss_result[2],"Kupiec LR":gauss_result[3],"p-value":gauss_result[4]},
    ])
    print("\nVaR Backtesting Results")
    print(results.to_string(index=False,float_format=lambda x:f"{x:.4f}"))

if __name__=="__main__":
    main()

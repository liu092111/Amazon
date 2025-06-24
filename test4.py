import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon, probplot

# 讀取資料
df = pd.read_csv("simulated_weibull_lifetime.csv")
data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
x = np.linspace(min(data), max(data), 200)

# 擬合三種分布
wb_params = weibull_min.fit(data, floc=0)
ln_params = lognorm.fit(data, floc=0)
ex_params = expon.fit(data, floc=0)

# 預測值
wb_pdf = weibull_min.pdf(x, *wb_params)
ln_pdf = lognorm.pdf(x, *ln_params)
ex_pdf = expon.pdf(x, *ex_params)

# 擬合品質
wb_ll = np.sum(weibull_min.logpdf(data, *wb_params))
ln_ll = np.sum(lognorm.logpdf(data, *ln_params))
ex_ll = np.sum(expon.logpdf(data, *ex_params))

# 最佳模型判斷
log_likelihoods = {
    "Weibull": wb_ll,
    "Lognormal": ln_ll,
    "Exponential": ex_ll
}
best_model = max(log_likelihoods, key=log_likelihoods.get)

# 圖 1: Histogram + PDF 疊圖
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.5, label="Histogram")
plt.plot(x, wb_pdf, label="Weibull PDF", linewidth=2)
plt.plot(x, ln_pdf, label="Lognormal PDF", linewidth=2)
plt.plot(x, ex_pdf, label="Exponential PDF", linewidth=2)
plt.title("Histogram + Fitted PDFs")
plt.xlabel("Time")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 圖 2: Empirical CDF vs Theoretical CDF
sorted_data = np.sort(data)
empirical_cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.figure(figsize=(10, 6))
plt.plot(sorted_data, empirical_cdf, marker='o', linestyle='none', label='Empirical CDF')
plt.plot(x, weibull_min.cdf(x, *wb_params), label="Weibull CDF")
plt.plot(x, lognorm.cdf(x, *ln_params), label="Lognormal CDF")
plt.plot(x, expon.cdf(x, *ex_params), label="Exponential CDF")
plt.title("Empirical vs Theoretical CDF")
plt.xlabel("Time")
plt.ylabel("CDF")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 圖 3: Log-likelihood 條形圖
plt.figure(figsize=(8, 5))
plt.bar(log_likelihoods.keys(), log_likelihoods.values(), color=['steelblue', 'orange', 'green'])
plt.title(f"Log-Likelihood Comparison (Best: {best_model})")
plt.ylabel("Log-Likelihood")
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# 圖 4: QQ plot - Weibull only (最佳模型)
plt.figure(figsize=(8, 5))
if best_model == "Weibull":
    probplot(data, dist=weibull_min, sparams=(wb_params[0], 0, wb_params[2]), plot=plt)
elif best_model == "Lognormal":
    probplot(data, dist=lognorm, sparams=(ln_params[0], 0, ln_params[2]), plot=plt)
elif best_model == "Exponential":
    probplot(data, dist=expon, sparams=(0, 1/ex_ll), plot=plt)
plt.title(f"QQ Plot of Best Fit: {best_model}")
plt.grid(True)
plt.tight_layout()
plt.show()

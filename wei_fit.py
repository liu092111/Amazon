import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon
from scipy.special import gamma

# === 載入資料 ===
#df = pd.read_csv("simulated_weibull_lifetime.csv")
#df = pd.read_csv("Sample_CSV__life_cycle_sample.csv")
df = pd.read_csv("Emperor pan and tilt test to failure.csv")
data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
x = np.linspace(min(data), max(data), 200)

# === 擬合 Weibull 分布 ===
wb_params = weibull_min.fit(data, floc=0)
wb_beta, _, wb_eta = wb_params
wb_sf = weibull_min.sf(x, *wb_params)
wb_pdf = weibull_min.pdf(x, *wb_params)
wb_ll = np.sum(weibull_min.logpdf(data, *wb_params))
wb_mttf = wb_eta * gamma(1 + 1 / wb_beta)
wb_label = rf"Weibull: $S(t) = \exp\left( -\left( \frac{{t}}{{\eta}} \right)^{{\beta}} \right)$, " \
           rf"$\beta = {wb_beta:.2f}$, $\eta = {wb_eta:.2f}$, MTTF = {wb_mttf:.2f}"

# === 擬合 Lognormal 分布 ===
ln_params = lognorm.fit(data, floc=0)
ln_sigma, _, ln_scale = ln_params
ln_mu = np.log(ln_scale)
ln_sf = lognorm.sf(x, *ln_params)
ln_pdf = lognorm.pdf(x, *ln_params)
ln_ll = np.sum(lognorm.logpdf(data, *ln_params))
ln_mttf = np.exp(ln_mu + 0.5 * ln_sigma**2)
ln_label = rf"Lognormal: $S(t) = 1 - \Phi\left( \frac{{\ln t - \mu}}{{\sigma}} \right)$, " \
           rf"$\mu = {ln_mu:.2f}$, $\sigma = {ln_sigma:.2f}$, MTTF = {ln_mttf:.2f}"
# === 擬合 Exponential 分布 ===
ex_params = expon.fit(data, floc=0)
ex_lambda = 1 / ex_params[1] if ex_params[1] != 0 else 1 / np.mean(data)
ex_sf = expon.sf(x, *ex_params)
ex_pdf = expon.pdf(x, *ex_params)
ex_ll = np.sum(expon.logpdf(data, *ex_params))
ex_mttf = 1 / ex_lambda
ex_label = rf"Exponential: $S(t) = \exp(-\lambda t)$, " \
            rf"$\lambda = {ex_lambda:.2f}$, MTTF = {ex_mttf:.2f}"

# === 決定最佳模型 ===
log_likelihoods = {
    "Weibull": wb_ll,
    "Lognormal": ln_ll,
    "Exponential": ex_ll
}
best_model = max(log_likelihoods, key=log_likelihoods.get)
best_label = f"Best Fit: {best_model}"

# === Plot 1: PDF Histogram ===
plt.figure(figsize=(8, 5))                     # 設定圖大小
plt.hist(data, bins=20, density=True, alpha=0.5, color='skyblue', edgecolor='black')
plt.xlabel("Life Cycles")                      # X 軸：壽命週期
plt.ylabel("Probability Density")              # Y 軸：機率密度
plt.title("Histogram of Lifetime Data")        # 圖標題
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# === Plot 2: PDF Comparison ===
plt.figure(figsize=(10, 6))
plt.hist(data, bins=20, density=True, alpha=0.5)
#plt.hist(data, bins=20, alpha=0.5)
plt.plot(x, wb_pdf, color='orange', label=wb_label)
plt.plot(x, ln_pdf, color='green', label=ln_label)
plt.plot(x, ex_pdf, color='red', label=ex_label)
# 加入 MTTF 標記與虛線
plt.axvline(wb_mttf, color='orange', linestyle='--', alpha=0.6)
plt.axvline(ln_mttf, color='green', linestyle='--', alpha=0.6)
plt.axvline(ex_mttf, color='red', linestyle='--', alpha=0.6)
#設座標
plt.xlabel("Time")
plt.ylabel("Density")
plt.title("PDF Distribution Comparison")
plt.legend(title=best_label, loc="best", fontsize=10)
plt.grid(True)
plt.show()

# === Plot 3: Survival Comparison ===
plt.figure(figsize=(10, 6))
plt.plot(x, wb_sf, color='orange', label=wb_label)
plt.plot(x, ln_sf, color='green', label=ln_label)
plt.plot(x, ex_sf, color='red', label=ex_label)
# 加入 MTTF 標記與虛線
plt.axvline(wb_mttf, color='orange', linestyle='--', alpha=0.6)
plt.axvline(ln_mttf, color='green', linestyle='--', alpha=0.6)
plt.axvline(ex_mttf, color='red', linestyle='--', alpha=0.6)
#設座標
plt.ylabel("Survival Probability")
plt.title("Survival Function Comparison")
plt.legend(title=f"Best Fit: {best_model}", loc="best", fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()

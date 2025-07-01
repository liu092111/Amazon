import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon
from scipy.special import gamma

# === 載入資料 ===
#df = pd.read_csv("simulated_weibull_lifetime.csv")
#df = pd.read_csv("Sample_CSV__life_cycle_sample.csv")
#df = pd.read_csv("Nanboom.csv")
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

#################################################################################
#利用蒙特卡羅輔助 大量隨機數模擬
def bootstrap_weibull_params(data, n_bootstrap=1000):
    """Bootstrap重抽樣估計參數不確定性"""
    n_samples = len(data)
    bootstrap_params = []
    
    for i in range(n_bootstrap):
        # 重抽樣
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        # 擬合Weibull
        params = weibull_min.fit(bootstrap_sample, floc=0)
        bootstrap_params.append(params)
    
    return np.array(bootstrap_params)

# 執行Bootstrap
bootstrap_results = bootstrap_weibull_params(data, 1000)
beta_samples = bootstrap_results[:, 0]
eta_samples = bootstrap_results[:, 2]

# 計算信賴區間
beta_ci = np.percentile(beta_samples, [2.5, 97.5])
eta_ci = np.percentile(eta_samples, [2.5, 97.5])

print(f"Beta 95% CI: [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
print(f"Eta 95% CI: [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")

def monte_carlo_lifetime_analysis(beta, eta, n_simulations=10000):
    """蒙特卡羅模擬生命週期分析"""
    
    # 生成大量模擬數據
    simulated_data = weibull_min.rvs(beta, scale=eta, size=n_simulations)
    
    # 計算各種可靠性指標
    percentiles = np.percentile(simulated_data, [10, 50, 90, 95, 99])
    
    results = {
        'simulated_data': simulated_data,
        'B10': percentiles[0],  # 10%失效時間
        'B50': percentiles[1],  # 中位數壽命
        'B90': percentiles[2],  # 90%失效時間
        'B95': percentiles[3],  # 95%失效時間
        'B99': percentiles[4],  # 99%失效時間
        'mean_life': np.mean(simulated_data),
        'std_life': np.std(simulated_data)
    }
    
    return results

# 執行蒙特卡羅模擬
mc_results = monte_carlo_lifetime_analysis(wb_beta, wb_eta, 50000)

print(f"B10 生命: {mc_results['B10']:.2f}")
print(f"B50 生命: {mc_results['B50']:.2f}")
print(f"B95 生命: {mc_results['B95']:.2f}")

def sensitivity_analysis(base_beta, base_eta, param_variation=0.1, n_sims=5000):
    """參數敏感度分析"""
    
    # 參數變化範圍
    beta_range = [base_beta * (1 - param_variation), base_beta * (1 + param_variation)]
    eta_range = [base_eta * (1 - param_variation), base_eta * (1 + param_variation)]
    
    results = {}
    
    # Beta參數敏感度
    for i, beta_val in enumerate(np.linspace(beta_range[0], beta_range[1], 5)):
        sim_data = weibull_min.rvs(beta_val, scale=base_eta, size=n_sims)
        results[f'beta_{i}'] = {
            'beta': beta_val,
            'eta': base_eta,
            'mttf': np.mean(sim_data),
            'B10': np.percentile(sim_data, 10)
        }
    
    # Eta參數敏感度
    for i, eta_val in enumerate(np.linspace(eta_range[0], eta_range[1], 5)):
        sim_data = weibull_min.rvs(base_beta, scale=eta_val, size=n_sims)
        results[f'eta_{i}'] = {
            'beta': base_beta,
            'eta': eta_val,
            'mttf': np.mean(sim_data),
            'B10': np.percentile(sim_data, 10)
        }
    
    return results

# 執行敏感度分析
sensitivity_results = sensitivity_analysis(wb_beta, wb_eta)

def reliability_prediction(original_data, beta, eta, target_time, n_monte_carlo=10000):
    """可靠性預測與信賴區間估算"""
    
    # 使用Bootstrap估算參數不確定性
    bootstrap_params = bootstrap_weibull_params(original_data, 1000)
    
    reliability_estimates = []
    
    for params in bootstrap_params:
        beta_boot, _, eta_boot = params
        # 計算在目標時間的可靠性
        reliability = weibull_min.sf(target_time, beta_boot, scale=eta_boot)
        reliability_estimates.append(reliability)
    
    reliability_estimates = np.array(reliability_estimates)
    
    # 計算信賴區間
    reliability_ci = np.percentile(reliability_estimates, [2.5, 50, 97.5])
    
    return {
        'reliability_median': reliability_ci[1],
        'reliability_lower': reliability_ci[0],
        'reliability_upper': reliability_ci[2],
        'all_estimates': reliability_estimates
    }

# 預測在600時間單位時的可靠性
reliability_pred = reliability_prediction(data, wb_beta, wb_eta, 15000)
print(f"在次數15000時的可靠性: {reliability_pred['reliability_median']:.3f}")
print(f"95%信賴區間: [{reliability_pred['reliability_lower']:.3f}, {reliability_pred['reliability_upper']:.3f}]")

#######################################################################################################
#視覺化
# 假設 data 是你的原始資料或模擬資料
np.random.seed(0)
data = weibull_min.rvs(3.09, scale=729.22, size=30)

def bootstrap_weibull_params(data, n_bootstrap=1000):
    n_samples = len(data)
    bootstrap_params = []
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        params = weibull_min.fit(bootstrap_sample, floc=0)
        bootstrap_params.append(params)
    return np.array(bootstrap_params)

bootstrap_results = bootstrap_weibull_params(data, 1000)
beta_samples = bootstrap_results[:, 0]
eta_samples = bootstrap_results[:, 2]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(beta_samples, bins=30, color='skyblue', edgecolor='black')
plt.title('Bootstrap Distribution of Beta')
plt.xlabel('Beta')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(eta_samples, bins=30, color='lightgreen', edgecolor='black')
plt.title('Bootstrap Distribution of Eta')
plt.xlabel('Eta')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

np.random.seed(0)
mc_simulated_data = weibull_min.rvs(3.09, scale=729.22, size=50000)
percentiles = np.percentile(mc_simulated_data, [10, 50, 90, 95, 99])

plt.figure(figsize=(10, 6))
plt.hist(mc_simulated_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
for p in percentiles:
    plt.axvline(p, color='red', linestyle='--')
    plt.text(p, plt.ylim()[1]*0.9, f'{p:.1f}', rotation=90, color='red')

plt.title('Monte Carlo Simulated Lifetime Distribution')
plt.xlabel('Lifetime')
plt.ylabel('Density')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


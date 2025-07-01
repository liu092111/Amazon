import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

df = pd.read_csv("Nanboom.csv")
data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values

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
reliability_pred = reliability_prediction(data, wb_beta, wb_eta, 600)
print(f"在時間600時的可靠性: {reliability_pred['reliability_median']:.3f}")
print(f"95%信賴區間: [{reliability_pred['reliability_lower']:.3f}, {reliability_pred['reliability_upper']:.3f}]")


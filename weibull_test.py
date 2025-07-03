import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon
from scipy.special import gamma

class LifetimeAnalyzer:
    def __init__(self, data):
        self.data = data
        self.wb_params = None
        self.ln_params = None
        self.ex_params = None
        self.wb_beta = None
        self.wb_eta = None

    def fit_distributions(self):
        self.wb_params = weibull_min.fit(self.data, floc=0)
        self.wb_beta, _, self.wb_eta = self.wb_params
        self.ln_params = lognorm.fit(self.data, floc=0)
        self.ex_params = expon.fit(self.data, floc=0)

    def calculate_mttf(self):
        wb_mttf = self.wb_eta * gamma(1 + 1 / self.wb_beta)
        ln_mu = np.log(self.ln_params[2])
        ln_sigma = self.ln_params[0]
        ln_mttf = np.exp(ln_mu + 0.5 * ln_sigma ** 2)
        ex_lambda = 1 / self.ex_params[1] if self.ex_params[1] != 0 else 1 / np.mean(self.data)
        ex_mttf = 1 / ex_lambda
        return wb_mttf, ln_mttf, ex_mttf

    def plot_pdf_comparison(self):
        x = np.linspace(min(self.data), max(self.data), 200)
        wb_pdf = weibull_min.pdf(x, *self.wb_params)
        ln_pdf = lognorm.pdf(x, *self.ln_params)
        ex_pdf = expon.pdf(x, *self.ex_params)

        wb_mttf, ln_mttf, ex_mttf = self.calculate_mttf()

        plt.figure(figsize=(10, 6))
        plt.hist(self.data, bins=20, density=True, alpha=0.5, label='Histogram')
        plt.plot(x, wb_pdf, color='orange', label=f'Weibull PDF (MTTF={wb_mttf:.1f})')
        plt.plot(x, ln_pdf, color='green', label=f'Lognormal PDF (MTTF={ln_mttf:.1f})')
        plt.plot(x, ex_pdf, color='red', label=f'Exponential PDF (MTTF={ex_mttf:.1f})')
        plt.axvline(wb_mttf, color='orange', linestyle='--')
        plt.axvline(ln_mttf, color='green', linestyle='--')
        plt.axvline(ex_mttf, color='red', linestyle='--')
        plt.xlabel('Life Cycles')
        plt.ylabel('Density')
        plt.title('PDF Comparison')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def bootstrap_weibull_params(self, n_bootstrap=1000):
        n_samples = len(self.data)
        bootstrap_params = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=n_samples, replace=True)
            params = weibull_min.fit(sample, floc=0)
            bootstrap_params.append(params)
        return np.array(bootstrap_params)

    def monte_carlo_simulation(self, n_samples=10000):
        sim_data = weibull_min.rvs(self.wb_beta, scale=self.wb_eta, size=n_samples)
        percentiles = np.percentile(sim_data, [10, 50, 90, 95, 99])
        return sim_data, percentiles

    def plot_monte_carlo_distribution(self, sim_data, percentiles):
        labels = ['B10', 'B50', 'B90', 'B95', 'B99']
        plt.figure(figsize=(10, 6))
        plt.hist(sim_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        for i, p in enumerate(percentiles):
            plt.axvline(p, color='red', linestyle='--')
            plt.text(p, plt.ylim()[1]*0.9, f'{labels[i]}: {p:.1f}', rotation=90, color='red')
        plt.title('Monte Carlo Simulated Lifetime Distribution')
        plt.xlabel('Lifetime')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def sensitivity_analysis(self, param_variation=0.1, n_sims=5000):
        results = {}
        beta_range = np.linspace(self.wb_beta * (1 - param_variation), self.wb_beta * (1 + param_variation), 5)
        eta_range = np.linspace(self.wb_eta * (1 - param_variation), self.wb_eta * (1 + param_variation), 5)
        for i, beta_val in enumerate(beta_range):
            sim_data = weibull_min.rvs(beta_val, scale=self.wb_eta, size=n_sims)
            results[f'beta_{i}'] = {'beta': beta_val, 'eta': self.wb_eta, 'mttf': np.mean(sim_data), 'B10': np.percentile(sim_data, 10)}
        for i, eta_val in enumerate(eta_range):
            sim_data = weibull_min.rvs(self.wb_beta, scale=eta_val, size=n_sims)
            results[f'eta_{i}'] = {'beta': self.wb_beta, 'eta': eta_val, 'mttf': np.mean(sim_data), 'B10': np.percentile(sim_data, 10)}
        return results

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, lognorm, expon
from scipy.special import gamma

class LifetimeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = pd.to_numeric(pd.read_csv(filepath).iloc[:, 0], errors='coerce').dropna().values
        self.x = np.linspace(min(self.data), max(self.data), 200)
        self.results = {}
        self.best_model = None

    def fit_distributions(self):
        # === Weibull ===
        wb_params = weibull_min.fit(self.data, floc=0)
        wb_beta, _, wb_eta = wb_params
        wb_pdf = weibull_min.pdf(self.x, *wb_params)
        wb_sf = weibull_min.sf(self.x, *wb_params)
        wb_ll = np.sum(weibull_min.logpdf(self.data, *wb_params))
        wb_mttf = wb_eta * gamma(1 + 1 / wb_beta)
        wb_label = rf"Weibull: $S(t)=\exp(-(\frac{{t}}{{\eta}})^{{\beta}})$, " \
                   rf"$\beta={wb_beta:.2f}$, $\eta={wb_eta:.2f}$, MTTF={wb_mttf:.2f}"

        self.results['Weibull'] = {
            'pdf': wb_pdf, 'sf': wb_sf, 'll': wb_ll,
            'mttf': wb_mttf, 'label': wb_label, 'color': 'orange'
        }
        self.weibull_params = {
            'beta': wb_beta,
            'eta': wb_eta
        }


        # === Lognormal ===
        ln_params = lognorm.fit(self.data, floc=0)
        ln_sigma, _, ln_scale = ln_params
        ln_mu = np.log(ln_scale)
        ln_pdf = lognorm.pdf(self.x, *ln_params)
        ln_sf = lognorm.sf(self.x, *ln_params)
        ln_ll = np.sum(lognorm.logpdf(self.data, *ln_params))
        ln_mttf = np.exp(ln_mu + 0.5 * ln_sigma**2)
        ln_label = rf"Lognormal: $S(t)=1-\Phi((\ln t-\mu)/\sigma)$, " \
                   rf"$\mu={ln_mu:.2f}$, $\sigma={ln_sigma:.2f}$, MTTF={ln_mttf:.2f}"

        self.results['Lognormal'] = {
            'pdf': ln_pdf, 'sf': ln_sf, 'll': ln_ll,
            'mttf': ln_mttf, 'label': ln_label, 'color': 'green'
        }

        # === Exponential ===
        ex_params = expon.fit(self.data, floc=0)
        ex_lambda = 1 / ex_params[1] if ex_params[1] != 0 else 1 / np.mean(self.data)
        ex_pdf = expon.pdf(self.x, *ex_params)
        ex_sf = expon.sf(self.x, *ex_params)
        ex_ll = np.sum(expon.logpdf(self.data, *ex_params))
        ex_mttf = 1 / ex_lambda
        ex_label = rf"Exponential: $S(t)=\exp(-\lambda t)$, " \
                   rf"$\lambda={ex_lambda:.2f}$, MTTF={ex_mttf:.2f}"

        self.results['Exponential'] = {
            'pdf': ex_pdf, 'sf': ex_sf, 'll': ex_ll,
            'mttf': ex_mttf, 'label': ex_label, 'color': 'red'
        }

        # === Determine Best Fit ===
        self.best_model = max(self.results, key=lambda k: self.results[k]['ll'])

    def plot_histogram(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.data, bins=20, density=True, alpha=0.5, color='skyblue', edgecolor='black')
        plt.xlabel("Life Cycles")
        plt.ylabel("Probability Density")
        plt.title("Histogram of Lifetime Data")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def plot_pdf_comparison(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.data, bins=20, density=True, alpha=0.5)
        for name, res in self.results.items():
            plt.plot(self.x, res['pdf'], color=res['color'], label=res['label'])
            plt.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6)
        plt.xlabel("Time")
        plt.ylabel("Density")
        plt.title("PDF Distribution Comparison")
        plt.legend(title=f"Best Fit: {self.best_model}", fontsize=10)
        plt.grid(True)
        plt.show()

    def plot_survival_comparison(self):
        plt.figure(figsize=(10, 6))
        for name, res in self.results.items():
            plt.plot(self.x, res['sf'], color=res['color'], label=res['label'])
            plt.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6)
        plt.ylabel("Survival Probability")
        plt.title("Survival Function Comparison")
        plt.legend(title=f"Best Fit: {self.best_model}", fontsize=10)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

class LifetimeAnalyzerMC:
    def __init__(self, data, beta=None, eta=None):
        self.data = np.array(data)
        self.beta = beta
        self.eta = eta

    def print_bootstrap_weibull_params(self, n_bootstrap=1000):
        n_samples = len(self.data)
        bootstrap_params = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=n_samples, replace=True)
            params = weibull_min.fit(sample, floc=0)
            bootstrap_params.append(params)
        bootstrap_params = np.array(bootstrap_params)
        beta_samples = bootstrap_params[:, 0]
        eta_samples = bootstrap_params[:, 2]
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        eta_ci = np.percentile(eta_samples, [2.5, 97.5])
        beta_mean = np.mean(beta_samples)
        eta_mean = np.mean(eta_samples)
        print("\n=== [1] Bootstrap Analysis — Parameter Estimation ===")
        print(f"Method: Resampling from original dataset (n = {n_samples}) with replacement, {n_bootstrap} iterations")
        print(f"Estimated Weibull parameters:")
        print(f"  - Beta (shape):")
        print(f"      ▸ 95% CI  : [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
        print(f"      ▸ Mean    : {beta_mean:.2f}")
        print(f"  - Eta (scale):")
        print(f"      ▸ 95% CI  : [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")
        print(f"      ▸ Mean    : {eta_mean:.2f}")

        self.beta = beta_mean
        self.eta = eta_mean

        return {
            'beta_ci': beta_ci,
            'eta_ci': eta_ci,
            'beta_mean': beta_mean,
            'eta_mean': eta_mean,
            'beta_samples': beta_samples,
            'eta_samples': eta_samples
        }

    def print_monte_carlo_lifetime(self, beta=None, eta=None, n_simulations=10000):
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        simulated_data = weibull_min.rvs(beta, scale=eta, size=n_simulations)
        percentiles = np.percentile(simulated_data, [10, 50, 90, 95, 99])
        mttf = np.mean(simulated_data)
        print("\n=== [2] Monte Carlo Simulation — Lifetime Prediction ===")
        print(f"Method: Simulated {n_simulations:,} failure times from Weibull(beta={beta:.2f}, eta={eta:.2f})")
        print(f"Predicted lifecycle percentiles:")
        print(f"  - B10 (10% fail)  : {percentiles[0]:.2f}")
        print(f"  - B50 (median)    : {percentiles[1]:.2f}")
        print(f"  - B95 (95% fail)  : {percentiles[3]:.2f}")
        print(f"  - MTTF (Average)  : {mttf:.2f}")
        return {
            'B10': percentiles[0],
            'B50': percentiles[1],
            'B95': percentiles[3],
            'MTTF': mttf,
            'simulated_data': simulated_data
        }

    def print_reliability_at_cycles(self, cycles, beta=None, eta=None, ci=0.95, n_bootstrap=1000):
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        reliabilities = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=len(self.data), replace=True)
            params = weibull_min.fit(sample, floc=0)
            b, _, e = params
            reliability = np.exp(-(cycles / e) ** b)
            reliabilities.append(reliability)
        reliabilities = np.array(reliabilities)
        lower = np.percentile(reliabilities, (1 - ci) / 2 * 100)
        upper = np.percentile(reliabilities, (1 + ci) / 2 * 100)
        median = np.median(reliabilities)
        print(f"\n=== [3] Reliability Estimation at {cycles} Cycles ===")
        print(f"Method: Bootstrap reliability curve using {n_bootstrap} resampled datasets")
        print(f"Estimated reliability at cycle = {cycles}:")
        print(f"  - Median reliability     : {median:.3f}")
        print(f"  - {int(ci*100)}% Confidence Interval : [{lower:.3f}, {upper:.3f}]")

        return {
            'reliability_median': median,
            'reliability_lower': lower,
            'reliability_upper': upper
        }

class LifetimeAnalyzerPlot:
    """
    提供 Bootstrap 參數分布與蒙特卡羅壽命分布的視覺化功能。
    """

    def __init__(self, data, beta=None, eta=None):
        self.data = np.array(data)
        self.beta = beta
        self.eta = eta

    def plot_bootstrap_weibull_params(self, n_bootstrap=1000):
        """
        繪製 Bootstrap 參數分布直方圖（Beta, Eta）
        """
        n_samples = len(self.data)
        bootstrap_params = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(self.data, size=n_samples, replace=True)
            params = weibull_min.fit(sample, floc=0)
            bootstrap_params.append(params)
        bootstrap_params = np.array(bootstrap_params)
        beta_samples = bootstrap_params[:, 0]
        eta_samples = bootstrap_params[:, 2]

        beta_mean = np.mean(beta_samples)
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])

        eta_mean = np.mean(eta_samples)
        eta_ci = np.percentile(eta_samples, [2.5, 97.5])

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.hist(beta_samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7, label='Beta Samples')
        plt.axvline(beta_mean, color='red', linestyle='-', label=f'Mean = {beta_mean:.2f}')
        plt.axvline(beta_ci[0], color='green', linestyle='--', label=f'2.5% = {beta_ci[0]:.2f}')
        plt.axvline(beta_ci[1], color='green', linestyle='--', label=f'97.5% = {beta_ci[1]:.2f}')
        plt.title('Bootstrap Distribution of Beta')
        plt.xlabel('Beta')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(eta_samples, bins=30, color='lightgreen', edgecolor='black', alpha=0.7, label='Eta Samples')
        plt.axvline(eta_mean, color='red', linestyle='-', label=f'Mean = {eta_mean:.2f}')
        plt.axvline(eta_ci[0], color='green', linestyle='--', label=f'2.5% = {eta_ci[0]:.2f}')
        plt.axvline(eta_ci[1], color='green', linestyle='--', label=f'97.5% = {eta_ci[1]:.2f}')
        plt.title('Bootstrap Distribution of Eta')
        plt.xlabel('Eta')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_monte_carlo_lifetime(self, beta=None, eta=None, n_simulations=10000):
        """
        繪製蒙特卡羅壽命分布直方圖，並標示 B10/B50/B90/B95/B99
        """
        beta = beta if beta is not None else self.beta
        eta = eta if eta is not None else self.eta
        np.random.seed(0)
        mc_simulated_data = weibull_min.rvs(beta, scale=eta, size=n_simulations)
        percentiles = np.percentile(mc_simulated_data, [10, 50, 90, 95, 99])

        plt.figure(figsize=(10, 6))
        plt.hist(mc_simulated_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')
        percentile_labels = ['B10', 'B50', 'B90', 'B95', 'B99']
        for i, p in enumerate(percentiles):
            plt.axvline(p, color='red', linestyle='--', alpha=0.7)
            y_text = plt.ylim()[1] * (0.85 - i * 0.01)
            x_text = p + (plt.xlim()[1] - plt.xlim()[0]) * 0.01
            plt.text(x_text, y_text, f'{percentile_labels[i]}: {p:.1f}', rotation=90,
                     verticalalignment='center', color='red', fontsize=9)
            #plt.text(p, plt.ylim()[1]*0.9, f'{percentile_labels[i]}: {p:.1f}', rotation=90,
                     #verticalalignment='center', color='red', fontsize=9)
        plt.title('Monte Carlo Simulated Lifetime Distribution')
        plt.xlabel('Lifetime')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
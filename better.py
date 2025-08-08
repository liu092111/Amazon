import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.stats import weibull_min, lognorm, expon, ks_2samp
from scipy.special import gamma
from statsmodels.distributions.empirical_distribution import ECDF

class LifetimeAnalyzer:
    def __init__(self, filepath):
        self.filepath = filepath
        ext = os.path.splitext(filepath)[-1].lower()

        if ext == ".xlsx":
            df = pd.read_excel(filepath, header=None)
        elif ext in [".csv", ".txt"]:
            df = pd.read_csv(filepath, header=None)
        else:
            raise ValueError("Unsupported file format. Please upload .csv, .txt, or .xlsx files.")

        # 將所有欄列合併為一維陣列，支援 row/column 格式
        flat_data = pd.to_numeric(df.values.flatten(), errors='coerce')
        self.data = flat_data[~np.isnan(flat_data)]  # 移除 NaN
        if len(self.data) == 0:
            raise ValueError("No valid numeric data found in the file.")

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
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.data, bins=30, density=True, alpha=0.5, color='skyblue', edgecolor='black')
        ax.set_xlabel("Life Cycles")
        ax.set_ylabel("Probability Density")
        ax.set_title("Histogram of Lifetime Data")
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def plot_pdf_comparison(self):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(self.data, bins=30, density=True, alpha=0.5)
        for name, res in self.results.items():
            ax.plot(self.x, res['pdf'], color=res['color'], label=res['label'])
            ax.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6)
        ax.set_xlabel("Lifetime Cycles")
        ax.set_ylabel("Frequency")
        ax.set_title("PDF Distribution Comparison")
        ax.legend(title=f"Best Fit: {self.best_model}", fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def plot_survival_comparison(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, res in self.results.items():
            ax.plot(self.x, res['sf'], color=res['color'], label=res['label'])
            ax.axvline(res['mttf'], color=res['color'], linestyle='--', alpha=0.6)
        ax.set_ylabel("Survival Probability")
        ax.set_title("Survival Function Comparison")
        ax.legend(title=f"Best Fit: {self.best_model}", fontsize=10)
        ax.grid(True)
        fig.tight_layout()
        return fig

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
        print("\n[BOLD14] === [1] Bootstrap Analysis - Parameter Estimation ===")
        print(f"Method: Resampling from original dataset (n = {n_samples}) with replacement, {n_bootstrap} iterations")
        print(f"Estimated Weibull parameters:")
        print(f"  - Beta (shape):")
        print(f"      - 95% CI  : [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]")
        print(f"      - Mean    : {beta_mean:.2f}")
        print(f"  - Eta (scale):")
        print(f"      - 95% CI  : [{eta_ci[0]:.2f}, {eta_ci[1]:.2f}]")
        print(f"      - Mean    : {eta_mean:.2f}")

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
        print("\n[BOLD14] === [2] Simulation - Lifetime Prediction ===")
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
        print(f"\n[BOLD14] === [3] Reliability Estimation at {cycles} Cycles ===")
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
    def __init__(self, data, bootstrap_result=None, mc_result=None):
        self.data = np.array(data)
        self.bootstrap_result = bootstrap_result
        self.mc_result = mc_result

    def plot_bootstrap_histograms(self):
        beta_samples = self.bootstrap_result['beta_samples']
        eta_samples = self.bootstrap_result['eta_samples']
        beta_mean = np.mean(beta_samples)
        beta_ci = np.percentile(beta_samples, [2.5, 97.5])
        eta_mean = np.mean(eta_samples)
        eta_ci = np.percentile(eta_samples, [2.5, 97.5])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(beta_samples, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axs[0].axvline(beta_mean, color='red', linestyle='-', label=f'Mean = {beta_mean:.2f}')
        axs[0].axvline(beta_ci[0], color='green', linestyle='--', label=f'2.5% = {beta_ci[0]:.2f}')
        axs[0].axvline(beta_ci[1], color='green', linestyle='--', label=f'97.5% = {beta_ci[1]:.2f}')
        axs[0].set_title('Bootstrap Distribution of Beta')
        axs[0].set_xlabel('Beta')
        axs[0].set_ylabel('Frequency')
        axs[0].legend()

        axs[1].hist(eta_samples, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axs[1].axvline(eta_mean, color='red', linestyle='-', label=f'Mean = {eta_mean:.2f}')
        axs[1].axvline(eta_ci[0], color='green', linestyle='--', label=f'2.5% = {eta_ci[0]:.2f}')
        axs[1].axvline(eta_ci[1], color='green', linestyle='--', label=f'97.5% = {eta_ci[1]:.2f}')
        axs[1].set_title('Bootstrap Distribution of Eta')
        axs[1].set_xlabel('Eta')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()

        fig.tight_layout()
        return fig

    def plot_mc_histogram(self):
        simulated_data = self.mc_result['simulated_data']
        percentiles = [
            self.mc_result['B10'],
            self.mc_result['B50'],
            np.percentile(simulated_data, 90),
            self.mc_result['B95'],
            np.percentile(simulated_data, 99)
        ]
        labels = ['B10', 'B50', 'B90', 'B95', 'B99']

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(simulated_data, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

        for i, p in enumerate(percentiles):
            ax.axvline(p, color='red', linestyle='--', alpha=0.7)
            y_text = ax.get_ylim()[1] * (0.85 - i * 0.03)
            x_text = p + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
            ax.text(x_text, y_text, f'{labels[i]}: {p:.1f}', rotation=90,
                    verticalalignment='center', color='red', fontsize=9)

        ax.set_title('Stochastic Simulated Lifetime Forecasting Distribution')
        ax.set_xlabel('Lifetime')
        ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.6)
        fig.tight_layout()
        return fig

    def plot_simulated_vs_actual(self):
        simulated_data = self.mc_result['simulated_data']
        actual_data = self.data
        beta_mean = self.bootstrap_result['beta_mean']
        eta_mean = self.bootstrap_result['eta_mean']
        loc = 0
        x_vals = np.linspace(0, max(max(simulated_data), max(actual_data)), 500)

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].hist(actual_data, bins=30, density=True, alpha=0.6, label='Actual', color='steelblue', edgecolor='black')
        axs[0].hist(simulated_data, bins=50, density=True, alpha=0.4, label='Simulated', color='orange', edgecolor='black')
        axs[0].plot(x_vals, weibull_min.pdf(x_vals, beta_mean, loc=loc, scale=eta_mean), 'r--', lw=2, label='Theoretical PDF')
        axs[0].set_title('Histogram: Simulated vs Actual')
        axs[0].set_xlabel('Lifetime')
        axs[0].set_ylabel('Density')
        axs[0].legend()
        axs[0].grid(True, linestyle='--', alpha=0.5)

        ecdf_actual = ECDF(actual_data)
        ecdf_sim = ECDF(simulated_data)
        ks_stat, p_value = ks_2samp(actual_data, simulated_data)
        x_common = np.linspace(min(actual_data.min(), simulated_data.min()),
                               max(actual_data.max(), simulated_data.max()), 1000)
        y_actual = ECDF(actual_data)(x_common)
        y_sim = ECDF(simulated_data)(x_common)
        diff = np.abs(y_actual - y_sim)
        max_diff_index = np.argmax(diff)
        ks_x = x_common[max_diff_index]
        ks_y1 = y_actual[max_diff_index]
        ks_y2 = y_sim[max_diff_index]

        axs[1].plot(ecdf_actual.x, ecdf_actual.y, label='Empirical CDF (Actual)', lw=2, color='blue')
        axs[1].plot(ecdf_sim.x, ecdf_sim.y, label='Empirical CDF (Simulated)', lw=2, color='orange', linestyle='--')
        axs[1].plot(x_vals, weibull_min.cdf(x_vals, beta_mean, loc=loc, scale=eta_mean), 'r-', label='Theoretical CDF')
        axs[1].vlines(ks_x, ks_y1, ks_y2, color='black', linestyle=':', linewidth=1.5, label='KS Distance')
        axs[1].text(ks_x + 10, (ks_y1 + ks_y2) / 2, f"D={ks_stat:.3f}", color='black', fontsize=9)
        axs[1].set_title('CDF Comparison (with KS Test)')
        axs[1].set_xlabel('Lifetime')
        axs[1].set_ylabel('Cumulative Probability')
        axs[1].legend()
        axs[1].grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()

        ks_result = {
            'ks_stat': ks_stat,
            'p_value': p_value,
            'conclusion': (
                "Simulated and actual data are likely from the same distribution (p > 0.05)."
                if p_value > 0.05
                else "Simulated and actual data are significantly different (p < 0.05)."
            )
        }
        return fig, ks_result


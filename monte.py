from weibull_test import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
import numpy as np
import pandas as pd

"""analyzer = LifetimeAnalyzer("Nanboom2.csv")
analyzer.fit_distributions()
analyzer.plot_histogram()
analyzer.plot_pdf_comparison()
analyzer.plot_survival_comparison()

mc = LifetimeAnalyzerMC(analyzer.data, analyzer.weibull_params['beta'], analyzer.weibull_params['eta'])
mc.print_bootstrap_weibull_params()
mc.print_monte_carlo_lifetime(n_simulations=10000)
mc.print_reliability_at_cycles(600)"""

# 1. 先用 LifetimeAnalyzer 進行分布擬合
analyzer = LifetimeAnalyzer("Nanboom2.csv")
analyzer.fit_distributions()
analyzer.plot_histogram()
analyzer.plot_pdf_comparison()
analyzer.plot_survival_comparison()
beta = analyzer.weibull_params['beta']
eta = analyzer.weibull_params['eta']

# 2. 建立 LifetimeAnalyzerMC 進行統計分析
mc = LifetimeAnalyzerMC(analyzer.data, beta, eta)
mc.print_bootstrap_weibull_params()
mc.print_monte_carlo_lifetime()

# 3. 建立 LifetimeAnalyzerPlot 進行視覺化
plotter = LifetimeAnalyzerPlot(analyzer.data, beta, eta)
plotter.plot_bootstrap_weibull_params(n_bootstrap=10000)
plotter.plot_monte_carlo_lifetime(n_simulations=10000)
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
import numpy as np
import pandas as pd

# 1. 先用 LifetimeAnalyzer 進行分布擬合
analyzer = LifetimeAnalyzer("Nanboom2.csv")
#analyzer = LifetimeAnalyzer("Emperor pan and tilt test to failure.csv")
analyzer.fit_distributions()
analyzer.plot_histogram()
analyzer.plot_pdf_comparison()
analyzer.plot_survival_comparison()
beta = analyzer.weibull_params['beta']
eta = analyzer.weibull_params['eta']

# 2. 建立 LifetimeAnalyzerMC 進行統計分析
mc = LifetimeAnalyzerMC(analyzer.data, beta, eta)
bootstrap_result = mc.print_bootstrap_weibull_params()
mc_result = mc.print_monte_carlo_lifetime()
mc.print_reliability_at_cycles(cycles=600)

# 3. 建立 LifetimeAnalyzerPlot 進行視覺化
plotter = LifetimeAnalyzerPlot(data=mc.data, bootstrap_result=bootstrap_result, mc_result=mc_result)
plotter.plot_bootstrap_histograms()
plotter.plot_mc_histogram()
plotter.plot_simulated_vs_actual()
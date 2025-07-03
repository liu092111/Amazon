from weibull_test import LifetimeAnalyzer
import numpy as np
import pandas as pd

df = pd.read_csv('Nanboom.csv', encoding='utf-8-sig')  # 解決 BOM 問題
data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values  # 只取第一欄並清洗空值與非數字
analyzer = LifetimeAnalyzer(data)

analyzer.fit_distributions()
analyzer.plot_pdf_comparison()
analyzer.calculate_mttf()

#beta_samples, eta_samples = analyzer.bootstrap_weibull_params()
analyzer.bootstrap_weibull_params()
analyzer.monte_carlo_simulation()

mc = analyzer.plot_monte_carlo_distribution()
analyzer.sensitivity_analysis()

analyzer.run_all()

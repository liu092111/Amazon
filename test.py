from statsmodels.stats.proportion import proportion_confint

failures = 1
n = 10

lower, upper = proportion_confint(count=failures, nobs=n, alpha=0.1, method='beta')
print(f"失效率估計：{failures/n:.2%}")
print(f"90% 信賴區間：{lower:.2%} ~ {upper:.2%}")

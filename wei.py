import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import seaborn as sns

def weibull_sample_size_vs_test_time(beta_list, R, CL, Td, r_list):
    fig, ax = plt.subplots(figsize=(10, 6))

    lnR = np.log(R)

    time_range = np.linspace(200, 1200, 100)

    for beta in beta_list:
        for r in r_list:
            chi_val = chi2.ppf(CL, 2 * (r + 1))
            eta = Td / (-lnR) ** (1 / beta)
            n_values = chi_val / ((time_range / eta) ** beta)
            label = f"β={beta}, r={r}"
            ax.plot(time_range, n_values, label=label)

    ax.set_title("Sample Size vs Test Time for different β and r")
    ax.set_xlabel("Test Time per Unit (cycles)")
    ax.set_ylabel("Required Sample Size")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig

# 測試參數
beta_list = [1.0, 1.5, 2.0]
r_list = [0, 1, 2]
R = 0.90
CL = 0.90
Td = 1000

fig = weibull_sample_size_vs_test_time(beta_list, R, CL, Td, r_list)
fig.show()
input("Press Enter to exit...")

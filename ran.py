import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 設定亂數種子（可重現結果）
np.random.seed(100)

# 真實參數設定
lambda_true = 1
sample_size = 1000

# 產生隨機樣本
samples = np.random.exponential(scale=1/lambda_true, size=sample_size)

# 繪製直方圖（取得 bin 中心與高度）
counts, bin_edges, _ = plt.hist(samples, bins=50, density=True, alpha=0.4, label='Sample Histogram')
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 自訂擬合函數（指數分布 PDF）
def exp_pdf(x, lambd):
    return lambd * np.exp(-lambd * x)

# 擬合 PDF 曲線
params, _ = curve_fit(exp_pdf, bin_centers, counts, p0=[1.0])
lambda_fit = params[0]

# 建立 x 軸與對應的 PDF 值
x_fit = np.linspace(0, np.max(samples), 500)
pdf_fit = exp_pdf(x_fit, lambda_fit)
pdf_true = exp_pdf(x_fit, lambda_true)

# 畫擬合與真實曲線
plt.plot(x_fit, pdf_true, 'r--', label=f'True PDF: $1/MTTF = {lambda_true}$')
plt.plot(x_fit, pdf_fit, 'g-', label=f'Fitted PDF: $1/MTTF = {lambda_fit:.2f}$')

# 標註長條圖點位置
plt.scatter(bin_centers, counts, s=15, color='blue', label='Histogram Points')

# 計算 MTTF（Mean Time To Failure = 1 / λ）
mttf = 1 / lambda_fit
true_mttf = 1 / lambda_true
mttf_error = abs(mttf - true_mttf) / true_mttf * 100

# 顯示方程式與 MTTF
plt.text(3.5, max(counts)*0.9, f'$f(x) = {lambda_fit:.2f}e^{{-{lambda_fit:.2f}x}}$\nMTTF = {mttf:.2f}\nError = {mttf_error:.2f}%', 
        fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

# 完善圖示
plt.title('Exponential Distribution Fit with Histogram Points')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
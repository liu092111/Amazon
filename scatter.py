import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# === 參數設定 ===
R = 8.314        # J/mol·K
Ea = 80000       # J/mol
A = 1e13         # Pre-exponential factor
n_points = 500   # 隨機點的數量，可自行調整
T_min = 273      # 最低溫
T_max = 373      # 最高溫
noise_ratio = 0.2  # 幾% 相對噪聲
power_factor = 0.7  # 採樣曲率控制 (0.5=平方採樣，1=線性採樣)

# === 非均勻溫度採樣 ===
T_nonuniform = np.linspace(T_min**power_factor, T_max**power_factor, n_points)**(1/power_factor)
T = np.sort(T_nonuniform)

# === 相對噪聲生成 ===
k_ideal = A * np.exp(-Ea/(R * T))
k = k_ideal * (1 + noise_ratio * np.random.randn(len(T)))

# === 對數回歸 ===
X = (1/T).reshape(-1,1)
y = np.log(np.abs(k))  # 避免 log 負數
reg = LinearRegression().fit(X, y)
y_pred_log = reg.predict(X)
k_pred = np.exp(y_pred_log)

# === Savitzky-Golay 濾波器平滑 ===
window_size = max(11, int(0.1 * len(T)))
window_size = window_size if window_size % 2 == 1 else window_size + 1
k_smooth = savgol_filter(k, window_length=window_size, polyorder=2)

# === 計算 R² ===
r2 = r2_score(k, k_smooth)

# === Plot 1：Ideal Curve Only ===
plt.figure(figsize=(8, 5))
T_plot = np.linspace(T_min, T_max, 500)
k_plot = A * np.exp(-Ea / (R * T_plot))
plt.plot(T_plot, k_plot, label='Ideal Arrhenius Curve', color='black')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Ideal Arrhenius Curve')
plt.grid(True)
plt.legend()
plt.show()

# === Plot 2：Noisy Data Only ===
plt.figure(figsize=(8, 5))
plt.scatter(T, k, s=10, alpha=0.5, label='Noisy Data', color='blue')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Noisy Sampled Arrhenius Data')
plt.grid(True)
plt.legend()
plt.show()

# === Plot 3：Noisy + Ideal + Smoothed ===
plt.figure(figsize=(10, 6))
plt.plot(T_plot, k_plot, label='Ideal Arrhenius Curve', color='black')
plt.scatter(T, k, s=10, alpha=0.5, label='Noisy Data', color='blue')
plt.plot(T, k_smooth, color='red', linewidth=2, label='Smoothed Prediction (Savitzky-Golay)')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Arrhenius Curve with Noisy Data and Smoothed Prediction')
plt.legend()
plt.grid(True)
plt.show()

# === Plot 4：Measured vs Smoothed + R² ===
plt.figure(figsize=(8, 6))
plt.scatter(k, k_smooth, color='green', s=10, label='Measured vs Smoothed')
plt.plot([min(k), max(k)], [min(k), max(k)], color='red', linestyle='--', label='Ideal')
plt.xlabel('Measured k')
plt.ylabel('Smoothed Predicted k')
plt.title(f'Measured vs Smoothed Prediction (R² = {r2:.4f})')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# === 參數設定 ===
Ts = 373.0       # 加速溫度（K）
Tn = 293.0       # 正常工作溫度（K）
M = 5.0          # Coffin-Manson 指數
n_points = 900   # 離散點數
T_min = 273
T_max = 373
noise_ratio = 0.25  # 相對誤差

equation = r"$AF = \left(\frac{T_s}{T_n}\right)^M$"

# === 1. Ideal Coffin-Manson Curve ===
T_curve = np.linspace(T_min, T_max, 500)
AF_curve = (T_curve / Tn)**M

plt.figure()
plt.plot(T_curve, AF_curve, label=f'Ideal curve {equation}', color='black')
plt.xlabel('Temperature (K)')
plt.ylabel('Acceleration Factor (AF)')
plt.title('Ideal Coffin-Manson Curve')
plt.legend()
plt.grid(True)
plt.show()

# === 2. 產生樣本點並加入噪聲 ===
T = np.arange(T_min, T_max, 1)
AF_true = (T / Tn)**M
AF_noisy = AF_true * (1 + noise_ratio * np.random.randn(len(T)))

plt.figure()
plt.scatter(T, AF_noisy, color='blue', s=10, label='Samples')
plt.plot(T_curve, AF_curve, color='gray', label=f'Ideal curve {equation}')
plt.xlabel('Temperature (K)')
plt.ylabel('Acceleration Factor (AF)')
plt.title('Noisy Coffin-Manson Data')
plt.legend()
plt.grid(True)
plt.show()

# === 3. 平滑預測（Savitzky-Golay） ===
sorted_idx = np.argsort(T)
T_sorted = T[sorted_idx]
AF_sorted = AF_noisy[sorted_idx]

AF_smooth = savgol_filter(AF_sorted, window_length=21, polyorder=1)

plt.figure()
plt.scatter(T_sorted, AF_sorted, color='gray', s=10, labeㄣl='Samples')
plt.plot(T_sorted, AF_smooth, color='red', linewidth=2, label='Predicted Line')
plt.plot(T_curve, AF_curve, label=f'Ideal curve {equation}')
plt.xlabel('Temperature (K)')
plt.ylabel('Acceleration Factor (AF)')
plt.title('Coffin Manson Equation')
plt.legend()
plt.grid(True)
plt.show()

# === 4. Measured vs Prediction plot + R² ===
r2 = r2_score(AF_sorted, AF_smooth)

plt.figure()
plt.scatter(AF_sorted, AF_smooth, color='green', s=10, label=f'Measured vs Estimated')
plt.plot([min(AF_sorted), max(AF_sorted)], [min(AF_sorted), max(AF_sorted)],
        color='red', linestyle='--', label='Ideal correlation')
plt.xlabel('Measured AF')
plt.ylabel('Smoothed Predicted AF')
plt.title(f'Measured vs Prediction (R² = {r2:.4f})')
plt.legend()
plt.grid(True)
plt.show()

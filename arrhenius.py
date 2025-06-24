import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# === 參數設定 ===
R = 8.314        # J/mol·K
Ea_true = 80000  # 真實 Ea，用來產生 k
A = 1e13         # Pre-exponential factor
n_points = 900   # 隨機點的數量
T_min = 293      # 最低溫
T_max = 373      # 最高溫
equation = r"$E_a = -RT \ln\left(\frac{k}{A}\right)$"

# === 1. 產生理想曲線（這裡為固定 Ea 對應的 k）===
T_curve = np.linspace(T_min, T_max, 500)
k_curve = A * np.exp(-Ea_true / (R * T_curve))

plt.figure()
plt.plot(T_curve, k_curve, label=f'Ideal k from Ea = {Ea_true/1000:.0f} kJ/mol')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Ideal Arrhenius Curve')
plt.legend()
plt.grid(True)
plt.show()

# === 2. 產生離散點，然後從 k 反推 Ea ===
T = np.arange(T_min, T_max, 1)
k = A * np.exp(-Ea_true / (R * T)) + 5 * np.random.randn(len(T))  # 含 noise

# 避免 log 負數與除零
k = np.clip(k, 1e-10, None)

Ea_estimated = -R * T * np.log(k / A)

plt.figure()
plt.scatter(T, Ea_estimated, color='blue', s=10, label='Estimated Ea')
plt.hlines(Ea_true, T_min, T_max, color='gray', linestyle='--', label='True Ea')
plt.xlabel('Temperature (K)')
plt.ylabel('Estimated Ea (J/mol)')
plt.title('Estimated Activation Energy from k')
plt.legend()
plt.grid(True)
plt.show()

# === 3. 平滑處理 ===
sorted_idx = np.argsort(T)
T_sorted = T[sorted_idx]
Ea_sorted = Ea_estimated[sorted_idx]

Ea_smooth = savgol_filter(Ea_sorted, window_length=31, polyorder=2)

plt.figure()
plt.scatter(T_sorted, Ea_sorted, color='gray', s=10, label='Estimated Ea')
plt.plot(T_sorted, Ea_smooth, color='red', linewidth=2, label='Smoothed Ea')
plt.hlines(Ea_true, T_min, T_max, color='black', linestyle='--', label='True Ea')
plt.xlabel('Temperature (K)')
plt.ylabel('Activation Energy (J/mol)')
plt.title('Smoothed Activation Energy (Savitzky-Golay)')
plt.legend()
plt.grid(True)
plt.show()

# === 4. Measured vs Smoothed Ea ===
r2 = r2_score(Ea_sorted, Ea_smooth)

plt.figure()
plt.scatter(Ea_sorted, Ea_smooth, color='green', s=10, label='Measured vs Smoothed')
plt.plot([min(Ea_sorted), max(Ea_sorted)], [min(Ea_sorted), max(Ea_sorted)],
         color='red', linestyle='--', label='Ideal')
plt.xlabel('Measured Ea (J/mol)')
plt.ylabel('Smoothed Ea (J/mol)')
plt.title(f'Measured vs Smoothed Ea (R² = {r2:.4f})')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.grid(True)
plt.show()

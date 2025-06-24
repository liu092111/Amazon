import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# === 參數設定 ===
R = 8.314        # J/mol·K
Ea = 80000       # J/mol
A = 1e13         # Pre-exponential factor
n_points = 500   # 隨機點的數量
T_min = 293      # 最低溫
T_max = 373      # 最高溫
equation = f"$k = A e^{{-Ea/RT}}$"

# === 1. 畫理想曲線 ===
T_curve = np.linspace(T_min, T_max, 500)
k_curve = A * np.exp(-Ea / (R * T_curve))

plt.figure()
plt.plot(T_curve, k_curve, label=f'Ideal: Ea={Ea} J/mol, \nA={A:.1e}')
plt.xlabel('T (K)')
plt.ylabel('k')
plt.title('Ideal Arrhenius Curve')
plt.legend()
plt.grid(True)
plt.show()

# === 2. 產生離散點，加入隨機干擾 ===
T_nonuniform = np.linspace(T_min**0.5, T_max**0.5, n_points)**2  # 平方採樣偏重高溫區
T = np.sort(T_nonuniform)  # 確保溫度有序
k = A * np.exp(-Ea/(R * T)) + 5 * np.random.randn(len(T))

plt.figure()
plt.scatter(T, k, color='blue', s=10, label='Noisy samples')
plt.plot(T_curve, k_curve, color='gray', label='Ideal curve')
plt.xlabel('T (K)')
plt.ylabel('k')
plt.title('Arrhenius Curve with Noise')
plt.legend()
plt.grid(True)
plt.show()

# === 3. 使用 Savitzky-Golay filter 平滑預測 ===
# 對資料按 T 排序
sorted_idx = np.argsort(T)
T_sorted = T[sorted_idx]
k_sorted = k[sorted_idx]

# 使用 Savitzky-Golay 濾波器進行平滑
window_size = int(0.1 * len(T))  # 動態窗口大小
window_size = window_size if window_size % 2 == 1 else window_size + 1  # 確保奇數
k_smooth = savgol_filter(k_sorted, window_length=window_size, polyorder=2)


plt.figure()
plt.scatter(T_sorted, k_sorted, color='gray', s=10, label='Noisy samples')
plt.plot(T_sorted, k_smooth, color='red', linewidth=2, label='Smoothed prediction')
plt.plot(T_curve, k_curve, label=f'Ideal: Ea={Ea} J/mol, A={A:.1e}')
plt.xlabel('T (K)')
plt.ylabel('k')
plt.title('Smoothed Prediction (Savitzky-Golay Filter)')
plt.legend()
plt.grid(True)
plt.show()

# === 4. Measured vs Smoothed plot + R² ===
r2 = r2_score(k_sorted, k_smooth)

plt.figure()
plt.scatter(k_sorted, k_smooth, color='green', s=10, label='Measured vs Smoothed')
plt.plot([min(k_sorted), max(k_sorted)], [min(k_sorted), max(k_sorted)],
        color='red', linestyle='--', label='Ideal')
plt.xlabel('Measured k')
plt.ylabel('Smoothed Predicted k')
plt.title(f'Measured vs Smoothed Prediction (R² = {r2:.4f})')
plt.legend()
plt.grid(True)
plt.show()
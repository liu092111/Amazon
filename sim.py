import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.signal import savgol_filter

# === 參數設定 ===
R = 8.314        # J/mol·K
Ea = 80000       # J/mol
A = 1e13         # Pre-exponential factor
n_points = 100   # 隨機點的數量
T_min = 293      # 最低溫
T_max = 373      # 最高溫
noise_ratio = 0.15  # 相對誤差
equation = f"$k = A e^{{-Ea/RT}}$"

# === 1. 畫理想曲線 ===
T_curve = np.linspace(T_min, T_max, 500)
k_curve = A * np.exp(-Ea / (R * T_curve))

plt.figure()
plt.plot(T_curve, k_curve, label=f'Ideal curve {equation}')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Ideal Arrhenius Curve')
plt.legend()
plt.grid(True)
plt.show()

# === 2. 產生離散點，加入隨機干擾 ===
T = np.arange(T_min, T_max, 1)
#k = A * np.exp(-Ea/(R * T)) + (1 + noise_ratio * np.random.randn(len(T)))

k = A * np.exp(-Ea/(R * T)) + 5 * np.random.randn(len(T))
print("len T: ", len(T))

plt.figure()
plt.scatter(T, k, color='blue', s=10, label='Samples')
plt.plot(T_curve, k_curve, color='gray', label=f'Ideal curve {equation}')
plt.xlabel('Temperature (K)')
plt.ylabel('k')
plt.title('Experiment (Simulation) Arrhenius Curve')
plt.legend()
plt.grid(True)
plt.show()

# === 3. 使用 Savitzky-Golay filter 平滑預測 ===
# 對資料按 T 排序
sorted_idx = np.argsort(T)
T_sorted = T[sorted_idx]
k_sorted = k[sorted_idx]

# 使用 Savitzky-Golay 濾波器進行平滑
k_smooth = savgol_filter(k_sorted, window_length=21, polyorder=1)

plt.figure()
plt.scatter(T_sorted, k_sorted, color='gray', s=10, label='Samples')
plt.plot(T_sorted, k_smooth, color='red', linewidth=2, label='Predicted Line')
plt.plot(T_curve, k_curve, label=f'Ideal curve {equation}')
plt.xlabel('T (K)')
plt.ylabel('k')
plt.title('Arrhenius Equation')
plt.legend()
plt.grid(True)
plt.show()

# === 4. Measured vs Smoothed plot + R² ===
r2 = r2_score(k_sorted, k_smooth)

plt.figure()
plt.scatter(k_sorted, k_smooth, color='green', s=10, label=f'Measured vs Estimated, R² = {r2:.4f})')
plt.plot([min(k_sorted), max(k_sorted)], [min(k_sorted), max(k_sorted)],
        color='red', linestyle='--', label='Correlation')
plt.xlabel('Measured k')
plt.ylabel('Estimated k')
plt.title(f'Measured vs Prediction (R² = {r2:.4f})')
plt.xlim(left=0)   # 強制 x 軸從 0 開始
plt.ylim(bottom=0) # 強制 y 軸從 0 開始
plt.legend()
plt.grid(True)
plt.show()
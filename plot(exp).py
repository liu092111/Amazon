import numpy as np
import matplotlib.pyplot as plt

# 定義 x 範圍
x = np.linspace(0, 5, 500)

# 定義所有需要比較的函數組（每組兩條線）
functions = [
    (np.exp(-x), np.exp(-2*x), r'$e^{-x}$', r'$e^{-2x}$'),                  # λ 變化
    (np.exp(-x), 2*np.exp(-x), r'$e^{-x}$', r'$2e^{-x}$'),                  # 前方係數變化（放大）
    (np.exp(-x), 2*np.exp(-2*x), r'$e^{-x}$', r'$2e^{-2*x}$'),              # 前方係數變化（縮小）
    (np.exp(-x), np.exp(-2*x), r'$e^{-x}$', r'$e^{-x/2}$'),                 # 衰減變慢
    (np.exp(-x), np.exp((-x))+0.5, r'$e^{-x}$', r'$e^{-x}+0.5$')         # 與常見非指數分布比較
]

# 畫出每一組比較圖
for i, (y1, y2, label1, label2) in enumerate(functions):
    plt.figure(figsize=(6, 4))
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2, linestyle='--')
    plt.title(f' Comparison{i+1}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
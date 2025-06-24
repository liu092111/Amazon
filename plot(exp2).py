import numpy as np
import matplotlib.pyplot as plt

# 自訂幾組參數 (A, lambda, B)
params = [
    (1, 1),
    (2, 1),
    (4, 1)
]

x = np.linspace(0, 10, 500)
plt.figure(figsize=(6, 4))

for A, lam in params:
    y = A * np.exp(-lam * x)
    equation = f"${A}e^{{-{lam}x}}$"
    plt.plot(x, y, label=equation)

plt.title('Comparison 4')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
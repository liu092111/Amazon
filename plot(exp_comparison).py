import numpy as np
import matplotlib.pyplot as plt

# 自訂幾組參數 (A, lambda, B)
params = [
    (1, 0.5, 0),
    (1.5, 1, 0.2),
    (0.8, 1.5, 0.1),
    (1.2, 2, 0.3),
]

x = np.linspace(0, 10, 500)
plt.figure(figsize=(8, 6))

for A, lam, B in params:
    y = A * np.exp(-lam * x) + B
    equation = f"${A}e^{{-{lam}x}} + {B}$"
    plt.plot(x, y, label=equation)

plt.title('exponential')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
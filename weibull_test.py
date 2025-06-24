import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

def calculate():
    try:
        beta = float(entry_beta.get())
        R = float(entry_reliability.get())
        CL = float(entry_confidence.get())
        Td = float(entry_demotime.get())
        r = int(entry_r.get())

        chi_val = chi2.ppf(CL, 2 * (r + 1))
        lnR = np.log(R)
        eta = Td / (-lnR) ** (1 / beta)

        mode = mode_var.get()

        if mode == "Sample Size":
            t = float(entry_testtime.get())
            n = chi_val / ((t / eta) ** beta)
            result_var.set(f"✅ Required Sample Size: {np.ceil(n):.0f} units")
            plot_sample_vs_time(beta, eta, chi_val)
        else:
            n = int(entry_samplesize.get())
            t = eta * (chi_val / n) ** (1 / beta)
            result_var.set(f"✅ Required Test Time per Unit: {t:.2f} cycles")

    except Exception as e:
        messagebox.showerror("Input Error", str(e))

def plot_sample_vs_time(beta, eta, chi_val):
    times = np.linspace(200, 1200, 50)
    samples = chi_val / ((times / eta) ** beta)
    plt.figure()
    plt.plot(times, samples, "o-")
    plt.title("Sample Size vs Test Time")
    plt.xlabel("Test Time per Unit (cycles)")
    plt.ylabel("Required Sample Size")
    plt.grid(True)
    plt.show()

# ===== GUI START =====
root = tk.Tk()
root.title("Weibull Test Planner")

# Mode
mode_var = tk.StringVar(value="Sample Size")
mode_menu = ttk.Combobox(root, textvariable=mode_var, values=["Sample Size", "Test Time"])
mode_menu.grid(row=0, column=1)
ttk.Label(root, text="Mode:").grid(row=0, column=0)

# Inputs
labels = ["Shape Parameter (β):", "Target Reliability (0.01–1.0):", "Confidence Level (0.01–1.0):",
          "Demonstration Time (cycles):", "Test Time per Unit (cycles):", "Sample Size:", "Max Allowed Failures (r):"]
entries = []

for i, text in enumerate(labels):
    ttk.Label(root, text=text).grid(row=i+1, column=0)
    entry = ttk.Entry(root)
    entry.grid(row=i+1, column=1)
    entries.append(entry)

entry_beta, entry_reliability, entry_confidence, entry_demotime, entry_testtime, entry_samplesize, entry_r = entries

# Default Values
entry_beta.insert(0, "1.5")
entry_reliability.insert(0, "0.90")
entry_confidence.insert(0, "0.90")
entry_demotime.insert(0, "1000")
entry_testtime.insert(0, "548")
entry_samplesize.insert(0, "100")
entry_r.insert(0, "0")

# Calculate Button
ttk.Button(root, text="Calculate", command=calculate).grid(row=8, column=0, columnspan=2, pady=10)

# Result
result_var = tk.StringVar()
ttk.Label(root, textvariable=result_var, foreground="green").grid(row=9, column=0, columnspan=2)

root.mainloop()

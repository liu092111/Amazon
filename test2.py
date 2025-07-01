import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import weibull_min, lognorm, expon

fitted_eta = None
fitted_beta = None
selected_model = 'All'
fitted_models = {}
loaded_data = None
loaded_file_path = None

# 計算函式
def compute_eta(Td, R, beta):
    return Td / (-np.log(R)) ** (1 / beta)

def compute_sample_size(model, R, CL, Td, t, r, eta=None, beta=None, sigma=None):
    chi_val = chi2.ppf(CL, 2 * (r + 1))
    if model == 'Weibull':
        if eta is None or beta is None:
            eta = compute_eta(Td, R, beta)
        n = chi_val / ((t / eta) ** beta)
    elif model == 'Lognormal' and sigma is not None:
        z = norm.ppf(R)
        mu = np.log(Td) - sigma * z
        n = chi_val / ((np.log(t) - mu) ** 2 / sigma ** 2)
    elif model == 'Exponential':
        lambda_ = -np.log(R) / Td
        n = chi_val / ((lambda_ * t))
    else:
        return None, None, None
    return np.ceil(n), eta, chi_val

def compute_test_time(model, R, CL, Td, n, r, eta=None, beta=None, sigma=None):
    chi_val = chi2.ppf(CL, 2 * (r + 1))
    if model == 'Weibull':
        if eta is None or beta is None:
            eta = compute_eta(Td, R, beta)
        t = eta * (chi_val / n) ** (1 / beta)
    elif model == 'Lognormal' and sigma is not None:
        z = norm.ppf(R)
        mu = np.log(Td) - sigma * z
        t = np.exp(mu + sigma * np.sqrt(chi_val / n))
    elif model == 'Exponential':
        lambda_ = -np.log(R) / Td
        t = chi_val / (lambda_ * n)
    else:
        return None, None, None
    return t, eta, chi_val

def fit_weibull_beta(data):
    def neg_log_likelihood(params):
        beta, eta = params
        if beta <= 0 or eta <= 0:
            return np.inf
        return -np.sum(np.log((beta / eta) * (data / eta) ** (beta - 1) * np.exp(-(data / eta) ** beta)))

    initial_guess = [1.5, np.mean(data)]
    result = minimize(neg_log_likelihood, initial_guess, method='L-BFGS-B')
    return result.x if result.success else (None, None)

def compare_models(data):
    results = {}
    try:
        wb_params = weibull_min.fit(data, floc=0)
        ln_params = lognorm.fit(data, floc=0)
        ex_params = expon.fit(data, floc=0)

        wb_ll = np.sum(weibull_min.logpdf(data, *wb_params))
        ln_ll = np.sum(lognorm.logpdf(data, *ln_params))
        ex_ll = np.sum(expon.logpdf(data, *ex_params))

        results['Weibull'] = (wb_params, -2 * wb_ll)
        results['Lognormal'] = (ln_params, -2 * ln_ll)
        results['Exponential'] = (ex_params, -2 * ex_ll)
    except Exception as e:
        messagebox.showerror("Model Fit Error", str(e))
    return results

def launch_gui():
    global fitted_eta, fitted_beta, selected_model, fitted_models, loaded_data, loaded_file_path
    root = tk.Tk()
    root.title("Advanced Weibull Reliability Test Planner")

    inputs = {}
    fields = [
        ("Shape Parameter (β)", "1.5"),
        ("Target Reliability (R)", "0.9"),
        ("Confidence Level (CL)", "0.9"),
        ("Demonstration Time (Td, cycles)", "1000"),
        ("Test Time per Unit (cycles)", "548"),
        ("Sample Size (n)", "100"),
        ("Allowed Failures (r)", "0")
    ]
    for i, (label, default) in enumerate(fields):
        ttk.Label(root, text=label).grid(row=i, column=0)
        entry = ttk.Entry(root)
        entry.insert(0, default)
        entry.grid(row=i, column=1)
        inputs[label] = entry

    mode_var = tk.StringVar(value="Sample Size")
    ttk.Label(root, text="Mode:").grid(row=0, column=2)
    mode_menu = ttk.Combobox(root, textvariable=mode_var, values=["Sample Size", "Test Time"])
    mode_menu.grid(row=0, column=3)

    result_var = tk.StringVar()
    ttk.Label(root, textvariable=result_var, foreground="green").grid(row=8, column=0, columnspan=4)

    eta_display_var = tk.StringVar()
    ttk.Label(root, textvariable=eta_display_var, foreground="blue").grid(row=9, column=0, columnspan=4)

    file_display_var = tk.StringVar()
    ttk.Label(root, textvariable=file_display_var, foreground="gray").grid(row=11, column=0, columnspan=4)

    use_eta_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(root, text="使用擬合 η 取代推導值", variable=use_eta_var).grid(row=10, column=0, columnspan=2)

    model_var = tk.StringVar(value="All")
    ttk.Label(root, text="選擇模型:").grid(row=10, column=2)
    model_menu = ttk.Combobox(root, textvariable=model_var, values=["Weibull", "Lognormal", "Exponential", "All"])
    model_menu.grid(row=10, column=3)

    def calculate():
        try:
            model = model_var.get()
            beta = float(inputs["Shape Parameter (β)"].get())
            R = float(inputs["Target Reliability (R)"].get())
            CL = float(inputs["Confidence Level (CL)"].get())
            Td = float(inputs["Demonstration Time (Td, cycles)"].get())
            r = int(inputs["Allowed Failures (r)"].get())
            mode = mode_var.get()
            eta = fitted_eta if use_eta_var.get() else compute_eta(Td, R, beta)
            sigma = fitted_models.get('Lognormal', ((None,),))[0][0] if 'Lognormal' in fitted_models else 0.4
            eta_display_var.set(f"η (scale) = {eta:.2f}")

            if model not in ['Weibull', 'Lognormal', 'Exponential']:
                messagebox.showinfo("Info", "請選擇單一模型來計算樣本數或測試時間。")
                return

            if mode == "Sample Size":
                t = float(inputs["Test Time per Unit (cycles)"].get())
                n, eta, chi_val = compute_sample_size(model, R, CL, Td, t, r, eta, beta, sigma)
                if n is not None:
                    result_var.set(f"✅ Required Sample Size: {n:.0f} units\n(eta={eta:.2f}, chi²={chi_val:.2f})")
                else:
                    result_var.set("❌ 此模型需要更多參數（如 sigma）")
            else:
                n = float(inputs["Sample Size (n)"].get())
                t, eta, chi_val = compute_test_time(model, R, CL, Td, n, r, eta, beta, sigma)
                if t is not None:
                    result_var.set(f"✅ Required Test Time per Unit: {t:.2f} cycles\n(eta={eta:.2f}, chi²={chi_val:.2f})")
                    plot_sample_vs_time(beta, R, CL, Td, r, eta)
                else:
                    result_var.set("❌ 此模型需要更多參數（如 sigma）")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_sample_vs_time(beta, R, CL, Td, r, eta=None):
        times = np.linspace(200, 1200, 100)
        chi_val = chi2.ppf(CL, 2 * (r + 1))
        if eta is None:
            eta = compute_eta(Td, R, beta)
        samples = chi_val / ((times / eta) ** beta)

        plt.figure(figsize=(8, 5))
        plt.plot(times, samples, label=f"β={beta}, r={r}")
        plt.xlabel("Test Time per Unit (cycles)")
        plt.ylabel("Required Sample Size")
        plt.title("Sample Size vs Test Time")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_survival_comparison(data):
        model_choice = model_var.get()
        x = np.linspace(min(data), max(data), 200)
        plt.figure(figsize=(10, 6))
        for model, (params, _) in fitted_models.items():
            if model_choice != "All" and model_choice != model:
                continue
            if model == 'Weibull':
                y = weibull_min.sf(x, *params)
            elif model == 'Lognormal':
                y = lognorm.sf(x, *params)
            elif model == 'Exponential':
                y = expon.sf(x, *params)
            plt.plot(x, y, label=model)
        plt.title("Survival Function")
        plt.xlabel("Time")
        plt.ylabel("Survival Probability")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_cost_tradeoff():
        try:
            cost_per_sample = float(simpledialog.askstring("輸入樣本成本", "每件樣本的成本是多少？", initialvalue="100"))
            cost_per_cycle = float(simpledialog.askstring("輸入單位週期成本", "每個週期的成本是多少？", initialvalue="0.5"))
            CL = float(inputs["Confidence Level (CL)"].get())
            R = float(inputs["Target Reliability (R)"].get())
            Td = float(inputs["Demonstration Time (Td, cycles)"].get())
            r = int(inputs["Allowed Failures (r)"].get())
            beta = float(inputs["Shape Parameter (β)"].get())
            eta = fitted_eta if use_eta_var.get() else compute_eta(Td, R, beta)
            time_range = np.linspace(200, 1200, 50)
            chi_val = chi2.ppf(CL, 2 * (r + 1))
            n_samples = chi_val / ((time_range / eta) ** beta)
            total_cost = n_samples * cost_per_sample + n_samples * time_range * cost_per_cycle

            plt.figure(figsize=(8, 5))
            plt.plot(time_range, total_cost, label='Total Cost')
            plt.xlabel('Test Time per Unit (cycles)')
            plt.ylabel('Total Cost')
            plt.title('Cost Optimization Curve')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Cost Plot Error", str(e))

    def load_csv():
        global fitted_eta, fitted_beta, selected_model, fitted_models, loaded_data, loaded_file_path
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            data = df.iloc[:, 0].dropna().values
            loaded_data = data
            loaded_file_path = file_path
            file_display_var.set(f"讀取檔案：{file_path}")
            beta, eta = fit_weibull_beta(data)
            if beta is not None:
                inputs["Shape Parameter (β)"].delete(0, tk.END)
                inputs["Shape Parameter (β)"].insert(0, f"{beta:.4f}")
                fitted_beta, fitted_eta = beta, eta
                eta_display_var.set(f"η (scale) = {eta:.2f}")
            fitted_models = compare_models(data)
            summary = "\n".join([f"{k}: AIC={v[1]:.2f}" for k, v in fitted_models.items()])
            best_model = min(fitted_models.items(), key=lambda x: x[1][1])[0]
            model_var.set("All")
            model_menu["values"] = list(fitted_models.keys()) + ["All"]
            messagebox.showinfo("AI Decision", f"最佳模型為：{best_model}\n\n{summary}")
            plot_survival_comparison(data)
            plt.figure(figsize=(8, 5))
            plt.hist(data, bins=20, alpha=0.7, label="Raw Data")
            plt.title("Histogram of Input CSV Data")
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))

    ttk.Button(root, text="Calculate", command=calculate).grid(row=12, column=0, pady=10)
    ttk.Button(root, text="Load CSV & Fit β", command=load_csv).grid(row=12, column=1, pady=10)
    ttk.Button(root, text="Plot Cost Curve", command=plot_cost_tradeoff).grid(row=12, column=2, pady=10)

    root.mainloop()

if __name__ == '__main__':
    launch_gui()
import tkinter as tk
import sys
from tkinter import filedialog, messagebox
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
import os

class LifetimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lifetime Analysis GUI")
        self.file_path = ""

        # ===== File Selection Section =====
        tk.Label(root, text="Input CSV File:").grid(row=0, column=0, sticky="e")
        self.file_entry = tk.Entry(root, width=50)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)

        self.load_button = tk.Button(root, text="Load", command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # ===== Expected Lifecycle Section =====
        tk.Label(root, text="Expected Lifecycle (cycles):").grid(row=1, column=0, sticky="e")
        self.lifecycle_entry = tk.Entry(root, width=20)
        self.lifecycle_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # ===== Execute Button =====
        self.run_button = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=2, column=1, pady=15)

        # ===== Console Output Section =====
        self.output_text = tk.Text(root, height=20, width=90, state='disabled', bg='black', fg='white')
        self.output_text.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        sys.stdout = TextRedirector(self.output_text)
        sys.stderr = TextRedirector(self.output_text)


    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def run_analysis(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return

        try:
            expected_cycles = int(self.lifecycle_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number for expected lifecycle.")
            return

        try:
            # Step 1: Load and fit distribution
            analyzer = LifetimeAnalyzer(self.file_path)
            analyzer.fit_distributions()
            analyzer.plot_histogram()
            analyzer.plot_pdf_comparison()
            analyzer.plot_survival_comparison()

            beta = analyzer.weibull_params['beta']
            eta = analyzer.weibull_params['eta']

            # Step 2: Monte Carlo Analysis
            mc = LifetimeAnalyzerMC(analyzer.data, beta, eta)
            bootstrap_result = mc.print_bootstrap_weibull_params()
            mc_result = mc.print_monte_carlo_lifetime()
            mc.print_reliability_at_cycles(cycles=expected_cycles)

            # Step 3: Plot Results
            plotter = LifetimeAnalyzerPlot(data=mc.data, bootstrap_result=bootstrap_result, mc_result=mc_result)
            plotter.plot_bootstrap_histograms()
            plotter.plot_mc_histogram()
            plotter.plot_simulated_vs_actual()

            messagebox.showinfo("Success", "Analysis completed successfully!")

        except Exception as e:
            messagebox.showerror("Error during analysis", str(e))

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')

    def flush(self):
        pass  # Required for file-like compatibility


if __name__ == "__main__":
    root = tk.Tk()
    app = LifetimeGUI(root)
    root.mainloop()

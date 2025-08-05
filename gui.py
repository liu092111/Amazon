import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import os
import matplotlib.pyplot as plt
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
from report_builder import ReportBuilder

class LifetimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lifetime Analysis GUI")

        self.file_path = ""
        self.figures = []
        self.output_buffer = []

        # Create text redirector
        sys.stdout = TextRedirector(self.output_buffer)
        sys.stderr = TextRedirector(self.output_buffer)

        # File selection
        tk.Label(root, text="Input CSV File:").grid(row=0, column=0, sticky="e")
        self.file_entry = tk.Entry(root, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)
        self.load_button = tk.Button(root, text="Upload", command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # Lifecycle input
        tk.Label(root, text="Expected Lifecycle (cycles):").grid(row=1, column=0, sticky="e")
        self.lifecycle_entry = tk.Entry(root, width=10)
        self.lifecycle_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # Run button
        self.run_button = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=2, column=1, pady=15)
        self.root.bind('<Return>', lambda event: self.run_analysis()) #按enter執行分析

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def save_figure_to_memory(self, fig):
        """Save matplotlib figure to BytesIO"""
        from io import BytesIO
        buf = BytesIO()
        try:
            fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
            buf.seek(0)
            return buf
        except Exception as e:
            print(f"Error saving figure to memory: {e}")
            return None
        finally:
            plt.close(fig)  # Close figure to free memory

    def run_analysis(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return

        try:
            expected_cycles = int(self.lifecycle_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid expected lifecycle number.")
            return

        try:
            # Clear previous results
            self.output_buffer.clear()
            self.figures.clear()
            
            # Main analysis
            analyzer = LifetimeAnalyzer(self.file_path)
            analyzer.fit_distributions()
            beta = analyzer.weibull_params['beta']
            eta = analyzer.weibull_params['eta']

            # Monte Carlo analysis
            mc = LifetimeAnalyzerMC(analyzer.data, beta, eta)
            bootstrap_result = mc.print_bootstrap_weibull_params()
            mc_result = mc.print_monte_carlo_lifetime()
            mc.print_reliability_at_cycles(expected_cycles)

            # Plotting
            plotter = LifetimeAnalyzerPlot(data=mc.data, bootstrap_result=bootstrap_result, mc_result=mc_result)

            # Generate all figures
            figures_data = []
            
            fig1 = analyzer.plot_histogram()
            buf1 = self.save_figure_to_memory(fig1)
            if buf1: figures_data.append(buf1)
            
            fig2 = analyzer.plot_pdf_comparison()
            buf2 = self.save_figure_to_memory(fig2)
            if buf2: figures_data.append(buf2)
            
            fig3 = analyzer.plot_survival_comparison()
            buf3 = self.save_figure_to_memory(fig3)
            if buf3: figures_data.append(buf3)
            
            fig4 = plotter.plot_bootstrap_histograms()
            buf4 = self.save_figure_to_memory(fig4)
            if buf4: figures_data.append(buf4)
            
            fig5 = plotter.plot_mc_histogram()
            buf5 = self.save_figure_to_memory(fig5)
            if buf5: figures_data.append(buf5)
            
            fig6, ks_result = plotter.plot_simulated_vs_actual()
            buf6 = self.save_figure_to_memory(fig6)
            if buf6: figures_data.append(buf6)

            self.figures = figures_data

            # Add KS test results to output buffer
            self.output_buffer.append("[BOLD14] === [4] Kolmogorov-Smirnov Test ===")
            self.output_buffer.append(f"KS Statistic (D): {ks_result['ks_stat']:.4f}")
            self.output_buffer.append(f"P-value: {ks_result['p_value']:.4f}")
            self.output_buffer.append(f"Conclusion: {ks_result['conclusion']}")

            # Generate PDF report
            report = ReportBuilder(self.output_buffer, self.figures)
            report.build("Analysis_Report.pdf")

            messagebox.showinfo("Success", "Analysis completed! PDF report generated.")

        except Exception as e:
            messagebox.showerror("Error during analysis", f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

class TextRedirector:
    def __init__(self, buffer_holder=None):
        self.buffer_holder = buffer_holder

    def write(self, message):
        if self.buffer_holder is not None and message.strip():
            self.buffer_holder.append(message.strip())

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = LifetimeGUI(root)
    root.mainloop()
import tkinter as tk
import sys
import re
from tkinter import filedialog, messagebox
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
from fpdf import FPDF
import os

class LifetimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lifetime Analysis GUI")
        self.file_path = ""

        # ===== File Selection Section =====
        tk.Label(root, text="Input CSV File:").grid(row=0, column=0, sticky="e")
        self.file_entry = tk.Entry(root, width=40)
        self.file_entry.grid(row=0, column=1, padx=5, pady=5)

        self.load_button = tk.Button(root, text="Upload", command=self.load_file)
        self.load_button.grid(row=0, column=2, padx=5, pady=5)

        # ===== Expected Lifecycle Section =====
        tk.Label(root, text="Expected Lifecycle (cycles):").grid(row=1, column=0, sticky="e")
        self.lifecycle_entry = tk.Entry(root, width=10)
        self.lifecycle_entry.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        # ===== Execute Button =====
        self.run_button = tk.Button(root, text="Run Analysis", command=self.run_analysis)
        self.run_button.grid(row=2, column=1, pady=15)

        # ===== Console Output Section =====
        self.output_buffer = []
        sys.stdout = TextRedirector(buffer_holder=self.output_buffer)
        sys.stderr = TextRedirector(buffer_holder=self.output_buffer)

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def clean_text(text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)

    def save_output_to_pdf(self, filename="Analysis_Report.pdf"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # ✅ 使用支援 Unicode 的 TTF 字型
        font_path = "fonts/NotoSans-Regular.ttf"  # 確保這個路徑正確
        if not os.path.exists(font_path):
            messagebox.showerror("Font Missing", f"請先下載 NotoSans-Regular.ttf 並放到 fonts 資料夾中。\n缺少：{font_path}")
            return
        pdf.add_font("Noto", "", font_path, uni=True)
        pdf.set_font("Noto", size=10)

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        for line in self.output_buffer:
            for subline in line.split('\n'):
                try:
                    pdf.cell(0, 8, txt=subline, ln=True)
                except Exception as e:
                    print(f"[Warning] 跳過一行無法編碼：{subline[:30]}...")

        pdf.output(filename)
        print(f"[PDF] 成功輸出報告：{filename}")

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

            print("\n=== 分析流程全部完成，準備產出 PDF 報告 ===\n")
            self.save_output_to_pdf()

        except Exception as e:
            messagebox.showerror("Error during analysis", str(e))

        self.save_output_to_pdf()

    
    def clean_text(text):
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', text)
        
    def show_output_window(self, content):
        output_window = tk.Toplevel(self.root)
        output_window.title("Analysis Output Log")
        text_widget = tk.Text(output_window, height=30, width=100, bg='black', fg='white')
        text_widget.pack(padx=10, pady=10)
        text_widget.insert(tk.END, content)
        text_widget.config(state='disabled')
    

class TextRedirector:
    def __init__(self, buffer_holder=None):
        self.buffer_holder = buffer_holder
        

    def write(self, message):
        if self.buffer_holder is not None:
            self.buffer_holder.append(message)

    def flush(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = LifetimeGUI(root)
    root.mainloop()

import tkinter as tk
import sys
import re
from tkinter import filedialog, messagebox
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
from fpdf import FPDF
from fpdf.enums import XPos, YPos
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
<<<<<<< HEAD
        self.output_text = tk.Text(root, height=20, width=90, state='disabled', bg='black', fg='white')
        self.output_text.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        sys.stdout = TextRedirector(self.output_text)
        sys.stderr = TextRedirector(self.output_text)

=======
        self.output_buffer = []
        sys.stdout = TextRedirector(buffer_holder=self.output_buffer)
        sys.stderr = TextRedirector(buffer_holder=self.output_buffer)
>>>>>>> ef3c4b8c69031195922907dc3f4aef789606f6c8

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
        font_path_regular = "fonts/NotoSans-Regular.ttf"
        font_path_bold = "fonts/NotoSans-Bold.ttf"
        if not os.path.exists(font_path_regular):
            messagebox.showerror("Font Missing", f"請先下載 NotoSans-Regular.ttf 並放到 fonts 資料夾中。\n缺少：{font_path_regular}")
            return
        pdf.add_font("Noto", "", font_path_regular)
        pdf.add_font("Noto", "B", font_path_bold)
        pdf.set_font("Noto", size=10,)            # 正常字體
        pdf.set_font("Noto", style="B", size=14) # 粗體字體

        for line in self.output_buffer:
            for subline in line.split('\n'):
                subline = subline.strip()
                if not subline:
                    continue

                if subline.startswith("[BOLD14]"):
                    text = subline.replace("[BOLD14]", "").strip()
                    pdf.set_font("Noto", "B", 14)
                    pdf.cell(0, 8, text=text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                    pdf.set_font("Noto", "", 10)
                else:
                    pdf.set_font("Noto", "", 10)
                    pdf.cell(0, 5, text=subline, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
        pdf.output(filename)

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

            messagebox.showinfo("Success", "Analysis completed successfully! PDF report is saved!")
            self.save_output_to_pdf()

        except Exception as e:
            messagebox.showerror("Error during analysis", str(e))
<<<<<<< HEAD

class TextRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.configure(state='normal')
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)
        self.widget.configure(state='disabled')
=======

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
>>>>>>> ef3c4b8c69031195922907dc3f4aef789606f6c8

    def flush(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = LifetimeGUI(root)
    root.mainloop()

import tkinter as tk
import sys
import os
import matplotlib.pyplot as plt
import threading
import subprocess
import platform
from tkinter import filedialog, messagebox, ttk
from better import LifetimeAnalyzer, LifetimeAnalyzerMC, LifetimeAnalyzerPlot
from report_builder import ReportBuilder
from io import BytesIO


class LoadingDialog:
    def __init__(self, parent):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Processing...")
        self.dialog.geometry("300x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center the dialog
        self.dialog.geometry("+%d+%d" % (parent.winfo_rootx() + 50, parent.winfo_rooty() + 50))
        
        # Create loading label
        tk.Label(self.dialog, text="Running Analysis...", font=('Arial', 12)).pack(pady=20)
        
        # Create progress bar (indeterminate style for one-direction animation)
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar")
        
        self.progress = ttk.Progressbar(
            self.dialog, 
            mode='indeterminate', 
            length=200,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=10)
        
        # Start the progress bar animation (slower speed for smoother one-direction effect)
        self.progress.start(8)
        
        # Disable the parent window
        self.dialog.protocol("WM_DELETE_WINDOW", lambda: None)  # Prevent closing
        
    def destroy(self):
        self.progress.stop()
        self.dialog.grab_release()
        self.dialog.destroy()


class LifetimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lifetime Analysis GUI")

        self.file_path = ""
        self.figures = []
        self.output_buffer = []
        self.loading_dialog = None

        # Don't redirect stdout/stderr to avoid capturing prints from better.py
        # self.output_buffer will be manually populated in run_analysis()

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
        self.run_button = tk.Button(root, text="Run Analysis", command=self.start_analysis_thread)
        self.run_button.grid(row=2, column=1, pady=15)

        # Bind Enter key to both entry fields and the root window
        self.file_entry.bind('<Return>', self.on_enter_pressed)
        self.lifecycle_entry.bind('<Return>', self.on_enter_pressed)
        self.root.bind('<Return>', self.on_enter_pressed)
        
        # Focus on the first entry field
        self.file_entry.focus_set()

    def on_enter_pressed(self, event):
        """Handle Enter key press"""
        self.start_analysis_thread()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.file_path = path
            self.file_entry.delete(0, tk.END)
            self.file_entry.insert(0, path)

    def save_figure_to_memory(self, fig):
        """Save matplotlib figure to BytesIO"""
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

    def open_pdf_file(self, filepath):
        """Open PDF file using system default application"""
        try:
            if platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', filepath))
            elif platform.system() == 'Windows':  # Windows
                os.startfile(filepath)
            else:  # Linux and other Unix systems
                subprocess.call(('xdg-open', filepath))
        except Exception as e:
            print(f"Could not open PDF file: {e}")
            messagebox.showwarning("Warning", f"Analysis completed but could not open PDF automatically.\nFile saved at: {filepath}")

    def start_analysis_thread(self):
        """Start analysis in a separate thread with loading dialog"""
        if not self.file_path:
            messagebox.showerror("Error", "Please select a CSV file.")
            return

        try:
            expected_cycles = int(self.lifecycle_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid expected lifecycle number.")
            return

        # Prevent multiple simultaneous analyses
        if hasattr(self, 'analysis_running') and self.analysis_running:
            return
            
        self.analysis_running = True

        # Show loading dialog
        self.loading_dialog = LoadingDialog(self.root)
        
        # Disable the run button
        self.run_button.config(state='disabled')
        
        # Start analysis in separate thread
        analysis_thread = threading.Thread(target=self.run_analysis, args=(expected_cycles,))
        analysis_thread.daemon = True
        analysis_thread.start()

    def run_analysis(self, expected_cycles):
        """Run the actual analysis (called in separate thread)"""
        try:
            # Clear previous results
            self.output_buffer.clear()
            self.figures.clear()
            
            # Create a custom TextRedirector that only captures for PDF
            pdf_text_buffer = []
            
            # Suppress console output but capture for PDF
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Create custom stdout that captures structured output for PDF
            sys.stdout = TextRedirector(pdf_text_buffer)
            sys.stderr = StringIO()  # Suppress errors from console
            
            try:
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

                # Add KS test results to captured output
                pdf_text_buffer.append("[BOLD14] === [4] Kolmogorov-Smirnov Test ===")
                pdf_text_buffer.append(f"KS Statistic (D): {ks_result['ks_stat']:.4f}")
                pdf_text_buffer.append(f"P-value: {ks_result['p_value']:.4f}")
                pdf_text_buffer.append(f"Conclusion: {ks_result['conclusion']}")

            finally:
                # Always restore stdout/stderr
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Set the captured output for PDF generation
            self.output_buffer = pdf_text_buffer

            # Generate PDF report using ReportBuilder with original captured output
            report_filename = "Analysis_Report.pdf"
            report = ReportBuilder(self.output_buffer, self.figures)
            report.build(report_filename)

            # Get full path of the generated PDF
            pdf_path = os.path.abspath(report_filename)

            # Schedule GUI updates in main thread
            self.root.after(0, self.analysis_completed, pdf_path)

        except Exception as e:
            # Schedule error message in main thread
            self.root.after(0, self.analysis_failed, str(e))

    def analysis_completed(self, pdf_path):
        """Called when analysis is completed successfully"""
        # Reset analysis flag
        self.analysis_running = False
        
        # Close loading dialog first
        if self.loading_dialog:
            self.loading_dialog.destroy()
            self.loading_dialog = None
        
        # Re-enable run button
        self.run_button.config(state='normal')
        
        # Open PDF file automatically (no success message popup)
        self.open_pdf_file(pdf_path)

    def analysis_failed(self, error_message):
        """Called when analysis fails"""
        # Reset analysis flag
        self.analysis_running = False
        
        # Close loading dialog
        if self.loading_dialog:
            self.loading_dialog.destroy()
            self.loading_dialog = None
        
        # Re-enable run button
        self.run_button.config(state='normal')
        
        # Show error message
        messagebox.showerror("Error during analysis", f"Error details: {error_message}")
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
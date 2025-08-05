from fpdf import FPDF
from fpdf.enums import XPos, YPos
import textwrap
import os
import tempfile

class ReportBuilder:
    def __init__(self, text_buffer, figures):
        self.text_buffer = text_buffer
        self.figures = figures
        self.pdf = FPDF()
        
    def setup_pdf(self):
        """初始化PDF設置"""
        # 設置頁面邊距 - 使用較保守的設置
        self.pdf.set_margins(left=10, top=15, right=10)
        self.pdf.set_auto_page_break(auto=True, margin=25)
        
        # 使用內建字體，避免字體文件問題
        self.pdf.set_font("Arial", size=10)
        
        # Calculate effective width
        self.effective_width = self.pdf.w - self.pdf.l_margin - self.pdf.r_margin

    def add_text(self):
        """添加文字內容"""
        self.pdf.add_page()
        
        for line in self.text_buffer:
            # 處理字符串列表
            if isinstance(line, list):
                line = ' '.join(str(item) for item in line)
            
            # 確保是字符串
            line = str(line)
            
            # 按換行符分割
            for subline in line.split('\n'):
                subline = subline.strip()
                if not subline:
                    self.pdf.ln(5)  # Empty line
                    continue

                # 處理粗體標題
                if subline.startswith("[BOLD14]"):
                    text = subline.replace("[BOLD14]", "").strip()
                    self.pdf.set_font("Arial", "B", 14)
                    
                    # 使用較小的寬度值來確保文字適合頁面
                    wrapped_lines = textwrap.wrap(text, width=100)
                    for wrapped_line in wrapped_lines:
                        self.pdf.cell(0, 8, wrapped_line, ln=True)
                    
                    self.pdf.set_font("Arial", "", 10)
                    self.pdf.ln(3)  # Add spacing after title
                
                else:
                    # Regular text
                    self.pdf.set_font("Arial", "", 10)
                    
                    # 使用較小的寬度值
                    wrapped_lines = textwrap.wrap(subline, width=70)
                    for wrapped_line in wrapped_lines:
                        self.pdf.cell(0, 5, wrapped_line, ln=True)

    def add_figures(self):
        """Add images"""
        for i, fig_buf in enumerate(self.figures):
            self.pdf.add_page()
            
            try:
                # Save BytesIO as temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    fig_buf.seek(0)  # Ensure reading from beginning
                    temp_file.write(fig_buf.read())
                    temp_file_path = temp_file.name
                
                # Add image to PDF, adjust size to fit page
                # Use conservative size settings
                img_width = min(170, self.effective_width)
                self.pdf.image(temp_file_path, x=20, y=30, w=img_width)
                
                # Clean up temporary file
                os.unlink(temp_file_path)
                
            except Exception as e:
                print(f"Error adding image {i+1}: {e}")
                # Add error message to PDF
                self.pdf.set_font("Arial", "", 12)
                self.pdf.cell(0, 10, f"Image {i+1} failed to load: {str(e)}", ln=True)

    def build(self, filename="Analysis_Report.pdf"):
        """Generate PDF report"""
        try:
            self.setup_pdf()
            self.add_text()
            self.add_figures()
            self.pdf.output(filename)
            print(f"PDF report generated: {filename}")
            
        except Exception as e:
            print(f"Error generating PDF: {e}")
            raise e
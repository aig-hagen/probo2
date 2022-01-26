from fpdf import FPDF
class Validation_Report(FPDF):

    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297

    def header(self):
        self.set_font('Arial', 'B', 11)
        self.cell(self.WIDTH - 80)
        self.cell(60, 1, 'Validation Report', 0, 0, 'R')
        self.ln(20)

    def summary_title(self,tag):
        self.set_font('Arial', '', 12)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 6, f'Summary {tag}', 0, 1, 'L', 1)
        # Line break
        self.ln(4)

    def summary_body(self,summary_text):
          # Times 12
        self.set_font('Times', '', 12)
        # Output justified text
        self.multi_cell(0, 5,summary_text)
        # Line break
        self.ln()
        # Mention in italics
        self.set_font('', 'I')
        self.cell(0, 5, '(end of excerpt)')

    def print_summary(self):
        self.add_page()
        self.summary_title()
        self.summary_body()



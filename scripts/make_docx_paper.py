import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def create_docx(md_path, docx_path):
    doc = docx.Document()
    
    # Page Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    i = 0
    in_code = False
    code_text = []

    while i < len(lines):
        line = lines[i]

        if line.startswith('```'):
            if in_code:
                # Flush code block
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.2)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(6)
                run = p.add_run('\n'.join(code_text))
                run.font.name = 'Consolas'
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(30, 30, 30)
                code_text = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        elif in_code:
            code_text.append(line)
            i += 1
            continue

        if line.startswith('# '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(8)
            run = p.add_run(line[2:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(18)
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif line.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line[3:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13)
            run.font.color.rgb = RGBColor(30, 58, 138)
        elif line.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(line[4:])
            run.bold = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(15, 23, 42)
        elif line.startswith('![') and '](' in line and line.endswith(')'):
            alt = line[2:line.index('](')]
            src = line[line.index('](')+2:-1]
            if os.path.exists(src):
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before = Pt(8)
                p.paragraph_format.space_after = Pt(2)
                run = p.add_run()
                run.add_picture(src, width=Inches(5.8))
        elif line.startswith('**Figure ') or line.startswith('**Table '):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(8)
            clean_text = line.replace('**', '')
            run = p.add_run(clean_text)
            run.italic = True
            run.font.name = 'Times New Roman'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(80, 80, 80)
        elif line.startswith('|') and line.endswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].startswith('|') and lines[i].endswith('|'):
                table_lines.append(lines[i])
                i += 1
            i -= 1

            # Check if this table has images inside
            if any('![' in tl for tl in table_lines):
                # Images side by side
                for tl in table_lines:
                    if '---' in tl:
                        continue
                    cells = [c.strip() for c in tl.strip('|').split('|')]
                    for c in cells:
                        if c.startswith('!['):
                            src = c[c.index('](')+2:-1]
                            if os.path.exists(src):
                                p = doc.add_paragraph()
                                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                                p.add_run().add_picture(src, width=Inches(5.5))
                        elif c.startswith('**Figure '):
                            p = doc.add_paragraph()
                            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                            run = p.add_run(c.replace('**', ''))
                            run.italic = True
                            run.font.size = Pt(9.5)
            else:
                # Regular data table
                rows_data = []
                for tl in table_lines:
                    if '---' in tl:
                        continue
                    rows_data.append([c.strip() for c in tl.strip('|').split('|')])

                if rows_data:
                    num_rows = len(rows_data)
                    num_cols = len(rows_data[0])
                    table = doc.add_table(rows=num_rows, cols=num_cols)
                    table.alignment = WD_TABLE_ALIGNMENT.CENTER

                    for r_idx, row in enumerate(rows_data):
                        for c_idx, val in enumerate(row):
                            cell = table.cell(r_idx, c_idx)
                            cell.text = val.replace('**', '').replace('$', '')
                            for p in cell.paragraphs:
                                p.paragraph_format.space_before = Pt(2)
                                p.paragraph_format.space_after = Pt(2)
                                for run in p.runs:
                                    run.font.name = 'Times New Roman'
                                    run.font.size = Pt(9)
                                    if r_idx == 0:
                                        run.bold = True
                            if r_idx == 0:
                                set_cell_background(cell, "F1F5F9")
                            elif r_idx % 2 == 1:
                                set_cell_background(cell, "FAFAFA")
                    
                    doc.add_paragraph().paragraph_format.space_after = Pt(6)

        elif line.strip() == '---':
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
        elif line.strip():
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.line_spacing = 1.15
            
            # Author formatting
            if 'Author 1' in line:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line.replace('**', ''))
                run.bold = True
                run.font.name = 'Times New Roman'
                run.font.size = Pt(11)
            elif 'Department of Computer Science' in line or 'Institution Name' in line or 'Emails:' in line:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run(line.replace('*', ''))
                run.italic = True
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(60, 60, 60)
            elif line.startswith('> '):
                p.paragraph_format.left_indent = Inches(0.3)
                run = p.add_run(line[2:])
                run.italic = True
                run.font.size = Pt(9.5)
            else:
                # Add text with basic markdown handling
                clean_text = line.replace('**', '').replace('$', '')
                run = p.add_run(clean_text)
                run.font.name = 'Times New Roman'
                run.font.size = Pt(10.5)

        i += 1

    doc.save(docx_path)
    print(f"Created Word document: {docx_path}")

if __name__ == '__main__':
    create_docx('FULL_CONFERENCE_PAPER.md', 'FULL_CONFERENCE_PAPER.docx')

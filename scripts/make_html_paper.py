import os
import re

def convert_md_to_html(md_path, html_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        text = f.read()

    lines = text.split('\n')
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Detection-Confirmed Emergency Vehicle Priority at a Simulated Four-Way Intersection</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {delimiters: [{left: '$$', right: '$$', display: true}, {left: '$', right: '$', display: false}]});"></script>
    <style>
        body {
            font-family: 'Times New Roman', Times, serif;
            line-height: 1.55;
            color: #111;
            max-width: 960px;
            margin: 40px auto;
            padding: 0 30px;
            font-size: 14.5px;
            background-color: #fafbfc;
        }
        .paper-container {
            background: #ffffff;
            padding: 50px 60px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            border-radius: 4px;
        }
        h1 {
            text-align: center;
            font-size: 23px;
            margin-bottom: 12px;
            font-weight: bold;
            color: #0b1a30;
            line-height: 1.3;
        }
        .authors {
            text-align: center;
            margin-bottom: 25px;
            font-size: 13.5px;
            color: #333;
            line-height: 1.4;
        }
        .abstract-box {
            background: #f8fafc;
            border-left: 4px solid #1e3a8a;
            padding: 15px 20px;
            margin: 25px 0;
            font-size: 13px;
            text-align: justify;
        }
        h2 {
            font-size: 16px;
            border-bottom: 1.5px solid #222;
            padding-bottom: 4px;
            margin-top: 35px;
            text-transform: uppercase;
            font-weight: bold;
            color: #1e3a8a;
            letter-spacing: 0.5px;
        }
        h3 {
            font-size: 14.5px;
            margin-top: 22px;
            font-weight: bold;
            color: #111;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 24px 0;
            font-size: 12.5px;
        }
        th, td {
            border: 1px solid #cbd5e1;
            padding: 7px 11px;
            text-align: left;
        }
        th {
            background-color: #f1f5f9;
            font-weight: bold;
            color: #0f172a;
        }
        tr:nth-child(even) {
            background-color: #f8fafc;
        }
        img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 18px auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .caption {
            text-align: center;
            font-size: 12px;
            font-style: italic;
            color: #475569;
            margin-top: -6px;
            margin-bottom: 24px;
        }
        pre {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 14px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 11.5px;
            font-family: Consolas, monospace;
            line-height: 1.4;
        }
        hr {
            border: 0;
            border-top: 1px solid #e2e8f0;
            margin: 30px 0;
        }
        .badge {
            display: inline-block;
            background: #e0f2fe;
            color: #0369a1;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11.5px;
            font-weight: bold;
            margin-bottom: 15px;
        }
        @media print {
            body {
                background: #fff;
                padding: 0;
                margin: 0;
                font-size: 11pt;
            }
            .paper-container {
                box-shadow: none;
                padding: 0;
            }
        }
    </style>
</head>
<body>
<div class="paper-container">
"""

    i = 0
    in_code = False

    while i < len(lines):
        line = lines[i]
        
        # Code blocks
        if line.startswith('```'):
            if in_code:
                html += '</pre>\n'
                in_code = False
            else:
                html += '<pre>'
                in_code = True
            i += 1
            continue
        elif in_code:
            html += line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;') + '\n'
            i += 1
            continue

        # Headings
        if line.startswith('# '):
            html += f'<h1>{line[2:]}</h1>\n'
        elif line.startswith('## '):
            heading = line[3:]
            html += f'<h2>{heading}</h2>\n'
        elif line.startswith('### '):
            html += f'<h3>{line[4:]}</h3>\n'
        # Images
        elif line.startswith('![') and '](' in line and line.endswith(')'):
            alt = line[2:line.index('](')]
            src = line[line.index('](')+2:-1]
            html += f'<img src="{src}" alt="{alt}">\n'
        elif line.startswith('**Figure ') or line.startswith('**Table '):
            html += f'<p class="caption">{line}</p>\n'
        # Markdown tables
        elif line.startswith('|') and line.endswith('|'):
            table_lines = []
            while i < len(lines) and lines[i].startswith('|') and lines[i].endswith('|'):
                table_lines.append(lines[i])
                i += 1
            i -= 1
            
            # Check if this is a side-by-side image table or data table
            is_image_table = any('![' in tl for tl in table_lines)
            if is_image_table:
                html += '<div style="display: flex; gap: 20px; justify-content: center; margin: 20px 0;">\n'
                for row_idx, r in enumerate(table_lines):
                    if '---' in r:
                        continue
                    cells = [c.strip() for c in r.strip('|').split('|')]
                    for cell in cells:
                        if cell.startswith('!['):
                            alt = cell[2:cell.index('](')]
                            src = cell[cell.index('](')+2:-1]
                            html += f'  <div style="flex: 1; text-align: center;"><img src="{src}" alt="{alt}" style="max-height: 280px; width: auto; margin: 0 auto 8px auto;">\n'
                        elif cell.startswith('**Figure '):
                            html += f'  <p class="caption">{cell}</p></div>\n'
                html += '</div>\n'
            else:
                html += '<table>\n'
                for row_idx, r in enumerate(table_lines):
                    if row_idx == 1 and '---' in r:
                        continue
                    cells = [c.strip() for c in r.strip('|').split('|')]
                    tag = 'th' if row_idx == 0 else 'td'
                    html += '  <tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>\n'
                html += '</table>\n'
        elif line.strip() == '---':
            html += '<hr>\n'
        elif line.strip():
            # Check for authors block
            if 'Author 1' in line:
                html += f'<div class="authors"><p><strong>{line}</strong><br>'
                while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].startswith('---') and not lines[i+1].startswith('##'):
                    i += 1
                    html += lines[i] + '<br>'
                html += '</p></div>\n'
            elif line.startswith('> '):
                html += f'<blockquote>{line[2:]}</blockquote>\n'
            else:
                html += f'<p>{line}</p>\n'
                
        i += 1

    html += """
</div>
</body>
</html>
"""
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Created HTML paper: {html_path}")

if __name__ == '__main__':
    convert_md_to_html('FULL_CONFERENCE_PAPER.md', 'FULL_CONFERENCE_PAPER.html')

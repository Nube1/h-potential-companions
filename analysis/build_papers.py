"""Build readable PDF manuscripts and standalone LaTeX from Markdown sources.

PDF uses ReportLab; LaTeX is also emitted for conventional journal typesetting.
Equations in the PDF are deliberately linear ASCII notation, never rasterised.
"""
from pathlib import Path
import re
from html import escape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
styles=getSampleStyleSheet()
styles['Heading2'].keepWithNext=True
styles['Heading3'].keepWithNext=True
styles.add(ParagraphStyle(name='BodyResearch',fontName='Times-Roman',fontSize=11,leading=15,spaceAfter=8,alignment=TA_LEFT))
styles.add(ParagraphStyle(name='PaperTitle',fontName='Times-Bold',fontSize=18,leading=22,spaceAfter=14))
styles.add(ParagraphStyle(name='SmallResearch',fontName='Helvetica',fontSize=8,leading=10,spaceAfter=5))
styles.add(ParagraphStyle(name='EquationResearch',fontName='Courier',fontSize=8,leading=11,spaceBefore=5,spaceAfter=10))


def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#B0BAC4'));canvas.line(50,45,A4[0]-50,45)
    canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#52606D'))
    canvas.drawString(50,32,'H-potential | Companion research draft | Synthetic evidence only')
    canvas.drawRightString(A4[0]-50,32,str(doc.page));canvas.restoreState()


def tex_escape(text):
    chars={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(chars.get(c,c) for c in text)


def build(path):
    lines=path.read_text(encoding='utf-8').splitlines();story=[];tex=[];i=0
    title=lines[0][2:]
    tex.append(r'\documentclass[11pt,a4paper]{article}'+'\n'+r'\usepackage[margin=25mm]{geometry}'+'\n'+r'\usepackage[T1]{fontenc}'+'\n'+r'\usepackage{lmodern,amsmath,amssymb,longtable,booktabs,hyperref}'+'\n'+r'\hypersetup{hidelinks}'+'\n'+r'\title{'+tex_escape(title)+'}\n'+r'\author{Ronald Rethabile Malamule Mahomane}'+'\n'+r'\date{Companion research draft}'+'\n'+r'\begin{document}\maketitle'+'\n')
    while i<len(lines):
        line=lines[i]
        if not line.strip(): i+=1;continue
        if line.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):
                block.append(lines[i]);i+=1
            if any(len(x)>89 for x in block): raise ValueError('Overlong equation in '+path.name)
            story.append(Preformatted('\n'.join(block),styles['EquationResearch']))
            tex.extend([r'\begin{verbatim}',*block,r'\end{verbatim}']);i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(set(x)<=set('-: ') for x in row): rows.append(row)
                i+=1
            cells=[[Paragraph(escape(x),styles['SmallResearch']) for x in row] for row in rows]
            widths=[(A4[0]-100)/len(rows[0])]*len(rows[0])
            table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF3')),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#52728A')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F8FAFC')])]))
            story.extend([table,Spacer(1,10)])
            tex.append(r'\begin{longtable}{'+'l'*len(rows[0])+'}')
            tex.extend(' & '.join(tex_escape(x) for x in row)+r' \\' for row in rows)
            tex.append(r'\end{longtable}');continue
        if line.startswith('# '):
            story.append(Paragraph(escape(line[2:]),styles['PaperTitle']));i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(escape(line[4:]),styles['Heading3']));tex.append(r'\subsection*{'+tex_escape(line[4:])+'}');i+=1;continue
        if line.startswith('## '):
            story.append(Paragraph(escape(line[3:]),styles['Heading2']));tex.append(r'\section*{'+tex_escape(line[3:])+'}');i+=1;continue
        paragraph=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```')):
            paragraph.append(lines[i]);i+=1
        value=' '.join(paragraph)
        style=styles['SmallResearch'] if value.startswith('Research manuscript') or value.startswith('Computational methods') or value.startswith('Theoretical and computational') else styles['BodyResearch']
        story.append(Paragraph(escape(value),style))
        # author already printed by maketitle
        if value!='Ronald Rethabile Malamule Mahomane':tex.append(tex_escape(value)+'\n')
    doc=SimpleDocTemplate(str(path.with_suffix('.pdf')),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=46,bottomMargin=60,title=title,author='Ronald Rethabile Malamule Mahomane')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    tex.append(r'\end{document}');path.with_suffix('.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    print(path.with_suffix('.pdf').name)


if __name__=='__main__':
    for p in sorted((ROOT/'papers').glob('0*.md')):build(p)

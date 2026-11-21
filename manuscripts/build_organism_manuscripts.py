"""Build arXiv-style and Nature-style DOCX versions of the organism manuscript."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "manuscript" / "organism_manuscript.md"
FIG = ROOT / "manuscript" / "figure1_architecture.png"
AUTHOR = "Pidong Xu"
AFFIL = ("School of Materials Science and Engineering, East China University of Technology, "
         "Nanchang, Jiangxi 330013, China")
EMAIL = "13394150188@163.com"
DATE = "19 November 2026"


def parse_md():
    lines = SRC.read_text(encoding="utf-8").splitlines()
    # header
    title = lines[0].lstrip("# ").strip()
    author = "Pidong Xu"
    affiliation = AFFIL
    corresponding = EMAIL
    body = []
    i = 0
    # Skip the title/author/affiliation/date block; it is rendered on the title page.
    while i < len(lines) and not lines[i].startswith("## "):
        i += 1
    while i < len(lines):
        line = lines[i]
        if line.startswith("## References"):
            body.append(("heading", "References", 1, line))
            i += 1
            refs = []
            while i < len(lines) and not lines[i].startswith("## "):
                if lines[i].strip():
                    refs.append(lines[i].strip())
                i += 1
            body.append(("refs", refs, 0, "References"))
            continue
        if line.startswith("[[FIG1]]"):
            body.append(("figure", "", 0, line))
            i += 1
            continue
        if line.startswith("### Table "):
            caption = line[4:].strip()
            i += 1
            table_lines = []
            while i < len(lines) and not lines[i].startswith("### ") and not lines[i].startswith("## "):
                if lines[i].strip().startswith("|"):
                    table_lines.append(lines[i].strip())
                i += 1
            body.append(("table", table_lines, 0, caption))
            continue
        if line.startswith("## "):
            body.append(("heading", line[3:].strip(), 1, line))
            i += 1
            continue
        if line.startswith("### "):
            body.append(("heading", line[4:].strip(), 2, line))
            i += 1
            continue
        if line.strip() == "":
            i += 1
            continue
        # paragraph: collect until blank / heading / table / figure
        para = [line.strip()]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith("#") \
                and not lines[i].startswith("[[FIG1]]") and not lines[i].strip().startswith("|"):
            para.append(lines[i].strip())
            i += 1
        text = " ".join(para)
        text = text.replace("**", "").replace("\\*", "*").replace("^1", "1")
        text = re.sub(r"\*Corresponding author:", "Corresponding author:", text)
        body.append(("para", text, 0, text))
    return title, author, affiliation, corresponding, body


def line_numbers(section):
    ln = OxmlElement("w:lnNumType")
    ln.set(qn("w:countBy"), "1")
    ln.set(qn("w:restart"), "continuous")
    section._sectPr.append(ln)


def page_numbers(section, prefix=""):
    p = section.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(prefix)
    f1 = OxmlElement("w:fldChar"); f1.set(qn("w:fldCharType"), "begin")
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = " PAGE "
    f2 = OxmlElement("w:fldChar"); f2.set(qn("w:fldCharType"), "end")
    r._r.append(f1); r._r.append(it); r._r.append(f2)
    r.font.name = "Times New Roman"; r.font.size = Pt(9)


def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill); tcPr.append(shd)


def style_base(doc, nature: bool):
    sec = doc.sections[0]
    sec.top_margin = Inches(1.0); sec.bottom_margin = Inches(1.0)
    sec.left_margin = Inches(1.0); sec.right_margin = Inches(1.0)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12 if nature else 11)
    normal._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Times New Roman")
    normal.paragraph_format.line_spacing_rule = (
        WD_LINE_SPACING.DOUBLE if nature else WD_LINE_SPACING.ONE_POINT_FIVE)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.space_before = Pt(0)
    for name, size in (("Heading 1", 14 if nature else 12), ("Heading 2", 12), ("Heading 3", 11)):
        st = doc.styles[name]
        st.font.name = "Times New Roman"; st.font.size = Pt(size)
        st.font.bold = True; st.font.color.rgb = RGBColor(0, 0, 0)
        st.paragraph_format.space_before = Pt(12 if nature else 8)
        st.paragraph_format.space_after = Pt(6 if nature else 3)
        st.paragraph_format.line_spacing_rule = (
            WD_LINE_SPACING.DOUBLE if nature else WD_LINE_SPACING.ONE_POINT_FIVE)
    if nature:
        line_numbers(sec)
        page_numbers(sec)
    else:
        page_numbers(sec, "arXiv preprint · ")


def add_para(doc, text, nature, style=None, bold=False, italic=False, size=None,
             align=WD_ALIGN_PARAGRAPH.JUSTIFY, first_indent=0.0, hanging=False):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    p.paragraph_format.space_after = Pt(0 if nature else 3)
    if first_indent:
        p.paragraph_format.first_line_indent = Inches(first_indent)
    if hanging:
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.first_line_indent = Inches(-0.25)
    pieces = re.split(r"(\[\d+(?:,\s*\d+)*\])", text)
    for piece in pieces:
        if not piece:
            continue
        r = p.add_run(piece)
        r.font.name = "Times New Roman"
        if size:
            r.font.size = Pt(size)
        if bold:
            r.bold = True
        if italic:
            r.italic = True
        if re.fullmatch(r"\[\d+(?:,\s*\d+)*\]", piece):
            r.font.superscript = True
    return p


def add_table(doc, caption, markdown_lines, nature):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    if caption.startswith("Table 4") or (nature and caption.startswith("Table 6")):
        p.paragraph_format.page_break_before = True
    r = p.add_run(caption.replace("|", "|"))
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(9.5 if nature else 9)
    rows = []
    for line in markdown_lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
            continue
        rows.append(cells)
    if not rows:
        return
    table = doc.add_table(rows=1, cols=len(rows[0]))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    # Repeat the header row and prevent a single record from splitting across pages.
    trPr = table.rows[0]._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)
    for i, h in enumerate(rows[0]):
        c = table.rows[0].cells[i]
        c.text = ""
        pp = c.paragraphs[0]
        rr = pp.add_run(h)
        rr.bold = True
        rr.font.name = "Times New Roman"
        rr.font.size = Pt(8.5 if nature else 8)
        shade(c, "E8EEEE")
    for row in rows[1:]:
        new_row = table.add_row()
        new_trPr = new_row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        new_trPr.append(cant)
        cells = new_row.cells
        for i, val in enumerate(row[:len(table.columns)]):
            cells[i].text = ""
            rr = cells[i].paragraphs[0].add_run(val.replace("**", ""))
            rr.font.name = "Times New Roman"
            rr.font.size = Pt(8.5 if nature else 8)
    q = doc.add_paragraph()
    q.paragraph_format.space_after = Pt(8)
    q.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    r = q.add_run("")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8)


def build(nature: bool, out: Path):
    title, author, affiliation, corresponding, body = parse_md()
    doc = Document()
    style_base(doc, nature)

    # title / author block
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(20 if not nature else 70)
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run(title); r.bold = True; r.font.name = "Times New Roman"
    r.font.size = Pt(18 if nature else 20)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(author); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(13)
    r = p.add_run("1,*"); r.font.superscript = True; r.font.size = Pt(10)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("1"); r.font.superscript = True; r.font.size = Pt(9)
    r = p.add_run(affiliation); r.italic = True; r.font.name = "Times New Roman"; r.font.size = Pt(10)

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Corresponding author: "); r.font.name = "Times New Roman"; r.font.size = Pt(10)
    r = p.add_run(f"Pidong Xu, {corresponding}"); r.font.name = "Times New Roman"; r.font.size = Pt(10)
    r.underline = True

    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(24 if not nature else 40)
    r = p.add_run("Preprint · prepared for arXiv" if not nature else "Article manuscript · Nature-style format")
    r.italic = True; r.font.name = "Times New Roman"; r.font.size = Pt(9)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(f"Manuscript date: {DATE}"); r.font.name = "Times New Roman"; r.font.size = Pt(9)
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Project: Christus (formerly WPHR)"); r.font.name = "Times New Roman"; r.font.size = Pt(9)
    r.italic = True

    if nature:
        doc.add_page_break()

    for kind, content, level, meta in body:
        if kind == "heading":
            h = doc.add_heading(level=1 if level == 1 else 2)
            h.alignment = WD_ALIGN_PARAGRAPH.LEFT
            r = h.add_run(content)
            r.font.name = "Times New Roman"
            r.font.size = Pt(14 if level == 1 else 12)
            r.bold = True
            r.font.color.rgb = RGBColor(0, 0, 0)
        elif kind == "para":
            add_para(doc, content, nature, first_indent=0.0)
        elif kind == "figure":
            doc.add_picture(str(FIG), width=Inches(5.5 if nature else 6.0))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = cap.add_run("Figure 1 | Christus as a modular artificial cognitive organism. ")
            r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(9)
            r = cap.add_run("World interaction, perception/event bus, SM-CTRN recurrent core, memory and "
                            "provenance, affect/coordination, self-related representations and auxiliary "
                            "language components. The local Transformer is a semantic reranker, not the core.")
            r.font.name = "Times New Roman"; r.font.size = Pt(9)
        elif kind == "table":
            add_table(doc, meta, content, nature)
        elif kind == "refs":
            for ref in content:
                add_para(doc, ref, nature, size=8.5, hanging=True,
                         align=WD_ALIGN_PARAGRAPH.LEFT)
    doc.save(out)
    return out


def main():
    a = build(False, ROOT / "CHRISTUS_ORGANISM_ARXIV_PREPRINT.docx")
    n = build(True, ROOT / "CHRISTUS_ORGANISM_NATURE_MANUSCRIPT.docx")
    print(a)
    print(n)


if __name__ == "__main__":
    main()

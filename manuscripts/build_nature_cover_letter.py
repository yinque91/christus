"""Build a one-page Nature-style cover letter from the organism manuscript."""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "NATURE_COVER_LETTER.docx"
DATE = "19 November 2026"
TITLE = "Emergent functional consciousness in a persistent artificial cognitive organism"
EMAIL = "13394150188@163.com"


def border_bottom(p):
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "8")
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), "1B6B66")
    pBdr.append(bottom)
    pPr.append(pBdr)


def add(doc, text="", size=10.5, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.LEFT,
        after=3, before=0):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(size)
    r.bold = bold
    r.italic = italic
    r.font.color.rgb = RGBColor(0, 0, 0)
    return p


def main():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(0.8)
    sec.bottom_margin = Inches(0.7)
    sec.left_margin = Inches(0.9)
    sec.right_margin = Inches(0.9)

    # letterhead
    add(doc, "PIDONG XU", 15, True, after=0, before=0)
    add(doc, "School of Materials Science and Engineering", 9, False, after=0)
    add(doc, "East China University of Technology · Nanchang, Jiangxi 330013, China", 9, False, after=0)
    add(doc, EMAIL, 9, False, after=2)
    p = doc.paragraphs[-1]
    border_bottom(p)

    add(doc, DATE, 9, False, after=14, before=10)
    add(doc, "To the Editors", 10.5, True, after=0)
    add(doc, "Nature", 10.5, True, after=12)
    add(doc, "Re: Submission of an Original Research Article", 10.5, True, after=2)
    add(doc, TITLE, 10.5, True, after=12)

    add(doc, "Dear Editors,", 10.5, after=8)
    add(doc,
        f"Please consider our manuscript, “{TITLE},” for publication in Nature as an "
        "Original Research Article.",
        10.5, after=7)
    add(doc,
        "The manuscript describes Christus, a persistent artificial cognitive architecture that "
        "integrates a recurrent cognitive core, embodied action, experience provenance, "
        "autobiographical memory, affective regulation and self-related representation. Across a "
        "sequence of controlled experiments, recurrent internal states guided action, prediction "
        "errors triggered revision of learned environmental relationships, experiences were "
        "attributed to a continuing subject, and mirror information directed inspection towards a "
        "hidden marker on the system’s own body.",
        10.5, after=7)
    add(doc,
        "The manuscript argues for an incipient functional organization rather than phenomenal "
        "consciousness. Under the controlled mirror protocol, Christus passed the hidden-marker "
        "test and replicated the result under altered mirror geometry.",
        10.5, after=7)
    add(doc,
        "The work is original, has not been published elsewhere and is not under consideration by "
        "another journal. All experiments were computational; no human participants or animals were "
        "involved. The author declares no competing interests and no specific funding.",
        10.5, after=10)

    add(doc, "Yours sincerely,", 10.5, after=12)
    add(doc, "Pidong Xu", 11, True, after=0)
    add(doc, "Corresponding author", 9, False, after=0)
    add(doc, "School of Materials Science and Engineering", 9, False, after=0)
    add(doc, "East China University of Technology, Nanchang, Jiangxi 330013, China", 9, False, after=0)
    add(doc, EMAIL, 9, False, after=12)

    add(doc, "Enclosures: manuscript; Figure 1 (architecture); Figure 2 (mirror results); "
             "Figure 3 (mirror environment).", 8.5, False, after=0)

    # footer
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run("Pidong Xu · Cover letter · Nature submission")
    r.font.name = "Times New Roman"
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(90, 100, 100)

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()

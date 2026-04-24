"""Generate the AAA-compliant title page docx for the ARS-VG manuscript."""

from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

TITLE = (
    "ARS-VG: A Graph-Theoretic Intelligent Artifact for Joint Detection of "
    "Accrual-Based and Real-Activities Earnings Management Substitution in "
    "SEC-Registered Firms — A Design Science Research Approach"
)

RUNNING_HEAD = (
    "ARS-VG: Graph-Based Detection of AEM–REM Substitution in "
    "SEC-Registered Firms"
)
assert len(RUNNING_HEAD) <= 115, f"Running head too long: {len(RUNNING_HEAD)}"

doc = Document()

core = doc.core_properties
core.author = "apoorv"
core.last_modified_by = "apoorv"
core.title = TITLE

style = doc.styles["Normal"]
style.font.name = "Times New Roman"
style.font.size = Pt(12)
fmt = style.paragraph_format
fmt.space_before = Pt(0)
fmt.space_after = Pt(0)
fmt.line_spacing = 2.0

for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)


def add_blank():
    doc.add_paragraph("")


def add_centered_bold(text, size=14):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = True
    run.font.size = Pt(size)
    run.font.name = "Times New Roman"
    return p


def add_centered(text, bold=False, italic=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    run.font.name = "Times New Roman"
    run.font.size = Pt(12)
    return p


def add_left(text, bold_label=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    if bold_label:
        r1 = p.add_run(bold_label)
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(12)
    r2 = p.add_run(text)
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(12)
    return p


add_centered_bold(TITLE, size=14)
add_blank()

add_centered("Apoorv Saxena", bold=True)
add_centered("College of Management, Yuan Ze University")
add_centered("Chung-Li, Taoyuan, Taiwan")
add_centered("ORCID: 0009-0002-4268-2436")
add_blank()

add_centered("Yan-Jie Yang*", bold=True)
add_centered("College of Management, Yuan Ze University")
add_centered("Chung-Li, Taoyuan, Taiwan")
add_centered("ORCID: 0000-0001-7346-5050")
add_blank()

add_left("Corresponding author. Email: yanjie@saturn.yzu.edu.tw", bold_label="*")
add_blank()

add_left(
    "ARS-VG: Graph-Based Detection of AEM–REM Substitution in "
    "SEC-Registered Firms",
    bold_label="Running Head: ",
)
add_blank()

add_left("Acknowledgments:", bold_label="")
doc.paragraphs[-1].runs[0].bold = True
add_left(
    "The authors thank colleagues for helpful comments on earlier versions "
    "of this paper. Any remaining errors are our own. This research received "
    "no specific grant from any funding agency in the public, commercial, or "
    "not-for-profit sectors. Apoorv Saxena has no financial conflicts of "
    "interest related to this research. Yan-Jie Yang has no financial "
    "conflicts of interest related to this research."
)
add_blank()

add_left(
    "Apoorv Saxena, Yuan Ze University, College of Management, Chung-Li, "
    "Taoyuan, Taiwan; Yan-Jie Yang, Yuan Ze University, College of Management, "
    "Chung-Li, Taoyuan, Taiwan."
)
add_blank()

add_left(
    "Data and code supporting the findings of this study are available from "
    "the authors on reasonable request.",
    bold_label="Data Availability: ",
)
add_blank()

add_left("M41; M42; C45; G30.", bold_label="JEL Classifications: ")
add_blank()

add_left(
    "Earnings management; AEM–REM substitution; Agency theory; Design "
    "Science Research; Financial knowledge graph; Graph-based anomaly "
    "detection; Forensic accounting.",
    bold_label="Keywords: ",
)
add_blank()

add_left(
    "No.",
    bold_label=(
        "Does this article have supplemental material(s) that are intended "
        "for publication? "
    ),
)

out_path = "/home/user/cli-2/title_page.docx"
doc.save(out_path)
print(f"Wrote {out_path}")
print(f"Running head length: {len(RUNNING_HEAD)} chars")

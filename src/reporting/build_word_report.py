"""Build a simple .docx Word report from the final Markdown report.

This keeps the project self-contained and avoids requiring python-docx.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import re


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = PROJECT_ROOT / "report" / "final_project_report.md"
OUTPUT_PATH = PROJECT_ROOT / "report" / "final_project_report.docx"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass
class TableBlock:
    rows: list[list[str]]


def strip_markdown(text: str) -> str:
    """Remove simple markdown markers for Word output."""

    text = text.replace("**", "")
    text = text.replace("`", "")
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    return text.strip()


def paragraph_xml(
    text: str,
    *,
    style: str | None = None,
    bold: bool = False,
    italic: bool = False,
    spacing_after: int = 120,
) -> str:
    """Create a Word paragraph XML string."""

    escaped = escape(text or "")
    p_pr = []
    if style:
        p_pr.append(f"<w:pStyle w:val=\"{style}\"/>")
    if spacing_after is not None:
        p_pr.append(f"<w:spacing w:after=\"{spacing_after}\"/>")
    p_pr_xml = f"<w:pPr>{''.join(p_pr)}</w:pPr>" if p_pr else ""

    r_pr = []
    if bold:
        r_pr.append("<w:b/>")
    if italic:
        r_pr.append("<w:i/>")
    r_pr_xml = f"<w:rPr>{''.join(r_pr)}</w:rPr>" if r_pr else ""

    return (
        "<w:p>"
        f"{p_pr_xml}"
        "<w:r>"
        f"{r_pr_xml}"
        f"<w:t xml:space=\"preserve\">{escaped}</w:t>"
        "</w:r>"
        "</w:p>"
    )


def table_cell_xml(text: str, width: int) -> str:
    """Create one table cell."""

    return (
        "<w:tc>"
        "<w:tcPr>"
        f"<w:tcW w:w=\"{width}\" w:type=\"dxa\"/>"
        "</w:tcPr>"
        f"{paragraph_xml(text, spacing_after=60)}"
        "</w:tc>"
    )


def table_xml(rows: list[list[str]]) -> str:
    """Create a simple Word table."""

    if not rows:
        return ""

    col_count = max(len(row) for row in rows)
    cell_width = max(1500, int(9000 / max(col_count, 1)))

    tbl_rows = []
    for row_index, row in enumerate(rows):
        padded = row + [""] * (col_count - len(row))
        cells = []
        for value in padded:
            cells.append(table_cell_xml(strip_markdown(value), cell_width))
        tbl_rows.append(
            "<w:tr>"
            + "".join(cells)
            + "</w:tr>"
        )

    return (
        "<w:tbl>"
        "<w:tblPr>"
        "<w:tblStyle w:val=\"TableGrid\"/>"
        "<w:tblW w:w=\"0\" w:type=\"auto\"/>"
        "<w:tblBorders>"
        "<w:top w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"BFC7D5\"/>"
        "<w:left w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"BFC7D5\"/>"
        "<w:bottom w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"BFC7D5\"/>"
        "<w:right w:val=\"single\" w:sz=\"8\" w:space=\"0\" w:color=\"BFC7D5\"/>"
        "<w:insideH w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"D6DCE5\"/>"
        "<w:insideV w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"D6DCE5\"/>"
        "</w:tblBorders>"
        "</w:tblPr>"
        "<w:tblGrid>"
        + "".join(
            f"<w:gridCol w:w=\"{cell_width}\"/>" for _ in range(col_count)
        )
        + "</w:tblGrid>"
        + "".join(tbl_rows)
        + "</w:tbl>"
    )


def parse_markdown(markdown: str) -> list[str | TableBlock]:
    """Convert simple markdown into a sequence of paragraph/table blocks."""

    blocks: list[str | TableBlock] = []
    lines = markdown.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|---"):
            header = [cell.strip() for cell in stripped.strip("|").split("|")]
            i += 2
            rows = [header]
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([cell.strip() for cell in lines[i].strip().strip("|").split("|")])
                i += 1
            blocks.append(TableBlock(rows))
            continue

        blocks.append(line)
        i += 1

    return blocks


def build_document_xml(markdown: str) -> str:
    """Build the main Word document XML."""

    body_parts: list[str] = []

    for block in parse_markdown(markdown):
        if isinstance(block, TableBlock):
            body_parts.append(table_xml(block.rows))
            body_parts.append(paragraph_xml("", spacing_after=100))
            continue

        line = block.strip()

        if line.startswith("# "):
            body_parts.append(paragraph_xml(strip_markdown(line[2:]), style="Title", spacing_after=180))
        elif line.startswith("## "):
            body_parts.append(paragraph_xml(strip_markdown(line[3:]), style="Heading1", spacing_after=140))
        elif line.startswith("### "):
            body_parts.append(paragraph_xml(strip_markdown(line[4:]), style="Heading2", spacing_after=120))
        elif re.match(r"^\d+\.\s+", line):
            body_parts.append(paragraph_xml(strip_markdown(line), spacing_after=80))
        elif line.startswith("- "):
            body_parts.append(paragraph_xml("• " + strip_markdown(line[2:]), spacing_after=80))
        else:
            body_parts.append(paragraph_xml(strip_markdown(line), spacing_after=120))

    sect_pr = (
        "<w:sectPr>"
        "<w:pgSz w:w=\"12240\" w:h=\"15840\"/>"
        "<w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\" "
        "w:header=\"708\" w:footer=\"708\" w:gutter=\"0\"/>"
        "</w:sectPr>"
    )

    return (
        f"<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        f"<w:document xmlns:w=\"{W_NS}\" xmlns:r=\"{R_NS}\">"
        f"<w:body>{''.join(body_parts)}{sect_pr}</w:body>"
        f"</w:document>"
    )


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def document_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:color w:val="1F2937"/>
      <w:sz w:val="34"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="220" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:color w:val="0F4C81"/>
      <w:sz w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="180" w:after="100"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:color w:val="334155"/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
</w:styles>"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties
    xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:dcterms="http://purl.org/dc/terms/"
    xmlns:dcmitype="http://purl.org/dc/dcmitype/"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Final Project Report</dc:title>
  <dc:creator>MachineVision Project</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
            xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
</Properties>"""


def build_docx(markdown_path: Path, output_path: Path) -> None:
    """Create the .docx file."""

    markdown = markdown_path.read_text(encoding="utf-8")
    document_xml = build_document_xml(markdown)

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as docx:
        docx.writestr("[Content_Types].xml", content_types_xml())
        docx.writestr("_rels/.rels", root_rels_xml())
        docx.writestr("word/document.xml", document_xml)
        docx.writestr("word/_rels/document.xml.rels", document_rels_xml())
        docx.writestr("word/styles.xml", styles_xml())
        docx.writestr("docProps/core.xml", core_xml())
        docx.writestr("docProps/app.xml", app_xml())


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_docx(MARKDOWN_PATH, OUTPUT_PATH)
    print(f"Saved Word report to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

"""Render the demo Markdown sources to PDFs (deterministic, reportlab).

Usage: python scripts/render_demo_docs.py [data/demos]

Only a small Markdown subset is supported: "# " title, "## " headings, "- " bullets and
plain paragraphs. That is all the demo sources use.
"""

from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t", parent=base["Title"], fontSize=17, alignment=TA_LEFT, spaceAfter=6
        ),
        "h2": ParagraphStyle(
            "h", parent=base["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4
        ),
        "body": ParagraphStyle("b", parent=base["BodyText"], fontSize=10.5, leading=14.5),
        "note": ParagraphStyle(
            "n", parent=base["BodyText"], fontSize=9, leading=12, textColor="#555555"
        ),
    }


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(source: Path, target: Path) -> None:
    styles = _styles()
    story: list[object] = []
    bullets: list[str] = []
    paragraph: list[str] = []
    first_paragraph = True

    def flush_paragraph() -> None:
        nonlocal first_paragraph
        if paragraph:
            style = styles["note"] if first_paragraph else styles["body"]
            story.append(Paragraph(_escape(" ".join(paragraph)), style))
            story.append(Spacer(1, 4))
            paragraph.clear()
            first_paragraph = False

    def flush_bullets() -> None:
        if bullets:
            items = [ListItem(Paragraph(_escape(b), styles["body"])) for b in bullets]
            story.append(ListFlowable(items, bulletType="bullet", leftIndent=12))
            bullets.clear()

    for raw in source.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            flush_paragraph()
            flush_bullets()
        elif line.startswith("# "):
            story.append(Paragraph(_escape(line[2:]), styles["title"]))
        elif line.startswith("## "):
            flush_paragraph()
            flush_bullets()
            story.append(Paragraph(_escape(line[3:]), styles["h2"]))
        elif line.startswith("- "):
            bullets.append(line[2:])
        else:
            paragraph.append(line)
    flush_paragraph()
    flush_bullets()

    target.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(target),
        pagesize=A4,
        leftMargin=22 * mm,
        rightMargin=22 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        title=source.stem,
        author="ForgeAI demo (synthetic)",
        invariant=1,
    )
    doc.build(story)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/demos")
    for source in sorted(root.glob("*/source/*.md")):
        target = source.parent.parent / "docs" / f"{source.stem}.pdf"
        render(source, target)
        print(f"rendered {target}")


if __name__ == "__main__":
    main()

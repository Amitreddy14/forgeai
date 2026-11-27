"""Builders for small, realistic document folders used across tests."""

from __future__ import annotations

from pathlib import Path

LEAVE_POLICY_PAGES = [
    (
        "Leave Policy\n\n"
        "1. Purpose\n"
        "This policy explains leave entitlements for shore-based staff of the company.\n\n"
        "2. Annual leave\n"
        "Shore staff earn 24 days of annual leave per calendar year. Leave accrues at two "
        "days per completed month of service. Up to 5 unused days may be carried forward "
        "to the next year with manager approval."
    ),
    (
        "3. Sick leave\n"
        "Employees receive 12 days of paid sick leave per year. A medical certificate is "
        "required for absences longer than 2 consecutive days.\n\n"
        "4. Parental leave\n"
        "Birth mothers receive 26 weeks of paid maternity leave. Other parents receive "
        "15 working days of paid parental leave, to be taken within 6 months of the birth."
    ),
]

TRAVEL_POLICY_MD = """# Travel and Expenses Policy

## 1. Booking
All business travel must be booked through the travel desk at least 7 days in advance.

## 2. Daily allowance
The daily meal allowance for domestic travel is INR 1,500. For international travel the
allowance is USD 60 per day. Receipts are required for any single expense above INR 2,000.

## 3. Claims
Expense claims must be submitted within 30 days of returning from travel.
"""

BERTHS_CSV = """berth_id,port,max_loa_m,max_draft_m,daily_rate_usd
B1,Kochi,250,12.5,4200
B2,Kochi,180,9.0,2600
M4,Mumbai,300,14.0,6100
"""


def write_pdf(path: Path, pages: list[str]) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas

    pdf = canvas.Canvas(str(path), pagesize=A4)
    _, height = A4
    for page_text in pages:
        y = height - 60
        for paragraph in page_text.split("\n"):
            # wrap long lines at ~90 chars
            line = ""
            for word in paragraph.split():
                if len(line) + len(word) + 1 > 90:
                    pdf.drawString(50, y, line)
                    y -= 16
                    line = word
                else:
                    line = f"{line} {word}".strip()
            pdf.drawString(50, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()


def write_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    document.add_heading("Code of Conduct", level=1)
    document.add_paragraph("Employees must declare any gift worth more than INR 5,000.")
    document.add_heading("Harassment", level=2)
    document.add_paragraph("Complaints go to the Internal Committee within 90 days.")
    document.save(str(path))


def build_docs_folder(root: Path) -> Path:
    folder = root / "docs"
    folder.mkdir()
    write_pdf(folder / "Leave_Policy.pdf", LEAVE_POLICY_PAGES)
    (folder / "travel-policy.md").write_text(TRAVEL_POLICY_MD, encoding="utf-8")
    (folder / "berths.csv").write_text(BERTHS_CSV, encoding="utf-8")
    write_docx(folder / "code_of_conduct.docx")
    (folder / "notes.txt").write_text("Office hours are 9 to 6 on weekdays.\n", encoding="utf-8")
    (folder / "malware.exe").write_bytes(b"MZ\x00\x00")
    (folder / ".hidden.md").write_text("secret", encoding="utf-8")
    (folder / "empty.txt").write_text("   \n", encoding="utf-8")
    return folder

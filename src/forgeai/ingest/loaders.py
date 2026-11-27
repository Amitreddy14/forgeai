"""Read a folder of documents into a Corpus. Unknown, oversized or unreadable files are
skipped with a reason (and reported), never silently dropped."""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Callable
from pathlib import Path

from forgeai.schemas.corpus import Corpus, DocKind, Document, Page, SkippedFile
from forgeai.textutil import clean

log = logging.getLogger(__name__)

ALLOWED: dict[str, DocKind] = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".csv": "csv",
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
}
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_FILES = 200
MAX_CSV_ROWS = 5_000

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    return _SLUG.sub("-", name.lower()).strip("-") or "doc"


def _keep_lines(text: str) -> str:
    """Clean each line but keep line structure: headings are detected per line later."""
    lines = [clean(line) for line in text.splitlines()]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def _load_pdf(path: Path) -> list[Page]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = _keep_lines(page.extract_text() or "")
        if text:
            pages.append(Page(number=i, text=text))
    return pages


def _load_docx(path: Path) -> list[Page]:
    import docx

    document = docx.Document(str(path))
    lines: list[str] = []
    for para in document.paragraphs:
        text = clean(para.text)
        if not text:
            continue
        style = (para.style.name if para.style is not None else "") or ""
        if style.lower().startswith("heading"):
            level = "".join(ch for ch in style if ch.isdigit()) or "1"
            lines.append("")
            lines.append(f"{'#' * min(int(level), 6)} {text}")
        else:
            lines.append(text)
    for table in document.tables:
        for row in table.rows:
            cells = [clean(c.text) for c in row.cells]
            lines.append(" | ".join(cells))
    text = "\n".join(lines).strip()
    return [Page(number=1, text=text)] if text else []


def _load_text(path: Path) -> list[Page]:
    text = _keep_lines(path.read_text(encoding="utf-8", errors="replace"))
    return [Page(number=1, text=text)] if text else []


def _load_csv(path: Path) -> tuple[list[Page], list[str], list[dict[str, str]]]:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    columns = [clean(c) for c in (reader.fieldnames or []) if c is not None]
    if not columns:
        return [], [], []
    rows: list[dict[str, str]] = []
    for record in reader:
        if len(rows) >= MAX_CSV_ROWS:
            break
        row = {clean(k): clean(v or "") for k, v in record.items() if k is not None}
        if any(row.values()):
            rows.append(row)
    # One "page" per row: row numbers become citable locations.
    pages = [
        Page(number=i, text=f"Row {i}: " + "; ".join(f"{k}: {v}" for k, v in row.items() if v))
        for i, row in enumerate(rows, start=1)
    ]
    return pages, columns, rows


_PAGE_LOADERS: dict[DocKind, Callable[[Path], list[Page]]] = {
    "pdf": _load_pdf,
    "docx": _load_docx,
    "md": _load_text,
    "txt": _load_text,
}


def _title_for(path: Path, pages: list[Page]) -> str:
    for page in pages[:1]:
        for line in page.text.splitlines()[:5]:
            candidate = line.lstrip("# ").strip()
            if 3 <= len(candidate) <= 90:
                return candidate
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def load_folder(folder: Path) -> Corpus:
    """Load every supported file directly inside `folder` (and its subfolders)."""
    if not folder.is_dir():
        raise FileNotFoundError(f"documents folder not found: {folder}")
    documents: list[Document] = []
    skipped: list[SkippedFile] = []
    used_ids: set[str] = set()

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    for path in files:
        rel = path.relative_to(folder).as_posix()
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        kind = ALLOWED.get(path.suffix.lower())
        if kind is None:
            skipped.append(SkippedFile(filename=rel, reason=f"unsupported type {path.suffix}"))
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            skipped.append(SkippedFile(filename=rel, reason="larger than 20 MB"))
            continue
        if len(documents) >= MAX_FILES:
            skipped.append(SkippedFile(filename=rel, reason=f"over the {MAX_FILES}-file limit"))
            continue

        columns: list[str] | None = None
        rows: list[dict[str, str]] | None = None
        try:
            if kind == "csv":
                pages, columns, rows = _load_csv(path)
            else:
                pages = _PAGE_LOADERS[kind](path)
        except Exception as exc:  # parser libraries raise many types; we report, not crash
            log.warning("could not read %s: %s", rel, exc)
            skipped.append(SkippedFile(filename=rel, reason=f"unreadable ({type(exc).__name__})"))
            continue
        if not pages:
            skipped.append(SkippedFile(filename=rel, reason="no extractable text"))
            continue

        doc_id = slugify(path.stem)
        base, n = doc_id, 2
        while doc_id in used_ids:
            doc_id, n = f"{base}-{n}", n + 1
        used_ids.add(doc_id)
        documents.append(
            Document(
                doc_id=doc_id,
                filename=rel,
                kind=kind,
                title=_title_for(path, pages) if kind != "csv" else path.stem,
                pages=pages,
                columns=columns,
                rows=rows,
            )
        )
    return Corpus(root=str(folder), documents=documents, skipped=skipped)

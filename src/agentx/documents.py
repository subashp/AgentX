"""Bounded extraction for public web documents."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from io import BytesIO
from typing import Any


class DocumentExtractionError(RuntimeError):
    """Raised when a document cannot be safely converted to bounded text."""


@dataclass(frozen=True)
class ExtractedDocument:
    media_type: str
    text: str
    title: str = ""
    page_count: int | None = None
    truncated: bool = False

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "media_type": self.media_type,
            "content": self.text,
            "truncated": self.truncated,
        }
        if self.title:
            result["title"] = self.title
        if self.page_count is not None:
            result["page_count"] = self.page_count
        return result


class DocumentExtractor:
    """Extract bounded text from HTML, JSON, plain text, Markdown, or PDF."""

    def __init__(self, *, max_chars: int = 12_000, max_pages: int = 32, pdf_reader_factory: Callable[[BytesIO], Any] | None = None) -> None:
        if isinstance(max_chars, bool) or not 1 <= max_chars <= 100_000:
            raise ValueError("max_chars must be from 1 to 100000")
        if isinstance(max_pages, bool) or not 1 <= max_pages <= 128:
            raise ValueError("max_pages must be from 1 to 128")
        self.max_chars = max_chars
        self.max_pages = max_pages
        self._pdf_reader_factory = pdf_reader_factory

    def extract(self, body: bytes, *, media_type: str, filename: str = "") -> ExtractedDocument:
        if not isinstance(body, bytes):
            raise DocumentExtractionError("document body must be bytes")
        if len(body) > 8 * 1024 * 1024:
            raise DocumentExtractionError("document exceeds the 8 MiB safety limit")
        normalized = media_type.split(";", 1)[0].strip().lower()
        if normalized == "application/pdf" or filename.lower().endswith(".pdf"):
            return self._extract_pdf(body)
        if normalized in {"text/html", "application/xhtml+xml"}:
            parser = _HTMLTextParser()
            parser.feed(body.decode("utf-8", errors="replace"))
            parser.close()
            return self._bounded(normalized, parser.text, title=parser.title)
        if normalized in {"application/json", "text/json"}:
            try:
                value = json.loads(body.decode("utf-8", errors="replace"))
                text = json.dumps(value, indent=2, ensure_ascii=False)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise DocumentExtractionError("document contains invalid JSON") from exc
            return self._bounded(normalized, text)
        if normalized.startswith("text/") or normalized in {"application/xml", "text/xml"}:
            return self._bounded(normalized, body.decode("utf-8", errors="replace"))
        raise DocumentExtractionError(f"unsupported document type: {normalized or 'unknown'}")

    def _extract_pdf(self, body: bytes) -> ExtractedDocument:
        factory = self._pdf_reader_factory
        if factory is None:
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise DocumentExtractionError(
                    "PDF extraction is optional. Install it with: python -m pip install pypdf"
                ) from exc
            factory = PdfReader
        try:
            reader = factory(BytesIO(body))
            pages = getattr(reader, "pages")
            page_count = len(pages)
            text_parts: list[str] = []
            for page in pages[: self.max_pages]:
                extract_text = getattr(page, "extract_text", None)
                if callable(extract_text):
                    text_parts.append(str(extract_text() or ""))
        except Exception as exc:
            raise DocumentExtractionError(f"PDF extraction failed: {type(exc).__name__}") from exc
        result = self._bounded("application/pdf", "\n".join(text_parts))
        return ExtractedDocument(
            media_type=result.media_type,
            text=result.text,
            title=result.title,
            page_count=page_count,
            truncated=result.truncated or page_count > self.max_pages,
        )

    def _bounded(self, media_type: str, text: str, *, title: str = "") -> ExtractedDocument:
        compact = " ".join(text.split())
        return ExtractedDocument(
            media_type=media_type,
            text=compact[: self.max_chars],
            title=" ".join(title.split())[:300],
            truncated=len(compact) > self.max_chars,
        )


class _HTMLTextParser(HTMLParser):
    _SUPPRESSED = frozenset({"script", "style", "noscript", "svg", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._suppressed = 0
        self._in_title = False
        self._title: list[str] = []
        self._parts: list[str] = []

    @property
    def title(self) -> str:
        return " ".join(self._title)

    @property
    def text(self) -> str:
        return " ".join(self._parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in self._SUPPRESSED:
            self._suppressed += 1
        elif tag == "title" and not self._suppressed:
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SUPPRESSED and self._suppressed:
            self._suppressed -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._suppressed:
            return
        (self._title if self._in_title else self._parts).append(data)


__all__ = ["DocumentExtractionError", "DocumentExtractor", "ExtractedDocument"]

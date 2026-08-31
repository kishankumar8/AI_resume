"""
resume_parser.py
----------------
Extracts text from a resume PDF using pypdf.

This module does not guess or invent anything about the candidate.
It only returns the text that is already in the PDF.
"""

from __future__ import annotations

import re
from typing import Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class ResumeParseError(Exception):
    """Raised when a PDF cannot be read or has no usable text."""


def extract_text_from_pdf(uploaded_file) -> str:
    """
    Read a Streamlit UploadedFile (or any file-like object) and return
    the combined text from every page.

    Parameters
    ----------
    uploaded_file : file-like
        A PDF opened in binary mode. Streamlit's st.file_uploader object works.

    Returns
    -------
    str
        Plain text extracted from the PDF.

    Raises
    ------
    ResumeParseError
        If the file is not a readable PDF, is encrypted, or has no text.
    """
    if uploaded_file is None:
        raise ResumeParseError("No file was uploaded.")

    try:
        reader = PdfReader(uploaded_file)
    except PdfReadError as exc:
        raise ResumeParseError(
            "Could not read this file as a PDF. Please upload a valid PDF resume."
        ) from exc
    except Exception as exc:
        raise ResumeParseError(f"Unexpected error while opening the PDF: {exc}") from exc

    if getattr(reader, "is_encrypted", False):
        # Try an empty password (some PDFs are "encrypted" but still readable).
        try:
            unlocked = reader.decrypt("")
        except Exception:
            unlocked = False
        if not unlocked:
            raise ResumeParseError(
                "This PDF is password-protected. Please upload an unlocked copy."
            )

    pages_text: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            raise ResumeParseError(
                f"Could not extract text from page {index}: {exc}"
            ) from exc
        pages_text.append(page_text)

    raw_text = "\n".join(pages_text)
    cleaned = _clean_extracted_text(raw_text)

    if len(cleaned) < 80:
        raise ResumeParseError(
            "Very little text was found in this PDF. It may be a scanned image. "
            "ATS systems (and this app) need a text-based PDF, not a photo of a resume."
        )

    return cleaned


def _clean_extracted_text(text: str) -> str:
    """Normalize whitespace without changing the candidate's actual words."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def looks_like_email(text: str) -> bool:
    """Return True if the text contains something that looks like an email address."""
    return re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text) is not None


def looks_like_phone(text: str) -> bool:
    """Return True if the text contains a phone-number-like sequence of digits."""
    digits = re.sub(r"\D", "", text)
    # A typical phone number has 10–15 digits somewhere in the document.
    return bool(re.search(r"\d{10,15}", digits))


def detect_section_headings(text: str) -> list[str]:
    """
    Find common resume section titles that appear in the extracted text.
    This is a simple heading check, not a full resume parser.
    """
    headings = [
        "summary",
        "objective",
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "education",
        "skills",
        "technical skills",
        "projects",
        "certifications",
        "achievements",
        "awards",
        "publications",
        "languages",
    ]
    lower = text.lower()
    found: list[str] = []
    for heading in headings:
        # Match the heading as its own line or followed by a colon.
        pattern = rf"(?m)^\s*{re.escape(heading)}\s*:?\s*$"
        if re.search(pattern, lower) or f"\n{heading}\n" in lower:
            found.append(heading.title())
    # De-duplicate while keeping order.
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique

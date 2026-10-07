from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock
from typing import Any, Callable

from paddle_ocr import PaddleOcrBackend


UNREADABLE = "[Unreadable Text]"
PADDLE_PAGE_BATCH_SIZE = 8
_PADDLE_OCR_BACKEND: PaddleOcrBackend | None = None
_PADDLE_OCR_LOCK = Lock()


class PdfExtractionError(RuntimeError):
    """Raised when a PDF cannot yield usable local text."""


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str
    method: str
    equations: tuple[str, ...] = ()


def _clean_text(value: object) -> str:
    text = str(value or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _meaningful_count(text: str) -> int:
    return sum(character.isalnum() for character in text)


def _line_looks_like_equation(line: str) -> bool:
    text = _clean_text(line)
    if not text or len(text) > 220:
        return False

    relation = re.search(r"(?:=|<=|>=|!=|≈|≃|≅|≤|≥|≠|∝)", text)
    derivative = re.search(
        r"(?:\b[dD]\s*[A-Za-z]\w*\s*/\s*[dD]\s*[A-Za-z]\w*|[∂∇]\s*\w+)",
        text,
    )
    math_symbol = re.search(r"[+*/^√∫∑ΣΠ±×÷]", text)
    number = re.search(r"\d", text)
    variable = re.search(r"\b[A-Za-z](?:_[A-Za-z0-9]+)?\b", text)

    if relation and (number or variable):
        return True
    if derivative and (relation or math_symbol):
        return True
    return bool(math_symbol and number and variable and len(text.split()) <= 18)


def _equation_key(text: str) -> str:
    return re.sub(r"[^a-z0-9=+*/^√∫∑≤≥≠≈.-]+", "", text.casefold())


def _normalize_equations(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = values.splitlines()
    else:
        try:
            candidates = list(values or [])
        except TypeError:
            candidates = []

    equations: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        for raw_line in _clean_text(candidate).splitlines():
            line = _clean_text(raw_line)
            if not _line_looks_like_equation(line) or _line_looks_gibberish(line):
                continue
            key = _equation_key(line)
            if not key or key in seen:
                continue
            seen.add(key)
            equations.append(line)
    return tuple(equations)


def _without_extracted_equations(text: str, equations: tuple[str, ...]) -> str:
    if not equations:
        return text
    extracted_keys = {
        key for equation in equations if (key := _equation_key(equation))
    }
    if not extracted_keys:
        return text
    return _clean_text(
        "\n".join(
            line
            for line in text.splitlines()
            if not (
                _line_looks_like_equation(line)
                and any(
                    extracted_key in _equation_key(line)
                    for extracted_key in extracted_keys
                )
            )
        )
    )


def _line_looks_gibberish(line: str) -> bool:
    """Detect the repeated all-caps fragments produced by broken slide fonts."""
    if re.search(r"\d", line) and re.search(r"[=+\-/*]", line):
        return False

    words = re.findall(r"[A-Za-z]+", line)
    letters = "".join(words)
    if len(words) < 2 or not letters or letters != letters.upper():
        return False

    suspicious = 0
    long_suspicious = False
    for word in words:
        vowels = sum(character in "AEIOU" for character in word)
        repeated_pair = bool(re.search(r"([A-Z])\1", word))
        consonant_only = len(word) >= 2 and vowels == 0
        if consonant_only or repeated_pair:
            suspicious += 1
        if len(word) >= 9 and (
            repeated_pair
            or vowels / len(word) < 0.2
            or len(set(word)) / len(word) < 0.65
        ):
            long_suspicious = True

    return long_suspicious or suspicious >= 2 or (len(words) >= 4 and suspicious >= 1)


def _filter_gibberish_lines(text: str) -> str:
    kept = [line for line in text.splitlines() if not _line_looks_gibberish(line)]
    return _clean_text("\n".join(kept))


def _text_looks_corrupt(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    corrupt_lines = sum(_line_looks_gibberish(line) for line in lines)
    return corrupt_lines >= 2 and corrupt_lines / len(lines) >= 0.4


def _open_document(path: Path) -> Any:
    try:
        import pymupdf
    except ImportError as exc:
        raise PdfExtractionError(
            "PyMuPDF is not installed; run python -m pip install -r requirements.txt"
        ) from exc
    try:
        return pymupdf.open(str(path))
    except Exception as exc:
        raise PdfExtractionError(f"could not open PDF {path}: {exc}") from exc


def _render_page_for_ocr(page: Any, dpi: int) -> bytes:
    try:
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return bytes(pixmap.tobytes("png"))
    except Exception as exc:
        raise PdfExtractionError(f"could not render PDF page for OCR: {exc}") from exc


def _paddle_model_root() -> Path:
    import sys

    base_dir = (
        Path(sys.executable).resolve().parent
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    return base_dir / "models" / "paddleocr"


def _get_paddle_ocr_backend() -> PaddleOcrBackend:
    global _PADDLE_OCR_BACKEND

    with _PADDLE_OCR_LOCK:
        if _PADDLE_OCR_BACKEND is None:
            _PADDLE_OCR_BACKEND = PaddleOcrBackend(_paddle_model_root())
        return _PADDLE_OCR_BACKEND


def _ocr_images(image_bytes: list[bytes]) -> list[str]:
    try:
        return _get_paddle_ocr_backend().recognize_many(image_bytes)
    except PdfExtractionError:
        raise
    except Exception as exc:
        raise PdfExtractionError(f"local PaddleOCR failed: {exc}") from exc


def _ocr_image(image_bytes: bytes) -> str:
    return _ocr_images([image_bytes])[0]


def _ocr_page(page: Any, dpi: int) -> str:
    return _ocr_image(_render_page_for_ocr(page, dpi))


def _ocr_equation_snapshots(page: Any, dpi: int) -> tuple[str, ...]:
    return _normalize_equations(_ocr_page(page, dpi))


def _build_extracted_page(
    page_number: int,
    direct_text: str,
    ocr_text: str,
    equations: tuple[str, ...],
    min_chars: int,
) -> ExtractedPage:
    direct_body = _without_extracted_equations(direct_text, equations)
    method_suffix = "+equation-ocr" if equations else ""
    if _meaningful_count(direct_text) >= min_chars and not _text_looks_corrupt(
        direct_text
    ):
        return ExtractedPage(
            page_number,
            direct_body,
            f"text{method_suffix}",
            equations,
        )

    recognized_body = _filter_gibberish_lines(_clean_text(ocr_text))
    recognized_body = _without_extracted_equations(recognized_body, equations)
    if _meaningful_count(recognized_body) > 0:
        return ExtractedPage(
            page_number,
            recognized_body,
            f"ocr{method_suffix}",
            equations,
        )
    if equations:
        return ExtractedPage(page_number, "", f"ocr{method_suffix}", equations)

    filtered_direct = _filter_gibberish_lines(direct_body)
    if _meaningful_count(filtered_direct) > 0:
        return ExtractedPage(page_number, filtered_direct, "text")
    return ExtractedPage(page_number, UNREADABLE, "ocr")


def _append_paddle_batch(
    pages: list[ExtractedPage],
    paddle_pages: list[tuple[int, str, bytes]],
    min_chars: int,
) -> None:
    recognized_pages = _ocr_images([image for _, _, image in paddle_pages])
    for (page_number, direct_text, _), recognized_text in zip(
        paddle_pages, recognized_pages, strict=True
    ):
        equations = _normalize_equations(recognized_text)
        pages.append(
            _build_extracted_page(
                page_number,
                direct_text,
                recognized_text,
                equations,
                min_chars,
            )
        )
    paddle_pages.clear()


def extract_pdf_pages(
    path: Path,
    *,
    min_chars: int = 40,
    dpi: int = 200,
    document_factory: Callable[[Path], Any] | None = None,
    ocr: Callable[[Any, int], str] | None = None,
    equation_ocr: Callable[[Any, int], object] | None = None,
) -> tuple[ExtractedPage, ...]:
    source = Path(path)
    if not source.is_file():
        raise PdfExtractionError(f"PDF not found: {source}")
    if source.suffix.casefold() != ".pdf":
        raise PdfExtractionError(f"input must be a PDF file: {source}")
    if min_chars < 1:
        raise ValueError("min_chars must be at least 1")
    if dpi < 72:
        raise ValueError("dpi must be at least 72")
    opener = document_factory or _open_document
    ocr_reader = ocr or _ocr_page
    document = opener(source)
    pages: list[ExtractedPage] = []
    paddle_pages: list[tuple[int, str, bytes]] = []
    use_batched_paddle = ocr is None and equation_ocr is None
    try:
        page_count = int(getattr(document, "page_count", 0))
        for index in range(page_count):
            page = document.load_page(index)
            direct_text = _clean_text(page.get_text("text", sort=True))
            if use_batched_paddle:
                paddle_pages.append(
                    (index + 1, direct_text, _render_page_for_ocr(page, dpi))
                )
                if len(paddle_pages) >= PADDLE_PAGE_BATCH_SIZE:
                    _append_paddle_batch(pages, paddle_pages, min_chars)
                continue

            if equation_ocr is not None:
                equations = _normalize_equations(equation_ocr(page, dpi))
            else:
                equations = ()
            needs_ocr = _meaningful_count(direct_text) < min_chars or _text_looks_corrupt(
                direct_text
            )
            recognized_text = ocr_reader(page, dpi) if needs_ocr else ""
            pages.append(
                _build_extracted_page(
                    index + 1,
                    direct_text,
                    recognized_text,
                    equations,
                    min_chars,
                )
            )
        if paddle_pages:
            _append_paddle_batch(pages, paddle_pages, min_chars)
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()

    completed_pages = tuple(pages)
    if not completed_pages:
        raise PdfExtractionError(f"PDF contains no pages: {source}")
    if not any(page.text != UNREADABLE or page.equations for page in completed_pages):
        raise PdfExtractionError(f"PDF contains no readable text after local OCR: {source}")
    return completed_pages

#!/usr/bin/env python3
"""
Convert PowerPoint-slide PDFs into one UTF-8 .txt file per PDF using local
PDF text extraction and PaddleOCR only.

Default use:
    python slides_pdf_to_txt.py

That processes every PDF in input/ and writes .txt files to output/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pdf_ingestion import ExtractedPage, extract_pdf_pages

UNREADABLE = "[Unreadable Text]"
NOT_SPECIFIED = "Not Specified"
NOT_SPECIFIED_DESCRIPTION = {
    "label": "Image/Diagram Description",
    "text": NOT_SPECIFIED,
}


def now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = runtime_base_dir()


def clean_field(value: Any) -> str:
    if value is None:
        return NOT_SPECIFIED
    text = str(value).strip()
    return text if text else NOT_SPECIFIED


def is_markdown_table_separator(line: str) -> bool:
    stripped = line.strip()
    return bool(re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped))


def markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if "|" not in stripped or is_markdown_table_separator(stripped):
        return []
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells = [cell.strip() for cell in stripped.split("|")]
    return [cell for cell in cells if cell]


def markdown_tables_to_plain_text(text: str) -> str:
    converted_lines: list[str] = []

    for line in text.splitlines():
        if is_markdown_table_separator(line):
            continue

        cells = markdown_table_cells(line)
        if len(cells) == 2:
            converted_lines.append(f"{cells[0]}: {cells[1]}")
        elif len(cells) > 2:
            converted_lines.append("; ".join(cells))
        else:
            converted_lines.append(line)

    converted = "\n".join(converted_lines).strip()
    return converted if converted else NOT_SPECIFIED


def _page_lines(page: ExtractedPage) -> list[str]:
    return [line.strip() for line in str(page.text or "").splitlines() if line.strip()]


def extract_module_metadata(first_page: ExtractedPage) -> tuple[str, str]:
    """Derive metadata directly from OCR text without generating new content."""
    lines = _page_lines(first_page)
    if not lines:
        return NOT_SPECIFIED, NOT_SPECIFIED

    module_number = NOT_SPECIFIED
    module_line_index: int | None = None
    pattern = re.compile(
        r"\bmodule\s*(?:#|no\.?|number)?\s*[:\-]?\s*([A-Za-z0-9]+(?:[.\-][A-Za-z0-9]+)*)",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        match = pattern.search(line)
        if match:
            module_number = match.group(1)
            module_line_index = index
            break

    if module_line_index is not None and module_line_index + 1 < len(lines):
        module_title = lines[module_line_index + 1]
    elif module_line_index != 0:
        module_title = lines[0]
    else:
        module_title = NOT_SPECIFIED
    return module_number, clean_field(module_title)


def format_slide(slide_number: int, page: ExtractedPage) -> dict[str, Any]:
    """Map OCR output to the existing text schema without an LLM."""
    del slide_number
    lines = _page_lines(page)
    if not lines or lines == [UNREADABLE]:
        title = NOT_SPECIFIED
        content = UNREADABLE
    else:
        title = lines[0]
        content = "\n".join(lines[1:]).strip() or NOT_SPECIFIED
    return {
        "title": clean_field(title),
        "content": markdown_tables_to_plain_text(content),
        "equations": tuple(page.equations or ()),
        "descriptions": [NOT_SPECIFIED_DESCRIPTION.copy()],
        "brief_explanation": NOT_SPECIFIED,
    }


def render_document(module_number: str, module_title: str, slides: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"Module #: {module_number}",
        "",
        f"Module Title: {module_title}",
        "",
        "---",
        "",
    ]

    for index, slide in enumerate(slides, start=1):
        lines.extend(
            [
                f"Slide {index}:",
                "{",
                "Title:",
                slide["title"],
                "",
                "Content:",
                slide["content"],
                "",
            ]
        )

        equations = tuple(slide.get("equations", ()) or ())
        if equations:
            lines.append("Equations (local OCR):")
            lines.extend(
                f"Equation {equation_index}: {equation}"
                for equation_index, equation in enumerate(equations, start=1)
            )
            lines.append("")

        for description in slide["descriptions"]:
            lines.extend(
                [
                    f"{description['label']}:",
                    description["text"],
                    "",
                ]
            )

        if lines and lines[-1] == "":
            lines.pop()

        lines.extend(
            [
                "}",
                "",
                "Brief Explanation:",
                slide["brief_explanation"],
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def safe_output_name(pdf_path: Path, fallback_index: int) -> str:
    stem = pdf_path.stem.strip() or f"Document_{fallback_index}"
    stem = re.sub(r'[<>:"/\\|?*]+', "_", stem)
    stem = stem.rstrip(" .")
    return stem or f"Document_{fallback_index}"


def unique_output_path(output_dir: Path, pdf_path: Path, index: int, overwrite: bool) -> Path:
    base_name = safe_output_name(pdf_path, index)
    candidate = output_dir / f"{base_name}.txt"
    if overwrite or not candidate.exists():
        return candidate

    counter = 2
    while True:
        candidate = output_dir / f"{base_name}_{counter}.txt"
        if not candidate.exists():
            return candidate
        counter += 1


def collect_pdfs(args: argparse.Namespace) -> list[Path]:
    if args.pdfs:
        pdfs = [Path(path).expanduser().resolve() for path in args.pdfs]
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        pdfs = sorted(input_dir.glob("*.pdf"))

    missing = [path for path in pdfs if not path.exists()]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"These PDF files do not exist:\n{missing_text}")

    non_pdfs = [path for path in pdfs if path.suffix.lower() != ".pdf"]
    if non_pdfs:
        non_pdf_text = "\n".join(str(path) for path in non_pdfs)
        raise ValueError(f"Only PDF files are supported:\n{non_pdf_text}")

    return pdfs


def collect_pdf_batches(input_dir: Path, output_dir: Path) -> list[tuple[Path, list[Path]]]:
    batches_by_output: dict[Path, list[Path]] = {}
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        relative_parent = pdf_path.parent.relative_to(input_dir)
        batch_output_dir = output_dir if str(relative_parent) == "." else output_dir / relative_parent
        batches_by_output.setdefault(batch_output_dir, []).append(pdf_path)

    return [(batch_output_dir, pdfs) for batch_output_dir, pdfs in batches_by_output.items()]


class ProgressTracker:
    def __init__(self, base_dir: Path) -> None:
        self.path = base_dir / "progress.json"
        self.log_path = base_dir / "progress.log"
        self.state: dict[str, Any] = {
            "started_at": now_text(),
            "updated_at": now_text(),
            "current_pdf": None,
            "processed": [],
            "failed": [],
            "deleted_inputs": [],
            "last_cleanup": None,
        }
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(loaded, dict):
            self.state.update(loaded)

    def save(self) -> None:
        self.state["updated_at"] = now_text()
        self.path.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2),
            encoding="utf-8",
            newline="\n",
        )

    def log(self, message: str) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(f"[{now_text()}] {message}\n")

    def record_start(self, pdf_path: Path, output_path: Path) -> None:
        self.state["current_pdf"] = str(pdf_path)
        self.state["current_output"] = str(output_path)
        self.log(f"Processing {pdf_path}")
        self.save()

    def record_success(self, pdf_path: Path, output_path: Path) -> None:
        self.state["current_pdf"] = None
        self.state["current_output"] = None
        self.state.setdefault("processed", []).append(
            {"pdf": str(pdf_path), "output": str(output_path), "finished_at": now_text()}
        )
        self.log(f"Finished {pdf_path} -> {output_path}")
        self.save()

    def record_failure(self, pdf_path: Path, error: Exception) -> None:
        self.state["current_pdf"] = str(pdf_path)
        self.state.setdefault("failed", []).append(
            {"pdf": str(pdf_path), "error": str(error), "failed_at": now_text()}
        )
        self.log(f"Failed {pdf_path}: {error}")
        self.save()

    def record_deleted_input(self, pdf_path: Path, reason: str) -> None:
        self.state.setdefault("deleted_inputs", []).append(
            {"pdf": str(pdf_path), "reason": reason, "deleted_at": now_text()}
        )
        self.state["last_cleanup"] = now_text()
        self.log(f"Deleted input {pdf_path}: {reason}")
        self.save()


def matching_output_exists(pdf_path: Path, input_dir: Path, output_dir: Path) -> bool:
    relative_parent = pdf_path.parent.relative_to(input_dir)
    batch_output_dir = output_dir if str(relative_parent) == "." else output_dir / relative_parent
    base_name = safe_output_name(pdf_path, 1)
    exact_output = batch_output_dir / f"{base_name}.txt"
    if exact_output.exists():
        return True
    if not batch_output_dir.exists():
        return False

    numbered_prefix = f"{base_name}_"
    for output_path in batch_output_dir.iterdir():
        if output_path.suffix.lower() != ".txt":
            continue
        if output_path.stem.startswith(numbered_prefix):
            return True
    return False


def delete_finished_input(pdf_path: Path, output_path: Path, progress: ProgressTracker, reason: str) -> None:
    if not output_path.exists() or not pdf_path.exists():
        return
    try:
        pdf_path.unlink()
    except OSError as exc:
        progress.log(f"Could not delete input {pdf_path}: {exc}")
        print(f"Warning: could not delete finished input PDF {pdf_path}: {exc}", file=sys.stderr)
        return
    progress.record_deleted_input(pdf_path, reason)
    print(f"Deleted finished input PDF: {pdf_path}", file=sys.stderr)


def cleanup_finished_inputs(input_dir: Path, output_dir: Path, progress: ProgressTracker) -> None:
    if not input_dir.exists():
        return
    for pdf_path in sorted(input_dir.rglob("*.pdf")):
        if matching_output_exists(pdf_path, input_dir, output_dir):
            try:
                pdf_path.unlink()
            except OSError as exc:
                progress.log(f"Could not delete already-finished input {pdf_path}: {exc}")
                print(f"Warning: could not delete already-finished input PDF {pdf_path}: {exc}", file=sys.stderr)
                continue
            progress.record_deleted_input(pdf_path, "matching output already exists")
            print(f"Deleted already-finished input PDF: {pdf_path}", file=sys.stderr)


def process_pdf(
    pdf_path: Path,
    output_path: Path,
    args: argparse.Namespace,
    progress: ProgressTracker | None = None,
) -> None:
    print(f"Processing {pdf_path.name}...", file=sys.stderr)
    if progress is not None:
        progress.record_start(pdf_path, output_path)

    try:
        pages = list(
            extract_pdf_pages(
                pdf_path,
                min_chars=args.ocr_min_chars,
                dpi=args.ocr_dpi,
            )
        )

        module_number, module_title = extract_module_metadata(pages[0])

        slides: list[dict[str, Any]] = []
        for slide_number, page in enumerate(pages, start=1):
            print(f"  Writing slide {slide_number}/{len(pages)}...", file=sys.stderr)
            slides.append(format_slide(slide_number, page))

        output_path.write_text(
            render_document(module_number, module_title, slides),
            encoding="utf-8",
            newline="\n",
        )
        if progress is not None:
            progress.record_success(pdf_path, output_path)
        if args.delete_inputs:
            if progress is None:
                progress = ProgressTracker(BASE_DIR)
            delete_finished_input(pdf_path, output_path, progress, "processed successfully")
        print(str(output_path))
    except Exception as exc:
        if progress is not None:
            progress.record_failure(pdf_path, exc)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one UTF-8 .txt file per PowerPoint-slide PDF using "
            "local extraction and PaddleOCR."
        )
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files to process. If omitted, every .pdf in --input-dir is processed.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(BASE_DIR / "input"),
        help="Folder to scan for PDFs when no PDF paths are provided. Default: input beside the app.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(BASE_DIR / "output"),
        help="Folder where .txt files are written. Default: output beside the app.",
    )
    parser.add_argument(
        "--ocr-min-chars",
        type=int,
        default=40,
        help="Use local OCR when a PDF page has fewer readable characters. Default: 40.",
    )
    parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=200,
        help="Rendering resolution for local PaddleOCR. Default: 200.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files instead of creating numbered filenames.",
    )
    parser.add_argument(
        "--keep-inputs",
        dest="delete_inputs",
        action="store_false",
        help="Keep input PDFs after successful extraction. By default, finished PDFs are deleted.",
    )
    parser.set_defaults(delete_inputs=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.ocr_min_chars < 1:
        raise ValueError("--ocr-min-chars must be at least 1.")
    if args.ocr_dpi < 72:
        raise ValueError("--ocr-dpi must be at least 72.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressTracker(BASE_DIR)

    if args.pdfs:
        batches = [(output_dir, collect_pdfs(args))]
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        input_dir.mkdir(parents=True, exist_ok=True)
        if args.delete_inputs:
            cleanup_finished_inputs(input_dir, output_dir, progress)
        batches = collect_pdf_batches(input_dir, output_dir)

    if not batches:
        print("No PDF files found to process.", file=sys.stderr)
        progress.log("No PDF files found to process.")
        progress.save()
        return 0

    for batch_output_dir, pdfs in batches:
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        for index, pdf_path in enumerate(pdfs, start=1):
            output_path = unique_output_path(
                batch_output_dir, pdf_path, index, args.overwrite
            )
            process_pdf(pdf_path, output_path, args, progress)

    return 0


def wait_before_close() -> None:
    if not getattr(sys, "frozen", False):
        return
    try:
        input("\nPress Enter to close this window.")
    except EOFError:
        pass


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as exc:  # noqa: BLE001 - show clean CLI errors.
        print(f"Error: {exc}", file=sys.stderr)
    finally:
        wait_before_close()
    raise SystemExit(exit_code)

#!/usr/bin/env python3
"""
Convert PowerPoint-slide PDFs into one strict UTF-8 .txt file per PDF using
local PDF text extraction, batched PaddleOCR page recognition, and local Qwen
formatting.

Default use:
    python slides_pdf_to_txt.py

That processes every PDF in input/ and writes .txt files to output/.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from local_qwen import LocalQwenBackend, ensure_model
from pdf_ingestion import extract_pdf_pages

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


MODULE_SYSTEM_PROMPT = """You extract metadata from the title slide of a PowerPoint PDF.

Use only the supplied OCR text and visible slide data. Do not use the filename.
If the module number or module title cannot be confidently identified from the title slide, return "Not Specified".
Return JSON only:
{
  "module_number": "string",
  "module_title": "string"
}
"""


SLIDE_SYSTEM_PROMPT = """You convert one OCR page from a PowerPoint-slide PDF into strict JSON for a plain-text extraction file.

Rules you must follow:
- Treat the supplied OCR page as one slide.
- Use only visible content supplied in the OCR JSON.
- Never invent, infer, summarize, or add outside information.
- Preserve the original order and wording whenever possible.
- Correct obvious OCR mistakes only when the intended word is clear.
- Silently omit repeated gibberish or random character sequences caused by broken fonts.
- If any text cannot be read, write "[Unreadable Text]".
- Extract slide title, section headings, bullets, numbered lists, tables, labels, figure captions, chart labels, axis labels, legends, definitions, key statistics, and readable footnotes.
- The equations field contains text produced locally from separate equation snapshots.
- Do not copy equation OCR into content. The application appends each equation separately and verbatim.
- Do not extract speaker notes, hidden slides, watermarks, repeated institutional logos, decorative elements, or page numbers unless they are part of slide content.
- Do not use Markdown tables. Convert tables into plain text rows or label-value lines.
- This is a text-only OCR pipeline. Never infer what an image, graph, or diagram contains.
- Always include one description item with label "Image/Diagram Description" and text "Not Specified".
- Prefer a brief_explanation with 2 or 3 complete sentences in a clear university teaching style.
- The brief_explanation must be strictly based on the slide content.

Return JSON only:
{
  "title": "string",
  "content": "string",
  "descriptions": [
    {
      "label": "Image/Diagram Description",
      "text": "string"
    }
  ],
  "brief_explanation": "string"
}
"""


def to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(value) for value in obj]
    if hasattr(obj, "model_dump"):
        return to_jsonable(obj.model_dump())
    if hasattr(obj, "dict"):
        return to_jsonable(obj.dict())
    if hasattr(obj, "__dict__"):
        return {
            key: to_jsonable(value)
            for key, value in vars(obj).items()
            if not key.startswith("_")
        }
    return str(obj)


def strip_large_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for key, value in obj.items():
            lowered = key.lower()
            if "base64" in lowered:
                cleaned[key] = "[image base64 omitted]"
            else:
                cleaned[key] = strip_large_values(value)
        return cleaned
    if isinstance(obj, list):
        return [strip_large_values(value) for value in obj]
    return obj


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object from local Qwen.")
    return data


def retry(
    label: str,
    attempts: int,
    delay_seconds: float,
    fn: Any,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - model/runtime errors vary.
            last_error = exc
            if attempt == attempts:
                break

            wait = delay_seconds * attempt

            print(
                f"{label} failed on attempt {attempt}; retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def chat_json(
    backend: Any,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    attempts: int,
) -> dict[str, Any]:
    def call() -> dict[str, Any]:
        response = backend.complete(
            system_prompt,
            user_prompt,
            max_tokens=max_tokens,
        )
        return extract_json_object(response)

    return retry("Local Qwen formatting", attempts, 2.0, call)


def page_for_prompt(page: Any) -> dict[str, Any]:
    json_page = to_jsonable(page)
    if isinstance(json_page, dict):
        json_page.pop("image_png", None)
    return strip_large_values(json_page)


def clean_field(value: Any) -> str:
    if value is None:
        return NOT_SPECIFIED
    text = str(value).strip()
    return text if text else NOT_SPECIFIED


def sentence_count(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    sentences = re.findall(r"[^.!?]+[.!?](?:\s+|$)", text)
    return len(sentences) if sentences else 1


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


def normalize_slide_data(
    data: dict[str, Any],
    equations: tuple[str, ...] = (),
) -> dict[str, Any]:
    title = clean_field(data.get("title"))
    content = markdown_tables_to_plain_text(clean_field(data.get("content")))
    brief_explanation = clean_field(data.get("brief_explanation"))

    return {
        "title": title,
        "content": content,
        "equations": equations,
        "descriptions": [NOT_SPECIFIED_DESCRIPTION.copy()],
        "brief_explanation": brief_explanation,
    }


def extract_module_metadata(
    backend: Any,
    first_page: Any,
    attempts: int,
    max_tokens: int,
) -> tuple[str, str]:
    prompt = (
        "Title slide OCR JSON:\n"
        f"{json.dumps(page_for_prompt(first_page), ensure_ascii=False, indent=2)}"
    )
    data = chat_json(
        backend=backend,
        system_prompt=MODULE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
        attempts=attempts,
    )
    return clean_field(data.get("module_number")), clean_field(data.get("module_title"))


def format_slide(
    backend: Any,
    slide_number: int,
    page: Any,
    attempts: int,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = (
        f"Slide number: {slide_number}\n"
        "OCR page JSON:\n"
        f"{json.dumps(page_for_prompt(page), ensure_ascii=False, indent=2)}"
    )

    last_error: Exception | None = None
    for _ in range(attempts):
        data = chat_json(
            backend=backend,
            system_prompt=SLIDE_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
            attempts=attempts,
        )
        try:
            return normalize_slide_data(
                data,
                tuple(getattr(page, "equations", ()) or ()),
            )
        except ValueError as exc:
            last_error = exc
            prompt += (
                "\n\nYour previous JSON failed validation: "
                f"{exc}. Return corrected JSON only, using the same OCR data."
            )

    raise RuntimeError(f"Slide {slide_number} could not be formatted strictly: {last_error}") from last_error


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
    backend: Any,
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

        module_number, module_title = extract_module_metadata(
            backend=backend,
            first_page=pages[0],
            attempts=args.attempts,
            max_tokens=args.metadata_max_tokens,
        )

        slides: list[dict[str, Any]] = []
        for slide_number, page in enumerate(pages, start=1):
            if args.request_delay > 0:
                time.sleep(args.request_delay)
            print(f"  Formatting slide {slide_number}/{len(pages)}...", file=sys.stderr)
            slides.append(
                format_slide(
                    backend=backend,
                    slide_number=slide_number,
                    page=page,
                    attempts=args.attempts,
                    max_tokens=args.slide_max_tokens,
                )
            )

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
            "Generate one strict UTF-8 .txt file per PowerPoint-slide PDF using "
            "local extraction, PaddleOCR fallback, and local Qwen."
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
        "--model-dir",
        default=str(BASE_DIR / "models"),
        help="Folder containing the local Qwen GGUF model. Default: models beside the app.",
    )
    parser.add_argument(
        "--n-ctx",
        type=int,
        default=8192,
        help="Qwen context-window size. Default: 8192.",
    )
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Qwen layers to offload to the GPU; -1 requests all layers. Default: -1.",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=8,
        help="CPU generation threads, tuned for the Ryzen 7 5800X. Default: 8.",
    )
    parser.add_argument(
        "--n-threads-batch",
        type=int,
        default=16,
        help="CPU prompt-processing threads. Default: 16.",
    )
    parser.add_argument(
        "--n-batch",
        type=int,
        default=512,
        help="Logical prompt batch size for the RTX 3060. Default: 512.",
    )
    parser.add_argument(
        "--n-ubatch",
        type=int,
        default=512,
        help="Physical prompt batch size for the RTX 3060. Default: 512.",
    )
    parser.add_argument(
        "--no-flash-attn",
        dest="flash_attn",
        action="store_false",
        help="Disable CUDA flash attention for troubleshooting.",
    )
    parser.add_argument(
        "--allow-cpu-fallback",
        dest="require_gpu",
        action="store_false",
        help="Allow a CPU-only llama.cpp build instead of requiring CUDA.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Qwen generation seed. Default: 42.",
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
        "--attempts",
        type=int,
        default=6,
        help="Retry attempts for local Qwen calls and strict JSON formatting. Default: 6.",
    )
    parser.add_argument(
        "--slide-max-tokens",
        type=int,
        default=6000,
        help="Maximum output tokens for each formatted slide. Default: 6000.",
    )
    parser.add_argument(
        "--metadata-max-tokens",
        type=int,
        default=800,
        help="Maximum output tokens for module metadata. Default: 800.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.0,
        help="Seconds to pause before each slide-formatting request. Default: 0.",
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
    parser.set_defaults(delete_inputs=True, flash_attn=True, require_gpu=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1.")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be 0 or greater.")
    if args.slide_max_tokens < 1000:
        raise ValueError("--slide-max-tokens must be at least 1000.")
    if args.metadata_max_tokens < 200:
        raise ValueError("--metadata-max-tokens must be at least 200.")
    if args.n_ctx < 2048:
        raise ValueError("--n-ctx must be at least 2048.")
    if args.n_gpu_layers < -1:
        raise ValueError("--n-gpu-layers must be -1 or greater.")
    if args.n_threads < 1:
        raise ValueError("--n-threads must be at least 1.")
    if args.n_threads_batch < 1:
        raise ValueError("--n-threads-batch must be at least 1.")
    if args.n_batch < 1:
        raise ValueError("--n-batch must be at least 1.")
    if args.n_ubatch < 1 or args.n_ubatch > args.n_batch:
        raise ValueError("--n-ubatch must be between 1 and --n-batch.")
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

    model_path = ensure_model(Path(args.model_dir).expanduser().resolve())
    backend = LocalQwenBackend(
        model_path,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        n_threads=args.n_threads,
        n_threads_batch=args.n_threads_batch,
        n_batch=args.n_batch,
        n_ubatch=args.n_ubatch,
        flash_attn=args.flash_attn,
        require_gpu=args.require_gpu,
        temperature=0,
        seed=args.seed,
    )
    try:
        for batch_output_dir, pdfs in batches:
            batch_output_dir.mkdir(parents=True, exist_ok=True)
            for index, pdf_path in enumerate(pdfs, start=1):
                output_path = unique_output_path(
                    batch_output_dir, pdf_path, index, args.overwrite
                )
                process_pdf(backend, pdf_path, output_path, args, progress)
    finally:
        backend.close()

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

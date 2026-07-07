#!/usr/bin/env python3
"""
Convert PowerPoint-slide PDFs into one strict UTF-8 .txt file per PDF using
Mistral Document AI OCR plus a formatting pass.

Default use:
    python slides_pdf_to_txt.py

That processes every PDF in input/ and writes .txt files to output/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


OCR_MODEL = "mistral-ocr-latest"
FORMAT_MODEL = "mistral-large-2512"

UNREADABLE = "[Unreadable Text]"
NOT_SPECIFIED = "Not Specified"


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
- If any text cannot be read, write "[Unreadable Text]".
- Extract slide title, section headings, bullets, numbered lists, tables, labels, figure captions, chart labels, axis labels, legends, equations, definitions, key statistics, and readable footnotes.
- Do not extract speaker notes, hidden slides, watermarks, repeated institutional logos, decorative elements, or page numbers unless they are part of slide content.
- Do not use Markdown tables. Convert tables into plain text rows or label-value lines.
- For every non-text element, add a concise factual description strictly based on OCR image/table/block annotations.
- If there is no non-text element, include one description item with label "Image/Diagram Description" and text "Not Specified".
- The brief_explanation must be exactly 2 or 3 complete sentences in a clear university teaching style.
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


def load_dotenv_file(path: Path) -> None:
    """Small .env loader so the script works without python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def import_mistral_client() -> Any:
    try:
        from mistralai import Mistral

        return Mistral
    except ImportError:
        try:
            from mistralai.client import Mistral

            return Mistral
        except ImportError as exc:
            raise RuntimeError(
                "The Mistral SDK is not installed. Run: python -m pip install -r requirements.txt"
            ) from exc


def build_bbox_annotation_format() -> Any:
    try:
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(
            "Pydantic is not installed. Run: python -m pip install -r requirements.txt"
        ) from exc

    class VisualElement(BaseModel):
        element_type: str = Field(
            ...,
            description=(
                "The visible non-text element type, such as chart, graph, table, "
                "flowchart, diagram, map, photo, illustration, or timeline."
            ),
        )
        description: str = Field(
            ...,
            description=(
                "One concise factual sentence describing only what is visible. "
                "Use [Unreadable Text] for unreadable text in the visual."
            ),
        )

    try:
        from mistralai.extra import response_format_from_pydantic_model

        return response_format_from_pydantic_model(VisualElement)
    except Exception:
        schema = VisualElement.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "visual_element_annotation",
                "schema": schema,
                "strict": True,
            },
        }


def get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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


def extract_message_content(response: Any) -> str:
    choices = get_value(response, "choices", [])
    if not choices:
        raise ValueError("Mistral chat response did not include choices.")

    message = get_value(choices[0], "message")
    content = get_value(message, "content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
            else:
                text_parts.append(str(get_value(part, "text", part)))
        return "\n".join(text_parts).strip()

    return str(content).strip()


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
        raise ValueError("Expected a JSON object from Mistral.")
    return data


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "status 429" in text
        or "rate limit" in text
        or "rate_limited" in text
        or '"code":"1300"' in text
    )


def retry_after_from_error(exc: Exception) -> float | None:
    text = str(exc)
    match = re.search(r"retry[-_ ]after[^0-9]*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def retry(
    label: str,
    attempts: int,
    delay_seconds: float,
    fn: Any,
    rate_limit_wait_seconds: float = 60.0,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - API errors vary by SDK version.
            last_error = exc
            if attempt == attempts:
                break

            if is_rate_limit_error(exc):
                retry_after = retry_after_from_error(exc)
                wait = retry_after if retry_after is not None else rate_limit_wait_seconds
                wait = max(wait, delay_seconds * attempt)
                reason = "rate limit"
            else:
                wait = delay_seconds * attempt
                reason = "error"

            print(
                f"{label} hit {reason} on attempt {attempt}; retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}") from last_error


def chat_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    attempts: int,
    rate_limit_wait_seconds: float = 60.0,
) -> dict[str, Any]:
    def call() -> dict[str, Any]:
        response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return extract_json_object(extract_message_content(response))

    return retry("Mistral chat formatting", attempts, 2.0, call, rate_limit_wait_seconds)


def upload_pdf(client: Any, pdf_path: Path, attempts: int) -> str:
    def call() -> str:
        with pdf_path.open("rb") as file_handle:
            try:
                uploaded = client.files.upload(
                    file={"file_name": pdf_path.name, "content": file_handle},
                    purpose="ocr",
                    visibility="workspace",
                )
            except TypeError:
                file_handle.seek(0)
                uploaded = client.files.upload(
                    file={"file_name": pdf_path.name, "content": file_handle},
                    visibility="workspace",
                )

        file_id = get_value(uploaded, "id")
        if not file_id:
            raise ValueError("Mistral upload response did not include a file id.")
        return str(file_id)

    return retry(f"Upload {pdf_path.name}", attempts, 3.0, call)


def get_signed_url(client: Any, file_id: str, attempts: int) -> str:
    def call() -> str:
        signed = client.files.get_signed_url(file_id=file_id, expiry=24)
        url = get_value(signed, "url")
        if not url:
            raise ValueError("Mistral signed URL response did not include a URL.")
        return str(url)

    return retry("Get signed URL", attempts, 2.0, call)


def run_ocr(client: Any, document_url: str, args: argparse.Namespace) -> Any:
    bbox_annotation_format = build_bbox_annotation_format()

    def call() -> Any:
        return client.ocr.process(
            model=args.ocr_model,
            document={"type": "document_url", "document_url": document_url},
            include_image_base64=True,
            include_blocks=True,
            table_format="markdown",
            confidence_scores_granularity="page",
            bbox_annotation_format=bbox_annotation_format,
        )

    return retry("Mistral OCR", args.attempts, 5.0, call)


def delete_remote_file(client: Any, file_id: str) -> None:
    try:
        client.files.delete(file_id=file_id)
    except Exception as exc:  # noqa: BLE001 - cleanup should not fail the output.
        print(f"Warning: could not delete uploaded Mistral file {file_id}: {exc}", file=sys.stderr)


def sort_pages(pages: list[Any]) -> list[Any]:
    indexed_pages: list[tuple[int, int, Any]] = []
    for fallback_index, page in enumerate(pages):
        raw_index = get_value(page, "index", fallback_index)
        try:
            page_index = int(raw_index)
        except (TypeError, ValueError):
            page_index = fallback_index
        indexed_pages.append((page_index, fallback_index, page))
    return [page for _, _, page in sorted(indexed_pages)]


def page_for_prompt(page: Any) -> dict[str, Any]:
    json_page = to_jsonable(page)
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


def contains_markdown_table(text: str) -> bool:
    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", stripped):
            return True
    return False


def normalize_descriptions(value: Any) -> list[dict[str, str]]:
    descriptions: list[dict[str, str]] = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            label = clean_field(item.get("label"))
            text = clean_field(item.get("text"))
            if label == NOT_SPECIFIED:
                label = "Image/Diagram Description"
            if "description" not in label.lower():
                label = f"{label} Description"
            descriptions.append({"label": label, "text": text})

    if not descriptions:
        descriptions.append({"label": "Image/Diagram Description", "text": NOT_SPECIFIED})
    return descriptions


def normalize_slide_data(data: dict[str, Any]) -> dict[str, Any]:
    title = clean_field(data.get("title"))
    content = clean_field(data.get("content"))
    descriptions = normalize_descriptions(data.get("descriptions"))
    brief_explanation = clean_field(data.get("brief_explanation"))

    if contains_markdown_table(content):
        raise ValueError("content contains Markdown table syntax")

    explanation_sentences = sentence_count(brief_explanation)
    if explanation_sentences < 2 or explanation_sentences > 3:
        raise ValueError("brief_explanation must contain exactly 2 or 3 sentences")

    return {
        "title": title,
        "content": content,
        "descriptions": descriptions,
        "brief_explanation": brief_explanation,
    }


def extract_module_metadata(
    client: Any,
    model: str,
    first_page: Any,
    attempts: int,
    rate_limit_wait_seconds: float,
    max_tokens: int,
) -> tuple[str, str]:
    prompt = (
        "Title slide OCR JSON:\n"
        f"{json.dumps(page_for_prompt(first_page), ensure_ascii=False, indent=2)}"
    )
    data = chat_json(
        client=client,
        model=model,
        system_prompt=MODULE_SYSTEM_PROMPT,
        user_prompt=prompt,
        max_tokens=max_tokens,
        attempts=attempts,
        rate_limit_wait_seconds=rate_limit_wait_seconds,
    )
    return clean_field(data.get("module_number")), clean_field(data.get("module_title"))


def format_slide(
    client: Any,
    model: str,
    slide_number: int,
    page: Any,
    attempts: int,
    rate_limit_wait_seconds: float,
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
            client=client,
            model=model,
            system_prompt=SLIDE_SYSTEM_PROMPT,
            user_prompt=prompt,
            max_tokens=max_tokens,
            attempts=attempts,
            rate_limit_wait_seconds=rate_limit_wait_seconds,
        )
        try:
            return normalize_slide_data(data)
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


def process_pdf(client: Any, pdf_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    print(f"Processing {pdf_path.name}...", file=sys.stderr)
    file_id = upload_pdf(client, pdf_path, args.attempts)

    try:
        signed_url = get_signed_url(client, file_id, args.attempts)
        ocr_response = run_ocr(client, signed_url, args)
        pages = sort_pages(list(get_value(ocr_response, "pages", [])))
        if not pages:
            raise RuntimeError("Mistral OCR returned no pages.")

        module_number, module_title = extract_module_metadata(
            client=client,
            model=args.format_model,
            first_page=pages[0],
            attempts=args.attempts,
            rate_limit_wait_seconds=args.rate_limit_wait,
            max_tokens=args.metadata_max_tokens,
        )

        slides: list[dict[str, Any]] = []
        for slide_number, page in enumerate(pages, start=1):
            if args.request_delay > 0:
                time.sleep(args.request_delay)
            print(f"  Formatting slide {slide_number}/{len(pages)}...", file=sys.stderr)
            slides.append(
                format_slide(
                    client=client,
                    model=args.format_model,
                    slide_number=slide_number,
                    page=page,
                    attempts=args.attempts,
                    rate_limit_wait_seconds=args.rate_limit_wait,
                    max_tokens=args.slide_max_tokens,
                )
            )

        output_path.write_text(
            render_document(module_number, module_title, slides),
            encoding="utf-8",
            newline="\n",
        )
        print(str(output_path))
    finally:
        if not args.keep_uploads:
            delete_remote_file(client, file_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one strict UTF-8 .txt file per PowerPoint-slide PDF using Mistral OCR."
    )
    parser.add_argument(
        "pdfs",
        nargs="*",
        help="PDF files to process. If omitted, every .pdf in --input-dir is processed.",
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Folder to scan for PDFs when no PDF paths are provided. Default: input.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Folder where .txt files are written. Default: output.",
    )
    parser.add_argument(
        "--ocr-model",
        default=OCR_MODEL,
        help=f"Mistral OCR model. Default: {OCR_MODEL}.",
    )
    parser.add_argument(
        "--format-model",
        default=FORMAT_MODEL,
        help=f"Mistral chat model used for strict formatting. Default: {FORMAT_MODEL}.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=6,
        help="Retry attempts for API calls and strict JSON formatting. Default: 6.",
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
        "--rate-limit-wait",
        type=float,
        default=60.0,
        help="Seconds to wait before retrying after a Mistral 429 rate limit. Default: 60.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=2.0,
        help="Seconds to pause before each slide-formatting request. Default: 2.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt files instead of creating numbered filenames.",
    )
    parser.add_argument(
        "--keep-uploads",
        action="store_true",
        help="Keep uploaded files in Mistral storage instead of deleting them after processing.",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv_file(Path.cwd() / ".env")
    load_dotenv_file(Path(__file__).resolve().parent / ".env")

    args = parse_args()
    if args.attempts < 1:
        raise ValueError("--attempts must be at least 1.")
    if args.rate_limit_wait < 0:
        raise ValueError("--rate-limit-wait must be 0 or greater.")
    if args.request_delay < 0:
        raise ValueError("--request-delay must be 0 or greater.")
    if args.slide_max_tokens < 1000:
        raise ValueError("--slide-max-tokens must be at least 1000.")
    if args.metadata_max_tokens < 200:
        raise ValueError("--metadata-max-tokens must be at least 200.")

    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Add it to your environment or create a .env file."
        )

    pdfs = collect_pdfs(args)
    if not pdfs:
        raise RuntimeError("No PDF files found to process.")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Mistral = import_mistral_client()
    client = Mistral(api_key=api_key)

    for index, pdf_path in enumerate(pdfs, start=1):
        output_path = unique_output_path(output_dir, pdf_path, index, args.overwrite)
        process_pdf(client, pdf_path, output_path, args)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 - show clean CLI errors.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)






from __future__ import annotations

import html
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from slides_pdf_to_txt import (
    FORMAT_MODEL,
    OCR_MODEL,
    import_mistral_client,
    load_dotenv_file,
    process_pdf,
    unique_output_path,
)

def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = runtime_base_dir()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

app = FastAPI(title="PDF Slide Extraction API")

COURSE_CODE_PATTERN = re.compile(r'[<>:"|?*\x00-\x1F]+')


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str, fallback: str) -> str:
    name = Path(filename or fallback).name.strip() or fallback
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail=f"Only PDF files are supported: {name}")
    return name


def safe_course_code(course_code: str) -> str:
    code = (course_code or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="course_code is required")
    if "/" in code or "\\" in code or code in {".", ".."}:
        raise HTTPException(status_code=400, detail="course_code must be one folder name")

    folder_name = COURSE_CODE_PATTERN.sub("_", code).strip(" .")
    if not folder_name:
        raise HTTPException(status_code=400, detail="course_code must contain folder-safe text")
    if len(folder_name) > 80:
        raise HTTPException(status_code=400, detail="course_code must be 80 characters or fewer")
    return folder_name


def input_dir_for_course(course_code: str) -> Path:
    return INPUT_DIR / safe_course_code(course_code)


def output_dir_for_course(course_code: str) -> Path:
    return OUTPUT_DIR / safe_course_code(course_code)


def batch_dirs(course_code: str | None) -> tuple[Path, Path]:
    if course_code and course_code.strip():
        return input_dir_for_course(course_code), output_dir_for_course(course_code)
    return INPUT_DIR, OUTPUT_DIR


def output_download_url(path: Path) -> str:
    relative_path = path.relative_to(OUTPUT_DIR).as_posix()
    return f"/download/{quote(relative_path, safe='/')}"


def output_file_response(file_path: str) -> FileResponse:
    ensure_dirs()
    candidate = (OUTPUT_DIR / file_path).resolve()
    try:
        candidate.relative_to(OUTPUT_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="File not found") from exc
    if not candidate.exists() or candidate.suffix.lower() != ".txt":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(candidate, media_type="text/plain; charset=utf-8", filename=candidate.name)


def build_processor_args(
    format_model: str,
    ocr_model: str,
    attempts: int,
    rate_limit_wait: float,
    request_delay: float,
    slide_max_tokens: int,
    metadata_max_tokens: int,
    overwrite: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        format_model=format_model,
        ocr_model=ocr_model,
        attempts=attempts,
        rate_limit_wait=rate_limit_wait,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
        keep_uploads=False,
    )


def get_mistral_client():
    load_dotenv_file(BASE_DIR / ".env")
    api_key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="MISTRAL_API_KEY is not set in .env")
    Mistral = import_mistral_client()
    return Mistral(api_key=api_key)


async def save_uploads(files: list[UploadFile], input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for index, upload in enumerate(files, start=1):
        filename = safe_filename(upload.filename or "", f"Document_{index}.pdf")
        target = input_dir / filename
        with target.open("wb") as file_handle:
            while chunk := await upload.read(1024 * 1024):
                file_handle.write(chunk)
        saved_paths.append(target)
    return saved_paths


def process_pdf_paths(pdf_paths: list[Path], args: SimpleNamespace) -> dict[str, list[dict[str, str]]]:
    return process_pdf_paths_to_output(pdf_paths, OUTPUT_DIR, args)


def process_pdf_paths_to_output(
    pdf_paths: list[Path], output_dir: Path, args: SimpleNamespace
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = get_mistral_client()
    outputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    for index, pdf_path in enumerate(pdf_paths, start=1):
        output_path = unique_output_path(output_dir, pdf_path, index, args.overwrite)
        try:
            process_pdf(client, pdf_path, output_path, args)
            outputs.append(
                {
                    "pdf": pdf_path.name,
                    "txt": output_path.relative_to(OUTPUT_DIR).as_posix(),
                    "download_url": output_download_url(output_path),
                }
            )
        except Exception as exc:  # noqa: BLE001 - return per-file failures to the browser/API client.
            errors.append({"pdf": pdf_path.name, "error": str(exc)})

    return {"outputs": outputs, "errors": errors}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    ensure_dirs()
    output_links = "".join(
        (
            f'<li><a href="{html.escape(output_download_url(path))}">'
            f"{html.escape(path.relative_to(OUTPUT_DIR).as_posix())}</a></li>"
        )
        for path in sorted(OUTPUT_DIR.rglob("*.txt"))
    )
    output_links = output_links or "<li>No output files yet.</li>"
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PDF Slide Extraction</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 32px; max-width: 900px; }}
    form {{ margin: 0 0 24px; padding: 16px; border: 1px solid #ddd; border-radius: 8px; }}
    label {{ display: block; margin: 10px 0 4px; font-weight: 600; }}
    input, button {{ font: inherit; }}
    input[type="text"], input[type="number"] {{ width: min(520px, 100%); padding: 8px; }}
    button {{ margin-top: 14px; padding: 9px 14px; cursor: pointer; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>PDF Slide Extraction</h1>

  <form action="/process" method="post" enctype="multipart/form-data">
    <h2>Upload PDF Batch</h2>
    <label for="course_code">Course code / output folder</label>
    <input id="course_code" name="course_code" type="text" placeholder="Example: CS101">
    <label for="files">PDF files</label>
    <input id="files" name="files" type="file" accept="application/pdf" multiple required webkitdirectory directory>
    <label for="format_model">Formatting model</label>
    <input id="format_model" name="format_model" type="text" value="{html.escape(FORMAT_MODEL)}">
    <button type="submit">Process Uploads</button>
  </form>

  <form action="/process-existing" method="post">
    <h2>Process Existing Input Batch</h2>
    <p>Use <code>input</code> or <code>input/&lt;course_code&gt;</code>.</p>
    <label for="existing_course_code">Course code / input-output folder</label>
    <input id="existing_course_code" name="course_code" type="text" placeholder="Example: CS101">
    <label for="existing_format_model">Formatting model</label>
    <input id="existing_format_model" name="format_model" type="text" value="{html.escape(FORMAT_MODEL)}">
    <button type="submit">Process Input Folder</button>
  </form>

  <h2>Generated Files</h2>
  <ul>{output_links}</ul>
</body>
</html>
"""


@app.post("/process")
async def process_uploads(
    files: Annotated[list[UploadFile], File(...)],
    course_code: Annotated[str | None, Form()] = None,
    format_model: Annotated[str, Form()] = FORMAT_MODEL,
    ocr_model: Annotated[str, Form()] = OCR_MODEL,
    attempts: Annotated[int, Form()] = 8,
    rate_limit_wait: Annotated[float, Form()] = 90.0,
    request_delay: Annotated[float, Form()] = 5.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    input_dir, output_dir = batch_dirs(course_code)
    saved_paths = await save_uploads(files, input_dir)

    args = build_processor_args(
        format_model=format_model,
        ocr_model=ocr_model,
        attempts=attempts,
        rate_limit_wait=rate_limit_wait,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths_to_output(saved_paths, output_dir, args)


@app.post("/process/{course_code}")
async def process_course_uploads(
    course_code: str,
    files: Annotated[list[UploadFile], File(...)],
    format_model: Annotated[str, Form()] = FORMAT_MODEL,
    ocr_model: Annotated[str, Form()] = OCR_MODEL,
    attempts: Annotated[int, Form()] = 8,
    rate_limit_wait: Annotated[float, Form()] = 90.0,
    request_delay: Annotated[float, Form()] = 5.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    input_dir, output_dir = batch_dirs(course_code)
    saved_paths = await save_uploads(files, input_dir)

    args = build_processor_args(
        format_model=format_model,
        ocr_model=ocr_model,
        attempts=attempts,
        rate_limit_wait=rate_limit_wait,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths_to_output(saved_paths, output_dir, args)


@app.post("/process-existing")
def process_existing(
    course_code: Annotated[str | None, Form()] = None,
    format_model: Annotated[str, Form()] = FORMAT_MODEL,
    ocr_model: Annotated[str, Form()] = OCR_MODEL,
    attempts: Annotated[int, Form()] = 8,
    rate_limit_wait: Annotated[float, Form()] = 90.0,
    request_delay: Annotated[float, Form()] = 5.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    input_dir, output_dir = batch_dirs(course_code)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(status_code=400, detail=f"No PDF files found in {input_dir}")

    args = build_processor_args(
        format_model=format_model,
        ocr_model=ocr_model,
        attempts=attempts,
        rate_limit_wait=rate_limit_wait,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths_to_output(pdf_paths, output_dir, args)


@app.post("/process-existing/{course_code}")
def process_existing_course(
    course_code: str,
    format_model: Annotated[str, Form()] = FORMAT_MODEL,
    ocr_model: Annotated[str, Form()] = OCR_MODEL,
    attempts: Annotated[int, Form()] = 8,
    rate_limit_wait: Annotated[float, Form()] = 90.0,
    request_delay: Annotated[float, Form()] = 5.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    input_dir, output_dir = batch_dirs(course_code)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(status_code=400, detail=f"No PDF files found in {input_dir}")

    args = build_processor_args(
        format_model=format_model,
        ocr_model=ocr_model,
        attempts=attempts,
        rate_limit_wait=rate_limit_wait,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths_to_output(pdf_paths, output_dir, args)


@app.get("/outputs")
def list_outputs() -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    return {
        "outputs": [
            {"txt": path.relative_to(OUTPUT_DIR).as_posix(), "download_url": output_download_url(path)}
            for path in sorted(OUTPUT_DIR.rglob("*.txt"))
        ]
    }


@app.get("/outputs/{course_code}")
def list_course_outputs(course_code: str) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    output_dir = output_dir_for_course(course_code)
    return {
        "outputs": [
            {"txt": path.relative_to(OUTPUT_DIR).as_posix(), "download_url": output_download_url(path)}
            for path in sorted(output_dir.glob("*.txt"))
        ]
    }


@app.get("/download/{file_path:path}")
def download_output(file_path: str) -> FileResponse:
    return output_file_response(file_path)

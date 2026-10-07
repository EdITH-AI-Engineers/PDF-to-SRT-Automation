from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import html
from pathlib import Path
import re
import sys
from threading import Lock
from types import SimpleNamespace
from typing import Annotated, Any, AsyncIterator
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from folder_queue import FolderQueue
from gpu_profiles import load_gpu_profile
from local_qwen import LocalQwenBackend, cuda_gpu_available, ensure_model
from slides_pdf_to_txt import ProgressTracker, process_pdf, unique_output_path


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = runtime_base_dir()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
PDF_INPUT_PORT = 8001
COURSE_CODE_PATTERN = re.compile(r'[<>:"|?*\x00-\x1f]+')
GPU_PROFILE = load_gpu_profile(BASE_DIR)

_backend: LocalQwenBackend | None = None
_backend_lock = Lock()
_processing_lock = Lock()
_folder_queue: FolderQueue | None = None


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_backend() -> LocalQwenBackend:
    global _backend
    with _backend_lock:
        if _backend is None:
            model_path = ensure_model(MODEL_DIR)
            _backend = LocalQwenBackend(
                model_path,
                n_ctx=GPU_PROFILE.n_ctx,
                n_gpu_layers=GPU_PROFILE.n_gpu_layers,
                n_threads=GPU_PROFILE.n_threads,
                n_threads_batch=GPU_PROFILE.n_threads_batch,
                n_batch=GPU_PROFILE.n_batch,
                n_ubatch=GPU_PROFILE.n_ubatch,
                flash_attn=GPU_PROFILE.flash_attn,
                offload_kqv=GPU_PROFILE.offload_kqv,
                op_offload=GPU_PROFILE.op_offload,
                require_gpu=True,
                target_gpu=GPU_PROFILE.target_gpu,
                cuda_runtime=GPU_PROFILE.cuda_runtime,
                temperature=0,
                seed=42,
            )
        return _backend


def close_backend() -> None:
    global _backend
    with _backend_lock:
        if _backend is not None:
            _backend.close()
            _backend = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_dirs()
    start_folder_queue()
    try:
        yield
    finally:
        stop_folder_queue()
        close_backend()


app = FastAPI(title="PDF Slide Extraction API", lifespan=lifespan)


def safe_filename(filename: str, fallback: str) -> str:
    name = Path(filename or fallback).name.strip() or fallback
    if not name.casefold().endswith(".pdf"):
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


def batch_dirs(course_code: str | None) -> tuple[Path, Path]:
    if course_code and course_code.strip():
        folder_name = safe_course_code(course_code)
        return INPUT_DIR / folder_name, OUTPUT_DIR / folder_name
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
    if not candidate.is_file() or candidate.suffix.casefold() != ".txt":
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        candidate,
        media_type="text/plain; charset=utf-8",
        filename=candidate.name,
    )


def build_processor_args(
    *,
    attempts: int,
    request_delay: float,
    slide_max_tokens: int,
    metadata_max_tokens: int,
    overwrite: bool,
) -> SimpleNamespace:
    if attempts < 1:
        raise HTTPException(status_code=400, detail="attempts must be at least 1")
    if request_delay < 0:
        raise HTTPException(status_code=400, detail="request_delay must be 0 or greater")
    if slide_max_tokens < 1000:
        raise HTTPException(status_code=400, detail="slide_max_tokens must be at least 1000")
    if metadata_max_tokens < 200:
        raise HTTPException(status_code=400, detail="metadata_max_tokens must be at least 200")
    return SimpleNamespace(
        attempts=attempts,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        ocr_min_chars=40,
        ocr_dpi=200,
        overwrite=overwrite,
        delete_inputs=True,
    )


async def save_uploads(files: list[UploadFile], input_dir: Path) -> list[Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for index, upload in enumerate(files, start=1):
        filename = safe_filename(upload.filename or "", f"Document_{index}.pdf")
        target = input_dir / filename
        try:
            with target.open("wb") as file_handle:
                while chunk := await upload.read(1024 * 1024):
                    file_handle.write(chunk)
        finally:
            await upload.close()
        saved_paths.append(target)
    return saved_paths


def process_pdf_paths(
    pdf_paths: list[Path], output_dir: Path, args: SimpleNamespace
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []

    # The local Qwen instance serves one CUDA generation at a time.
    with _processing_lock:
        backend: Any = get_backend()
        progress = ProgressTracker(BASE_DIR)
        for index, pdf_path in enumerate(pdf_paths, start=1):
            # An API request and the background queue may have discovered the
            # same file. The processing lock makes the work sequential; this
            # second check prevents the later caller from processing a file
            # that the first caller has already completed and deleted.
            if not pdf_path.is_file():
                continue
            output_path = unique_output_path(output_dir, pdf_path, index, args.overwrite)
            try:
                process_pdf(backend, pdf_path, output_path, args, progress)
                outputs.append(
                    {
                        "pdf": pdf_path.name,
                        "txt": output_path.relative_to(OUTPUT_DIR).as_posix(),
                        "download_url": output_download_url(output_path),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - report each PDF failure to the caller.
                errors.append({"pdf": pdf_path.name, "error": str(exc)})

    return {"outputs": outputs, "errors": errors}


def process_queued_folder(
    pdf_paths: list[Path], output_dir: Path
) -> dict[str, list[dict[str, str]]]:
    args = build_processor_args(
        attempts=6,
        request_delay=0,
        slide_max_tokens=6000,
        metadata_max_tokens=800,
        overwrite=True,
    )
    return process_pdf_paths(pdf_paths, output_dir, args)


def start_folder_queue() -> FolderQueue:
    global _folder_queue
    if _folder_queue is None:
        _folder_queue = FolderQueue(
            INPUT_DIR,
            OUTPUT_DIR,
            process_queued_folder,
            poll_seconds=1.0,
            settle_seconds=5.0,
        )
    _folder_queue.start()
    return _folder_queue


def stop_folder_queue() -> None:
    global _folder_queue
    if _folder_queue is not None:
        _folder_queue.stop()
        _folder_queue = None


@app.get("/health")
def health() -> dict[str, str | int | bool]:
    return {
        "status": "ok",
        "pdf_input_port": PDF_INPUT_PORT,
        "gpu_backend": "cuda",
        "gpu_required": True,
        "gpu_runtime_available": cuda_gpu_available(),
        "gpu_profile": GPU_PROFILE.profile_id,
        "target_gpu": GPU_PROFILE.target_gpu,
        "cuda_runtime": GPU_PROFILE.cuda_runtime,
        "gpu_layers": GPU_PROFILE.n_gpu_layers,
        "model": "Qwen3-8B Q5_K_M",
        "ocr": "PaddleOCR PP-OCRv5 mobile (CPU)",
        "vision_model": False,
    }


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
    input[type="text"] {{ width: min(520px, 100%); padding: 8px; }}
    button {{ margin-top: 14px; padding: 9px 14px; cursor: pointer; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>PDF Slide Extraction</h1>
  <p>Immediate PDF API: <code>http://127.0.0.1:{PDF_INPUT_PORT}/process</code></p>
  <p>Queued PDF API: <code>http://127.0.0.1:{PDF_INPUT_PORT}/queue/upload</code></p>

  <p><strong>Automatic folder queue:</strong> copy module folders into
  <code>input</code>. Stable folders are processed one at a time and their
  results are written to the matching folder under <code>output</code>.
  <a href="/queue">View queue status</a>.</p>

  <form action="/process" method="post" enctype="multipart/form-data">
    <h2>Upload PDF Batch</h2>
    <label for="course_code">Course code / output folder (optional)</label>
    <input id="course_code" name="course_code" type="text" placeholder="Example: CS101">
    <label for="files">PDF files</label>
    <input id="files" name="files" type="file" accept="application/pdf" multiple required>
    <button type="submit">Process Uploads</button>
  </form>

  <form action="/queue/upload" method="post" enctype="multipart/form-data">
    <h2>Queue PDF Folder</h2>
    <label for="folder_name">Folder / job name (optional)</label>
    <input id="folder_name" name="folder_name" type="text" placeholder="Example: Module 7">
    <label for="queue_files">PDF files</label>
    <input id="queue_files" name="files" type="file" accept="application/pdf" multiple required>
    <button type="submit">Add to Queue</button>
  </form>

  <form action="/process-existing" method="post">
    <h2>Process Existing Input Batch</h2>
    <p>Uses <code>input</code> or <code>input/&lt;course_code&gt;</code>.</p>
    <label for="existing_course_code">Course code (optional)</label>
    <input id="existing_course_code" name="course_code" type="text" placeholder="Example: CS101">
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
    attempts: Annotated[int, Form()] = 6,
    request_delay: Annotated[float, Form()] = 0.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    input_dir, output_dir = batch_dirs(course_code)
    saved_paths = await save_uploads(files, input_dir)
    args = build_processor_args(
        attempts=attempts,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths(saved_paths, output_dir, args)


@app.post("/process/{course_code}")
async def process_course_uploads(
    course_code: str,
    files: Annotated[list[UploadFile], File(...)],
    attempts: Annotated[int, Form()] = 6,
    request_delay: Annotated[float, Form()] = 0.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    input_dir, output_dir = batch_dirs(course_code)
    saved_paths = await save_uploads(files, input_dir)
    args = build_processor_args(
        attempts=attempts,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths(saved_paths, output_dir, args)


@app.post("/process-existing")
def process_existing(
    course_code: Annotated[str | None, Form()] = None,
    attempts: Annotated[int, Form()] = 6,
    request_delay: Annotated[float, Form()] = 0.0,
    slide_max_tokens: Annotated[int, Form()] = 6000,
    metadata_max_tokens: Annotated[int, Form()] = 800,
    overwrite: Annotated[bool, Form()] = True,
) -> dict[str, list[dict[str, str]]]:
    input_dir, output_dir = batch_dirs(course_code)
    pdf_paths = sorted(input_dir.glob("*.pdf"))
    if not pdf_paths:
        raise HTTPException(status_code=400, detail=f"No PDF files found in {input_dir}")
    args = build_processor_args(
        attempts=attempts,
        request_delay=request_delay,
        slide_max_tokens=slide_max_tokens,
        metadata_max_tokens=metadata_max_tokens,
        overwrite=overwrite,
    )
    return process_pdf_paths(pdf_paths, output_dir, args)


@app.get("/queue")
def queue_status() -> dict[str, Any]:
    queue = _folder_queue
    if queue is None:
        return {
            "running": False,
            "current": None,
            "pending": [],
            "failed": [],
            "completed": [],
            "settle_seconds": 5.0,
        }
    queue.discover()
    return queue.snapshot()


def allocate_queue_job_id(folder_name: str | None) -> str:
    if folder_name and folder_name.strip():
        base = safe_course_code(folder_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"job-{timestamp}-{uuid4().hex[:8]}"

    candidate = base
    suffix = 2
    while (INPUT_DIR / candidate).exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


@app.post("/queue/upload", status_code=202)
async def queue_upload(
    files: Annotated[list[UploadFile], File(...)],
    folder_name: Annotated[str | None, Form()] = None,
) -> dict[str, str | int]:
    ensure_dirs()
    job_id = allocate_queue_job_id(folder_name)
    saved_paths = await save_uploads(files, INPUT_DIR / job_id)
    queue = _folder_queue or start_folder_queue()
    queue.discover()
    return {
        "job_id": job_id,
        "state": "queued",
        "pdf_count": len(saved_paths),
        "status_url": f"/queue/{quote(job_id, safe='')}",
        "queue_url": "/queue",
    }


@app.get("/queue/{job_id:path}")
def queue_job_status(job_id: str) -> dict[str, Any]:
    normalized = job_id.strip("/")
    if not normalized or any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise HTTPException(status_code=404, detail="Queue job not found")
    queue = _folder_queue
    if queue is None:
        raise HTTPException(status_code=404, detail="Queue job not found")
    queue.discover()
    status = queue.job_status(normalized)
    if status is None:
        raise HTTPException(status_code=404, detail="Queue job not found")
    return {"job_id": normalized, **status}


@app.post("/queue/retry")
def retry_failed_queue_folders() -> dict[str, int]:
    queue = _folder_queue
    return {"folders_requeued": queue.retry_failed() if queue is not None else 0}


@app.get("/outputs")
def list_outputs() -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    return {
        "outputs": [
            {
                "txt": path.relative_to(OUTPUT_DIR).as_posix(),
                "download_url": output_download_url(path),
            }
            for path in sorted(OUTPUT_DIR.rglob("*.txt"))
        ]
    }


@app.get("/download/{file_path:path}")
def download_output(file_path: str) -> FileResponse:
    return output_file_response(file_path)

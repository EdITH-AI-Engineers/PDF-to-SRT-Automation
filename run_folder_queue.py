from __future__ import annotations

from pathlib import Path
import sys
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any

from folder_queue import FolderQueue
from slides_pdf_to_txt import ProgressTracker, process_pdf, unique_output_path


def runtime_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = runtime_base_dir()
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
_processing_lock = Lock()
_folder_queue: FolderQueue | None = None


def ensure_dirs() -> None:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def processor_args() -> SimpleNamespace:
    return SimpleNamespace(
        ocr_min_chars=40,
        ocr_dpi=200,
        overwrite=True,
        delete_inputs=True,
    )


def process_pdf_paths(
    pdf_paths: list[Path], output_dir: Path
) -> dict[str, list[dict[str, str]]]:
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    args = processor_args()

    with _processing_lock:
        progress = ProgressTracker(BASE_DIR)
        for index, pdf_path in enumerate(pdf_paths, start=1):
            if not pdf_path.is_file():
                continue
            output_path = unique_output_path(output_dir, pdf_path, index, args.overwrite)
            try:
                process_pdf(pdf_path, output_path, args, progress)
                outputs.append({"pdf": pdf_path.name, "txt": output_path.name})
            except Exception as exc:  # noqa: BLE001 - isolate failures per PDF.
                errors.append({"pdf": pdf_path.name, "error": str(exc)})
    return {"outputs": outputs, "errors": errors}


def start_folder_queue() -> FolderQueue:
    global _folder_queue
    if _folder_queue is None:
        _folder_queue = FolderQueue(
            INPUT_DIR,
            OUTPUT_DIR,
            process_pdf_paths,
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


def run_folder_service(stop_event: Event | None = None) -> None:
    """Watch the input folder until stopped, without starting a network server."""
    ensure_dirs()
    queue = start_folder_queue()
    requested_stop = stop_event or Event()
    print(f"Watching for PDFs in: {INPUT_DIR}")
    print(f"Writing extracted text to: {OUTPUT_DIR}")
    print("Press Ctrl+C to stop.")
    try:
        while queue.running and not requested_stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        print("Stopping folder watcher...")
    finally:
        stop_folder_queue()


def main() -> None:
    run_folder_service()


if __name__ == "__main__":
    main()

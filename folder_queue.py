from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import Any, Callable


def _now_text() -> str:
    return datetime.now().isoformat(timespec="seconds")


@dataclass(frozen=True)
class FolderBatch:
    input_dir: Path
    output_dir: Path
    pdfs: tuple[Path, ...]
    fingerprint: tuple[tuple[str, int, int], ...]
    queued_at: float


class FolderQueue:
    """Continuously process stable PDF folders in first-arrived order."""

    def __init__(
        self,
        input_root: Path,
        output_root: Path,
        processor: Callable[[list[Path], Path], dict[str, Any]],
        *,
        poll_seconds: float = 1.0,
        settle_seconds: float = 5.0,
    ) -> None:
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.processor = processor
        self.poll_seconds = poll_seconds
        self.settle_seconds = settle_seconds

        self._lock = Lock()
        self._stop = Event()
        self._thread: Thread | None = None
        self._observed: dict[Path, tuple[tuple[tuple[str, int, int], ...], float]] = {}
        self._failed: dict[Path, tuple[tuple[tuple[str, int, int], ...], str, str]] = {}
        self._pending: list[FolderBatch] = []
        self._current: dict[str, Any] | None = None
        self._completed: list[dict[str, Any]] = []

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.input_root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self._stop.clear()
        self._thread = Thread(
            target=self._run,
            name="pdf-folder-queue",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = None) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    def retry_failed(self) -> int:
        with self._lock:
            count = len(self._failed)
            self._failed.clear()
            return count

    def _folder_name(self, folder: Path) -> str:
        relative = folder.relative_to(self.input_root)
        return "." if str(relative) == "." else relative.as_posix()

    def _output_dir(self, folder: Path) -> Path:
        relative = folder.relative_to(self.input_root)
        return self.output_root if str(relative) == "." else self.output_root / relative

    def _fingerprint(self, folder: Path, pdfs: list[Path]) -> tuple[tuple[str, int, int], ...]:
        entries: list[tuple[str, int, int]] = []
        for pdf in pdfs:
            try:
                stat = pdf.stat()
            except OSError:
                continue
            entries.append((pdf.name.casefold(), stat.st_size, stat.st_mtime_ns))
        return tuple(sorted(entries))

    def discover(self, *, now: float | None = None) -> list[FolderBatch]:
        current_time = time.monotonic() if now is None else now
        grouped: dict[Path, list[Path]] = {}
        try:
            candidates = sorted(self.input_root.rglob("*.pdf"))
        except OSError:
            candidates = []
        for pdf in candidates:
            if pdf.is_file():
                grouped.setdefault(pdf.parent, []).append(pdf)

        batches: list[FolderBatch] = []
        active_folders = set(grouped)
        with self._lock:
            for folder in list(self._observed):
                if folder not in active_folders:
                    self._observed.pop(folder, None)
                    self._failed.pop(folder, None)

            for folder, pdfs in grouped.items():
                if (
                    self._current is not None
                    and self._current.get("folder") == self._folder_name(folder)
                ):
                    continue
                fingerprint = self._fingerprint(folder, pdfs)
                if not fingerprint:
                    continue

                previous = self._observed.get(folder)
                if previous is None or previous[0] != fingerprint:
                    stable_since = current_time
                    self._observed[folder] = (fingerprint, stable_since)
                    failed = self._failed.get(folder)
                    if failed is not None and failed[0] != fingerprint:
                        self._failed.pop(folder, None)
                else:
                    stable_since = previous[1]

                failed = self._failed.get(folder)
                if failed is not None and failed[0] == fingerprint:
                    continue

                batch = FolderBatch(
                    input_dir=folder,
                    output_dir=self._output_dir(folder),
                    pdfs=tuple(sorted(pdfs)),
                    fingerprint=fingerprint,
                    queued_at=stable_since,
                )
                batches.append(batch)

            batches.sort(
                key=lambda item: (
                    item.queued_at,
                    self._folder_name(item.input_dir).casefold(),
                )
            )
            self._pending = batches
        return batches

    def next_ready(self, *, now: float | None = None) -> FolderBatch | None:
        current_time = time.monotonic() if now is None else now
        for batch in self.discover(now=current_time):
            if current_time - batch.queued_at >= self.settle_seconds:
                return batch
        return None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = [
                {
                    "folder": self._folder_name(batch.input_dir),
                    "pdf_count": len(batch.pdfs),
                    "state": "queued",
                }
                for batch in self._pending
            ]
            failed = [
                {
                    "folder": self._folder_name(folder),
                    "state": "failed",
                    "error": details[1],
                    "failed_at": details[2],
                }
                for folder, details in sorted(
                    self._failed.items(), key=lambda item: str(item[0]).casefold()
                )
            ]
            return {
                "running": self.running,
                "current": dict(self._current) if self._current else None,
                "pending": pending,
                "failed": failed,
                "completed": [dict(item) for item in self._completed[-20:]],
                "settle_seconds": self.settle_seconds,
            }

    def job_status(self, folder_name: str) -> dict[str, Any] | None:
        """Return the latest queue state for one folder-backed job."""
        snapshot = self.snapshot()
        current = snapshot["current"]
        if current is not None and current.get("folder") == folder_name:
            return current
        for section in ("pending", "failed", "completed"):
            for item in snapshot[section]:
                if item.get("folder") == folder_name:
                    return item
        return None

    def _record_failure(self, batch: FolderBatch, message: str) -> None:
        remaining = [pdf for pdf in batch.pdfs if pdf.is_file()]
        fingerprint = self._fingerprint(batch.input_dir, remaining)
        if not fingerprint:
            fingerprint = batch.fingerprint
        with self._lock:
            self._observed[batch.input_dir] = (fingerprint, batch.queued_at)
            self._failed[batch.input_dir] = (fingerprint, message, _now_text())

    def _process(self, batch: FolderBatch) -> None:
        folder_name = self._folder_name(batch.input_dir)
        with self._lock:
            self._current = {
                "folder": folder_name,
                "pdf_count": len(batch.pdfs),
                "state": "processing",
                "started_at": _now_text(),
            }
            self._pending = [
                item for item in self._pending if item.input_dir != batch.input_dir
            ]

        try:
            existing_pdfs = [pdf for pdf in batch.pdfs if pdf.is_file()]
            if not existing_pdfs:
                return
            result = self.processor(existing_pdfs, batch.output_dir)
            errors = result.get("errors", []) if isinstance(result, dict) else []
            outputs = result.get("outputs", []) if isinstance(result, dict) else []
            if errors:
                message = "; ".join(
                    f"{item.get('pdf', 'PDF')}: {item.get('error', 'processing failed')}"
                    for item in errors
                    if isinstance(item, dict)
                ) or "Folder processing failed"
                self._record_failure(batch, message)
            elif any(pdf.is_file() for pdf in existing_pdfs):
                errors = [
                    {
                        "pdf": folder_name,
                        "error": "Output was created, but a processed input PDF could not be deleted.",
                    }
                ]
                self._record_failure(batch, errors[0]["error"])
            with self._lock:
                if not errors:
                    self._observed.pop(batch.input_dir, None)
                self._completed.append(
                    {
                        "folder": folder_name,
                        "state": "failed" if errors else "completed",
                        "pdf_count": len(batch.pdfs),
                        "output_count": len(outputs),
                        "error_count": len(errors),
                        "finished_at": _now_text(),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - queue must survive a bad folder.
            self._record_failure(batch, str(exc))
        finally:
            with self._lock:
                self._current = None

    def _run(self) -> None:
        while not self._stop.is_set():
            batch = self.next_ready()
            if batch is not None:
                self._process(batch)
                continue
            self._stop.wait(self.poll_seconds)

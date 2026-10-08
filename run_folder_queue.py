from __future__ import annotations

from threading import Event

from app import (
    INPUT_DIR,
    OUTPUT_DIR,
    close_backend,
    ensure_dirs,
    start_folder_queue,
    stop_folder_queue,
)


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
        close_backend()


def main() -> None:
    run_folder_service()


if __name__ == "__main__":
    main()

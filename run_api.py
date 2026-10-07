from __future__ import annotations

import uvicorn

from app import PDF_INPUT_PORT, app, ensure_dirs


HOST = "127.0.0.1"
PORT = PDF_INPUT_PORT


def main() -> None:
    ensure_dirs()
    print(f"PDF input API running at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()

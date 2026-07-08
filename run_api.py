from __future__ import annotations

import uvicorn

from app import app, ensure_dirs


HOST = "127.0.0.1"
PORT = 8000


def main() -> None:
    ensure_dirs()
    print(f"PDF Slide Extraction API running at http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()

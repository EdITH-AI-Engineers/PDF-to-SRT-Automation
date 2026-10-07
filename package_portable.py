from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def create_archive(source_dir: Path, archive_path: Path) -> None:
    source_dir = source_dir.resolve()
    archive_path = archive_path.resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(source_dir.parent))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("archive_path", type=Path)
    args = parser.parse_args()
    create_archive(args.source_dir, args.archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

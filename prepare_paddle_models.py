from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from paddle_ocr import (
    DEFAULT_CPU_THREADS,
    DEFAULT_RECOGNITION_BATCH_SIZE,
    DETECTION_MODEL,
    RECOGNITION_MODEL,
    model_bundle_is_complete,
    write_model_bundle_manifest,
)


def prepare_models(
    destination: Path,
    *,
    cache_root: Path | None = None,
    pipeline_factory=None,
) -> None:
    destination = Path(destination).resolve()
    required_models = (DETECTION_MODEL, RECOGNITION_MODEL)
    if model_bundle_is_complete(destination):
        return

    if pipeline_factory is None:
        from paddleocr import PaddleOCR

        pipeline_factory = PaddleOCR

    pipeline_factory(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
        enable_mkldnn=False,
        cpu_threads=DEFAULT_CPU_THREADS,
        text_recognition_batch_size=DEFAULT_RECOGNITION_BATCH_SIZE,
    )

    cache_root = (
        Path(cache_root).resolve()
        if cache_root is not None
        else Path.home() / ".paddlex" / "official_models"
    )
    temporary = destination.with_name(f".{destination.name}.preparing")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        for model_name in required_models:
            source = cache_root / model_name
            if not source.is_dir():
                raise RuntimeError(
                    f"PaddleOCR did not download the expected model: {source}"
                )
            shutil.copytree(source, temporary / model_name)
        write_model_bundle_manifest(temporary)
        if not model_bundle_is_complete(temporary):
            raise RuntimeError(f"PaddleOCR model verification failed in {temporary}")
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    prepare_models(args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Callable, Iterable


DETECTION_MODEL = "PP-OCRv5_mobile_det"
RECOGNITION_MODEL = "en_PP-OCRv5_mobile_rec"
MODEL_MANIFEST = "model-manifest.json"
DEFAULT_CPU_THREADS = min(8, max(1, (os.cpu_count() or 1) // 2))
DEFAULT_RECOGNITION_BATCH_SIZE = 8


def _model_snapshot(model_root: Path) -> dict[str, list[dict[str, object]]]:
    snapshot: dict[str, list[dict[str, object]]] = {}
    for model_name in (DETECTION_MODEL, RECOGNITION_MODEL):
        model_dir = Path(model_root) / model_name
        files = []
        if model_dir.is_dir():
            for path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                files.append(
                    {
                        "path": path.relative_to(model_dir).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": digest.hexdigest(),
                    }
                )
        snapshot[model_name] = files
    return snapshot


def write_model_bundle_manifest(model_root: Path) -> None:
    model_root = Path(model_root)
    snapshot = _model_snapshot(model_root)
    if not all(snapshot.values()):
        raise RuntimeError(f"Incomplete PaddleOCR model bundle in {model_root}")
    (model_root / MODEL_MANIFEST).write_text(
        json.dumps({"models": snapshot}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def model_bundle_is_complete(model_root: Path) -> bool:
    model_root = Path(model_root)
    try:
        manifest = json.loads((model_root / MODEL_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    expected = manifest.get("models") if isinstance(manifest, dict) else None
    if not isinstance(expected, dict):
        return False
    actual = _model_snapshot(model_root)
    return all(actual.values()) and actual == expected


def _recognized_lines(result: Any) -> tuple[str, ...]:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unexpected result")

    values = payload.get("res", payload).get("rec_texts", ())
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("PaddleOCR result did not contain recognized text")
    return tuple(text for value in values if (text := str(value).strip()))


class PaddleOcrBackend:
    """Reusable CPU-only PaddleOCR pipeline for scanned slide pages."""

    def __init__(
        self,
        model_root: Path,
        *,
        pipeline_factory: Callable[..., Any] | None = None,
    ) -> None:
        os.environ["FLAGS_enable_pir_api"] = "0"
        model_root = Path(model_root)
        detection_dir = model_root / DETECTION_MODEL
        recognition_dir = model_root / RECOGNITION_MODEL
        local_models = (detection_dir.is_dir(), recognition_dir.is_dir())
        if any(local_models) and not all(local_models):
            raise RuntimeError(f"Incomplete bundled PaddleOCR models in {model_root}")
        if getattr(sys, "frozen", False) and not model_bundle_is_complete(model_root):
            raise RuntimeError(
                f"The bundled PaddleOCR models are missing or incomplete in {model_root}; "
                "reinstall or rebuild the portable package"
            )

        if pipeline_factory is None:
            try:
                from paddleocr import PaddleOCR
            except (ImportError, OSError, RuntimeError) as exc:
                raise RuntimeError(
                    "PaddleOCR is unavailable; install the project requirements"
                ) from exc
            pipeline_factory = PaddleOCR

        pipeline_options: dict[str, Any] = {
            "text_detection_model_name": DETECTION_MODEL,
            "text_recognition_model_name": RECOGNITION_MODEL,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
            "enable_mkldnn": False,
            "cpu_threads": DEFAULT_CPU_THREADS,
            "text_recognition_batch_size": DEFAULT_RECOGNITION_BATCH_SIZE,
        }
        if all(local_models):
            pipeline_options.update(
                text_detection_model_dir=str(detection_dir),
                text_recognition_model_dir=str(recognition_dir),
            )

        self._pipeline = pipeline_factory(**pipeline_options)
        self._lock = Lock()

    def recognize_many(self, image_bytes: Iterable[bytes]) -> list[str]:
        from PIL import Image
        import numpy as np

        inputs = []
        for value in image_bytes:
            with Image.open(BytesIO(value)) as image:
                rgb = np.asarray(image.convert("RGB"))
                inputs.append(np.ascontiguousarray(rgb[:, :, ::-1]))
        if not inputs:
            return []

        with self._lock:
            results = list(self._pipeline.predict(inputs))
        if len(results) != len(inputs):
            raise RuntimeError(
                f"PaddleOCR returned {len(results)} results for {len(inputs)} images"
            )
        return ["\n".join(_recognized_lines(result)) for result in results]

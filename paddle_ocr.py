from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Any, Callable, Iterable


SMALL_DETECTION_MODEL = "PP-OCRv6_small_det"
SMALL_RECOGNITION_MODEL = "PP-OCRv6_small_rec"
MEDIUM_DETECTION_MODEL = "PP-OCRv6_medium_det"
MEDIUM_RECOGNITION_MODEL = "PP-OCRv6_medium_rec"
DETECTION_MODEL = SMALL_DETECTION_MODEL
RECOGNITION_MODEL = SMALL_RECOGNITION_MODEL
MODEL_PAIRS = (
    (SMALL_DETECTION_MODEL, SMALL_RECOGNITION_MODEL),
    (MEDIUM_DETECTION_MODEL, MEDIUM_RECOGNITION_MODEL),
)
MODEL_NAMES = tuple(model_name for pair in MODEL_PAIRS for model_name in pair)
MODEL_MANIFEST = "model-manifest.json"
DEFAULT_CPU_THREADS = min(8, max(1, (os.cpu_count() or 1) // 2))
DEFAULT_RECOGNITION_BATCH_SIZE = 8
DEFAULT_PREPROCESS_WORKERS = min(4, max(1, (os.cpu_count() or 1) // 4))
DEFAULT_MEDIUM_FALLBACK_CONFIDENCE = 0.82


@dataclass(frozen=True)
class RecognizedText:
    text: str
    confidence: float


def _model_snapshot(model_root: Path) -> dict[str, list[dict[str, object]]]:
    snapshot: dict[str, list[dict[str, object]]] = {}
    for model_name in MODEL_NAMES:
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


def _recognized_text(result: Any) -> RecognizedText:
    payload = getattr(result, "json", result)
    if callable(payload):
        payload = payload()
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise RuntimeError("PaddleOCR returned an unexpected result")

    result_payload = payload.get("res", payload)
    values = result_payload.get("rec_texts", ())
    if not isinstance(values, (list, tuple)):
        raise RuntimeError("PaddleOCR result did not contain recognized text")

    score_values = result_payload.get("rec_scores", ())
    if not isinstance(score_values, (list, tuple)):
        score_values = ()

    lines: list[str] = []
    weighted_scores: list[tuple[float, int]] = []
    for index, value in enumerate(values):
        text = str(value).strip()
        if not text:
            continue
        lines.append(text)
        if index < len(score_values):
            try:
                score = min(1.0, max(0.0, float(score_values[index])))
            except (TypeError, ValueError):
                continue
            weight = max(1, sum(character.isalnum() for character in text))
            weighted_scores.append((score, weight))

    if weighted_scores:
        total_weight = sum(weight for _, weight in weighted_scores)
        confidence = sum(score * weight for score, weight in weighted_scores) / total_weight
    else:
        confidence = 1.0 if lines else 0.0
    return RecognizedText("\n".join(lines), confidence)


def _decode_image(value: bytes) -> Any:
    from PIL import Image
    import numpy as np

    with Image.open(BytesIO(value)) as image:
        rgb = np.asarray(image.convert("RGB"))
        return np.ascontiguousarray(rgb[:, :, ::-1])


class PaddleOcrBackend:
    """Reusable CPU-only PaddleOCR pipeline for scanned slide pages."""

    def __init__(
        self,
        model_root: Path,
        *,
        pipeline_factory: Callable[..., Any] | None = None,
        fallback_confidence: float = DEFAULT_MEDIUM_FALLBACK_CONFIDENCE,
    ) -> None:
        os.environ["FLAGS_enable_pir_api"] = "0"
        model_root = Path(model_root)
        local_models = tuple((model_root / name).is_dir() for name in MODEL_NAMES)
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

        if not 0.0 <= fallback_confidence <= 1.0:
            raise ValueError("fallback_confidence must be between 0 and 1")

        self._model_root = model_root
        self._local_models = all(local_models)
        self._pipeline_factory = pipeline_factory
        self._fallback_confidence = fallback_confidence
        self._medium_pipeline: Any | None = None
        self._small_pipeline = self._create_pipeline(
            SMALL_DETECTION_MODEL,
            SMALL_RECOGNITION_MODEL,
        )
        self._lock = Lock()

    def _create_pipeline(self, detection_model: str, recognition_model: str) -> Any:
        pipeline_options: dict[str, Any] = {
            "text_detection_model_name": detection_model,
            "text_recognition_model_name": recognition_model,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": "cpu",
            "enable_mkldnn": False,
            "cpu_threads": DEFAULT_CPU_THREADS,
            "text_recognition_batch_size": DEFAULT_RECOGNITION_BATCH_SIZE,
        }
        if self._local_models:
            pipeline_options.update(
                text_detection_model_dir=str(self._model_root / detection_model),
                text_recognition_model_dir=str(self._model_root / recognition_model),
            )
        return self._pipeline_factory(**pipeline_options)

    def _get_medium_pipeline(self) -> Any:
        if self._medium_pipeline is None:
            self._medium_pipeline = self._create_pipeline(
                MEDIUM_DETECTION_MODEL,
                MEDIUM_RECOGNITION_MODEL,
            )
        return self._medium_pipeline

    @staticmethod
    def _predict(pipeline: Any, inputs: list[Any]) -> list[RecognizedText]:
        results = list(pipeline.predict(inputs))
        if len(results) != len(inputs):
            raise RuntimeError(
                f"PaddleOCR returned {len(results)} results for {len(inputs)} images"
            )
        return [_recognized_text(result) for result in results]

    def recognize_many(self, image_bytes: Iterable[bytes]) -> list[str]:
        encoded_images = list(image_bytes)
        if not encoded_images:
            return []

        workers = min(DEFAULT_PREPROCESS_WORKERS, len(encoded_images))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                inputs = list(executor.map(_decode_image, encoded_images))
        else:
            inputs = [_decode_image(encoded_images[0])]

        with self._lock:
            recognized = self._predict(self._small_pipeline, inputs)
            fallback_indexes = [
                index
                for index, result in enumerate(recognized)
                if not result.text or result.confidence < self._fallback_confidence
            ]
            if fallback_indexes:
                fallback_inputs = [inputs[index] for index in fallback_indexes]
                medium_results = self._predict(
                    self._get_medium_pipeline(),
                    fallback_inputs,
                )
                for index, medium_result in zip(
                    fallback_indexes, medium_results, strict=True
                ):
                    small_result = recognized[index]
                    if medium_result.text and (
                        not small_result.text
                        or medium_result.confidence >= small_result.confidence
                    ):
                        recognized[index] = medium_result
        return [result.text for result in recognized]

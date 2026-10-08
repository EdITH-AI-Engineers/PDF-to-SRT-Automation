from io import BytesIO
import importlib
import importlib.util
from pathlib import Path
import socket
import sys
from threading import Event
from types import SimpleNamespace
import zipfile

import pytest

import slides_pdf_to_txt


def test_folder_queue_waits_for_stable_folders_and_preserves_output_structure(tmp_path):
    from folder_queue import FolderQueue

    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    module_dir = input_root / "CS101" / "Module 1"
    module_dir.mkdir(parents=True)
    pdf = module_dir / "slides.pdf"
    pdf.write_bytes(b"pdf")
    calls = []

    def processor(paths, destination):
        calls.append((paths, destination))
        paths[0].unlink()
        return {"outputs": [{"pdf": paths[0].name}], "errors": []}

    queue = FolderQueue(input_root, output_root, processor, settle_seconds=5)
    assert queue.next_ready(now=10) is None
    batch = queue.next_ready(now=15)
    assert batch is not None
    assert batch.pdfs == (pdf,)
    assert batch.output_dir == output_root / "CS101" / "Module 1"

    queue._process(batch)

    assert calls == [([pdf], output_root / "CS101" / "Module 1")]
    assert queue.snapshot()["completed"][-1]["folder"] == "CS101/Module 1"


def test_folder_queue_does_not_spin_on_unchanged_failure(tmp_path):
    from folder_queue import FolderQueue

    input_root = tmp_path / "input"
    module_dir = input_root / "Broken Module"
    module_dir.mkdir(parents=True)
    pdf = module_dir / "slides.pdf"
    pdf.write_bytes(b"pdf")
    queue = FolderQueue(
        input_root,
        tmp_path / "output",
        lambda paths, destination: {
            "outputs": [],
            "errors": [{"pdf": paths[0].name, "error": "bad PDF"}],
        },
        settle_seconds=0,
    )

    batch = queue.next_ready(now=1)
    assert batch is not None
    queue._process(batch)

    assert queue.next_ready(now=100) is None
    assert queue.snapshot()["failed"][0]["folder"] == "Broken Module"
    assert queue.retry_failed() == 1
    assert queue.next_ready(now=100) is not None


def test_folder_only_runner_never_creates_a_network_socket(monkeypatch):
    module_spec = importlib.util.find_spec("run_folder_queue")
    assert module_spec is not None
    runner = importlib.import_module("run_folder_queue")
    events = []

    class FakeQueue:
        running = True

    def reject_socket(*args, **kwargs):
        raise AssertionError("folder-only runner must not create a network socket")

    monkeypatch.setattr(socket, "socket", reject_socket)
    monkeypatch.setattr(runner, "ensure_dirs", lambda: events.append("ensure_dirs"))
    monkeypatch.setattr(
        runner,
        "start_folder_queue",
        lambda: events.append("start_folder_queue") or FakeQueue(),
    )
    monkeypatch.setattr(
        runner, "stop_folder_queue", lambda: events.append("stop_folder_queue")
    )
    stop_event = Event()
    stop_event.set()

    runner.run_folder_service(stop_event)

    assert events == ["ensure_dirs", "start_folder_queue", "stop_folder_queue"]


def test_dependencies_and_packaging_are_ocr_only():
    project_root = Path(__file__).parents[1]
    requirements = (project_root / "requirements-common.txt").read_text(encoding="utf-8")
    spec = (project_root / "PDFSlideTextExtractor.spec").read_text(encoding="utf-8")
    build = (project_root / "build_portables.ps1").read_text(encoding="utf-8")

    assert "paddleocr==3.7.0" in requirements
    assert "paddlepaddle==3.3.1" in requirements
    for forbidden in ("llama", "huggingface", "nvidia-cuda", "fastapi", "uvicorn"):
        assert forbidden not in requirements.casefold()
    assert "llama_cpp" not in spec
    assert "cuda" not in spec.casefold()
    assert "PDFSlideTextExtractor-OCR" in spec
    assert "Qwen" not in build
    assert "cuda" not in build.casefold()
    assert "prepare_paddle_models.py" in build


def test_parse_args_exposes_only_ocr_processing_options():
    args = slides_pdf_to_txt.parse_args([])

    assert args.ocr_min_chars == 40
    assert args.ocr_dpi == 200
    assert args.delete_inputs is True
    for removed in (
        "model_dir",
        "n_ctx",
        "n_gpu_layers",
        "slide_max_tokens",
        "metadata_max_tokens",
        "request_delay",
    ):
        assert not hasattr(args, removed)


def test_ocr_text_is_formatted_deterministically_without_generation():
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    page = pdf_ingestion.ExtractedPage(
        1,
        "Module 4\nOperating Systems\nProcesses\nA process is a running program.",
        "text+equation-ocr",
        ("t = 4 s",),
    )

    assert slides_pdf_to_txt.extract_module_metadata(page) == ("4", "Operating Systems")
    slide = slides_pdf_to_txt.format_slide(1, page)

    assert slide == {
        "title": "Module 4",
        "content": "Operating Systems\nProcesses\nA process is a running program.",
        "equations": ("t = 4 s",),
        "descriptions": [
            {"label": "Image/Diagram Description", "text": "Not Specified"}
        ],
    }
    rendered = slides_pdf_to_txt.render_document("4", "Operating Systems", [slide])
    assert "Equation 1: t = 4 s" in rendered
    assert "Brief Explanation:" not in rendered


def test_paddle_ocr_disables_incompatible_onednn_and_preserves_line_order(
    tmp_path, monkeypatch
):
    paddle_ocr = importlib.import_module("paddle_ocr")
    captured = {}

    class Result:
        def __init__(self, lines):
            self.json = {"res": {"rec_texts": lines, "rec_scores": [0.99] * len(lines)}}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, inputs):
            return [Result(["Title", "First bullet", "Second bullet"]) for _ in inputs]

    backend = paddle_ocr.PaddleOcrBackend(tmp_path, pipeline_factory=FakePipeline)
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
    assert backend.recognize_many([stream.getvalue()]) == [
        "Title\nFirst bullet\nSecond bullet"
    ]
    assert captured["device"] == "cpu"
    assert captured["enable_mkldnn"] is False
    assert captured["text_recognition_batch_size"] == 8
    assert captured["text_detection_model_name"] == "PP-OCRv6_small_det"
    assert captured["text_recognition_model_name"] == "PP-OCRv6_small_rec"


def test_paddle_ocr_retries_only_low_confidence_pages_with_medium(tmp_path):
    paddle_ocr = importlib.import_module("paddle_ocr")
    calls = []

    class Result:
        def __init__(self, text, score):
            self.json = {"res": {"rec_texts": [text], "rec_scores": [score]}}

    class FakePipeline:
        def __init__(self, **kwargs):
            self.model = kwargs["text_recognition_model_name"]
            calls.append(("create", self.model))

        def predict(self, inputs):
            calls.append(("predict", self.model, len(inputs)))
            if self.model == paddle_ocr.SMALL_RECOGNITION_MODEL:
                return [Result("Clear title", 0.97), Result("uncertain", 0.51)]
            return [Result("Corrected medium text", 0.94)]

    backend = paddle_ocr.PaddleOcrBackend(tmp_path, pipeline_factory=FakePipeline)
    from PIL import Image

    stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="PNG")
    image = stream.getvalue()

    assert backend.recognize_many([image, image]) == [
        "Clear title",
        "Corrected medium text",
    ]
    assert calls == [
        ("create", paddle_ocr.SMALL_RECOGNITION_MODEL),
        ("predict", paddle_ocr.SMALL_RECOGNITION_MODEL, 2),
        ("create", paddle_ocr.MEDIUM_RECOGNITION_MODEL),
        ("predict", paddle_ocr.MEDIUM_RECOGNITION_MODEL, 1),
    ]


def test_local_pdf_extraction_uses_text_then_paddle_fallback(tmp_path):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "slides.pdf"
    source.write_bytes(b"pdf")
    ocr_calls = []

    class FakePage:
        def __init__(self, text):
            self.text = text

        def get_text(self, *args, **kwargs):
            return self.text

    class FakeDocument:
        page_count = 2

        def load_page(self, index):
            return [
                FakePage("Readable embedded title\nEnough embedded slide text"),
                FakePage("x"),
            ][index]

        def close(self):
            pass

    def ocr(page, dpi):
        ocr_calls.append((page.text, dpi))
        return "Scanned title\nRecognized body"

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        dpi=200,
        document_factory=lambda path: FakeDocument(),
        ocr=ocr,
        equation_ocr=lambda page, dpi: (),
    )

    assert pages[0].method == "text"
    assert pages[1].method == "ocr"
    assert pages[1].text == "Scanned title\nRecognized body"
    assert ocr_calls == [("x", 200)]


def test_default_paddle_path_batches_pages_and_reuses_text_for_equations(
    tmp_path, monkeypatch
):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "slides.pdf"
    source.write_bytes(b"pdf")
    batches = []

    class FakePage:
        def __init__(self, index):
            self.index = index

        def get_text(self, *args, **kwargs):
            return ""

    class FakeDocument:
        page_count = 10

        def load_page(self, index):
            return FakePage(index)

        def close(self):
            pass

    class Backend:
        def recognize_many(self, images):
            batches.append(list(images))
            return [f"Slide {value}\nx = {value}" for value in images]

    monkeypatch.setattr(pdf_ingestion, "_render_page_for_ocr", lambda page, dpi: page.index)
    monkeypatch.setattr(pdf_ingestion, "_get_paddle_ocr_backend", lambda: Backend())

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        document_factory=lambda path: FakeDocument(),
    )

    assert [len(batch) for batch in batches] == [8, 2]
    assert len(pages) == 10
    assert pages[3].equations == ("x = 3",)
    assert "x = 3" not in pages[3].text


def test_gibberish_lines_are_dropped_but_equations_are_kept():
    pdf_ingestion = importlib.import_module("pdf_ingestion")

    page = pdf_ingestion._build_extracted_page(
        1,
        "",
        "QWRT PLMNB\nUseful definition\ny = 2x + 1",
        ("y = 2x + 1",),
        40,
    )

    assert page.text == "Useful definition"
    assert page.equations == ("y = 2x + 1",)


def test_paddle_ocr_uses_and_verifies_bundled_models(tmp_path, monkeypatch):
    paddle_ocr = importlib.import_module("paddle_ocr")
    model_root = tmp_path / "paddleocr"
    for model_name in paddle_ocr.MODEL_NAMES:
        (model_root / model_name).mkdir(parents=True)
    detection_dir = model_root / paddle_ocr.SMALL_DETECTION_MODEL
    recognition_dir = model_root / paddle_ocr.SMALL_RECOGNITION_MODEL
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    paddle_ocr.PaddleOcrBackend(model_root, pipeline_factory=FakePipeline)
    assert captured["text_detection_model_dir"] == str(detection_dir)
    assert captured["text_recognition_model_dir"] == str(recognition_dir)

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with pytest.raises(RuntimeError, match="bundled PaddleOCR models are missing"):
        paddle_ocr.PaddleOcrBackend(
            tmp_path / "missing", pipeline_factory=lambda **kwargs: object()
        )


def test_prepare_paddle_models_replaces_and_reuses_complete_bundle(tmp_path):
    paddle_ocr = importlib.import_module("paddle_ocr")
    prepare = importlib.import_module("prepare_paddle_models")
    destination = tmp_path / "bundle"
    stale_model = destination / paddle_ocr.DETECTION_MODEL
    stale_model.mkdir(parents=True)
    (stale_model / "partial.bin").write_bytes(b"partial")
    cache_root = tmp_path / "cache"
    for model_name in paddle_ocr.MODEL_NAMES:
        model_dir = cache_root / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "inference.bin").write_bytes(model_name.encode())

    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    prepare.prepare_models(
        destination, cache_root=cache_root, pipeline_factory=FakePipeline
    )
    prepare.prepare_models(
        destination,
        cache_root=cache_root,
        pipeline_factory=lambda **kwargs: pytest.fail("complete bundle rebuilt"),
    )

    assert len(calls) == 2
    assert not (destination / paddle_ocr.DETECTION_MODEL / "partial.bin").exists()
    assert paddle_ocr.model_bundle_is_complete(destination)


def test_process_pdf_writes_only_ocr_derived_content(tmp_path, monkeypatch):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    output = tmp_path / "module.txt"
    source.write_bytes(b"pdf")
    page = pdf_ingestion.ExtractedPage(
        1,
        "Module 4\nOperating Systems\nProcesses\nA process is a running program.",
        "text",
    )
    monkeypatch.setattr(
        slides_pdf_to_txt, "extract_pdf_pages", lambda path, **kwargs: (page,)
    )
    args = slides_pdf_to_txt.parse_args([str(source), "--keep-inputs"])

    slides_pdf_to_txt.process_pdf(source, output, args)

    text = output.read_text(encoding="utf-8")
    assert "Module #: 4" in text
    assert "Module Title: Operating Systems" in text
    assert "A process is a running program." in text
    assert "Brief Explanation:" not in text
    assert source.is_file()


def test_main_processes_batch_without_loading_a_model(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    for name in ("first.pdf", "second.pdf"):
        (input_dir / name).write_bytes(b"pdf")
    monkeypatch.setattr(slides_pdf_to_txt, "BASE_DIR", tmp_path)
    events = []

    def process(source, output, args, progress):
        events.append(source.name)
        output.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(slides_pdf_to_txt, "process_pdf", process)
    exit_code = slides_pdf_to_txt.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--keep-inputs",
        ]
    )

    assert exit_code == 0
    assert events == ["first.pdf", "second.pdf"]


def test_ocr_portable_archive_contains_no_qwen_or_llama_runtime():
    archive_path = (
        Path(__file__).parents[1] / "dist" / "PDFSlideTextExtractor-OCR.zip"
    )
    if not archive_path.is_file():
        pytest.skip("OCR-only portable archive has not been built")
    with zipfile.ZipFile(archive_path) as archive:
        names = [name.replace("\\", "/").casefold() for name in archive.namelist()]

    assert any(name.endswith("/pdfslidetextextractor.exe") for name in names)
    assert any("models/paddleocr/" in name for name in names)
    assert not any("qwen" in name or "llama_cpp" in name for name in names)
    assert not any(Path(name).name.startswith("cudart") for name in names)

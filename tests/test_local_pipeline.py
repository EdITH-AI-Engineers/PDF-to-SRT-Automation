from pathlib import Path
from io import BytesIO
import hashlib
import importlib
import importlib.util
import os
import socket
import sys
from threading import Event
from types import SimpleNamespace
import zipfile

import pytest

import slides_pdf_to_txt


def requirement_include_closure(path):
    discovered = set()

    def visit(requirement_path):
        requirement_path = requirement_path.resolve()
        if requirement_path in discovered:
            return
        discovered.add(requirement_path)
        for raw_line in requirement_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("-r "):
                visit(requirement_path.parent / line[3:].strip())

    visit(Path(path))
    return {item.name for item in discovered}


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

    queue = FolderQueue(
        input_root,
        output_root,
        processor,
        settle_seconds=5,
    )

    assert queue.next_ready(now=10) is None
    batch = queue.next_ready(now=15)
    assert batch is not None
    assert batch.pdfs == (pdf,)
    assert batch.output_dir == output_root / "CS101" / "Module 1"

    queue._process(batch)

    assert calls == [([pdf], output_root / "CS101" / "Module 1")]
    assert queue.snapshot()["completed"][-1]["folder"] == "CS101/Module 1"


def test_folder_queue_does_not_spin_on_an_unchanged_failed_folder(tmp_path):
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


def test_folder_queue_reports_one_job_state(tmp_path):
    from folder_queue import FolderQueue

    input_root = tmp_path / "input"
    job_dir = input_root / "Module 7"
    job_dir.mkdir(parents=True)
    (job_dir / "slides.pdf").write_bytes(b"pdf")
    queue = FolderQueue(
        input_root,
        tmp_path / "output",
        lambda paths, destination: {"outputs": [], "errors": []},
        settle_seconds=5,
    )

    queue.discover(now=10)

    assert queue.job_status("Module 7") == {
        "folder": "Module 7",
        "pdf_count": 1,
        "state": "queued",
    }
    assert queue.job_status("missing") is None


def test_folder_only_runner_watches_directories_without_binding_port(monkeypatch):
    module_spec = importlib.util.find_spec("run_folder_queue")
    assert module_spec is not None, "folder-only runner is missing"
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
        runner,
        "stop_folder_queue",
        lambda: events.append("stop_folder_queue"),
    )
    monkeypatch.setattr(
        runner,
        "close_backend",
        lambda: events.append("close_backend"),
    )
    stop_event = Event()
    stop_event.set()

    runner.run_folder_service(stop_event)

    assert events == [
        "ensure_dirs",
        "start_folder_queue",
        "stop_folder_queue",
        "close_backend",
    ]


def test_build_dependencies_do_not_pull_a_second_gpu_runtime():
    project_root = Path(__file__).parents[1]

    dev_files = requirement_include_closure(project_root / "requirements-dev.txt")
    gtx_files = requirement_include_closure(
        project_root / "requirements-gtx1050ti.txt"
    )

    assert "requirements-rtx3060.txt" not in dev_files
    assert "requirements.txt" not in dev_files
    assert "requirements-gtx1050ti.txt" in gtx_files
    assert "requirements-common.txt" in gtx_files
    common_requirements = (project_root / "requirements-common.txt").read_text(
        encoding="utf-8"
    )
    assert "paddleocr==3.7.0" in common_requirements
    assert "paddlepaddle==3.3.1" in common_requirements
    assert "pytesseract" not in common_requirements


def test_gtx_portable_build_uses_custom_cuda_102_wheel_and_runtime():
    project_root = Path(__file__).parents[1]
    build_script = (project_root / "build_portables.ps1").read_text(encoding="utf-8")
    requirements = (project_root / "requirements-gtx1050ti.txt").read_text(
        encoding="utf-8"
    )
    spec = (project_root / "PDFSlideTextExtractor.spec").read_text(encoding="utf-8")

    assert 'CudaMajor = "10"' in build_script
    assert 'VenvName = "venv-cu102"' in build_script
    assert "cuda102-toolchain" in build_script
    assert "PDF_EXTRACTOR_CUDA_BIN" in build_script
    assert "-m ensurepip --upgrade" in build_script
    assert "cu118" not in requirements
    assert "nvidia-cuda-runtime-cu11" not in requirements
    assert "cudart64_102.dll" in spec
    assert "cublas64_10.dll" in spec
    assert "cublasLt64_10.dll" in spec
    assert '"llava.dll"' in spec
    assert "TesseractDir" not in build_script
    assert "prepare_paddle_models.py" in build_script
    assert 'collect_data_files("paddlex")' in spec
    assert 'collect_dynamic_libs("paddle")' in spec


def test_health_reports_the_selected_gpu_profile(monkeypatch):
    api = importlib.import_module("app")
    monkeypatch.setattr(api, "cuda_gpu_available", lambda: True)
    monkeypatch.setattr(
        api,
        "GPU_PROFILE",
        SimpleNamespace(
            profile_id="gtx-1050-ti",
            target_gpu="NVIDIA GeForce GTX 1050 Ti 4 GB",
            cuda_runtime="10.2",
            n_ctx=8192,
            n_gpu_layers=16,
            n_threads=8,
            n_threads_batch=16,
            n_batch=128,
            n_ubatch=64,
            flash_attn=False,
            offload_kqv=False,
            op_offload=False,
        ),
        raising=False,
    )

    assert api.health() == {
        "status": "ok",
        "pdf_input_port": 8001,
        "gpu_backend": "cuda",
        "gpu_required": True,
        "gpu_runtime_available": True,
        "gpu_profile": "gtx-1050-ti",
        "target_gpu": "NVIDIA GeForce GTX 1050 Ti 4 GB",
        "cuda_runtime": "10.2",
        "gpu_layers": 16,
        "model": "Qwen3-8B Q5_K_M",
        "ocr": "PaddleOCR PP-OCRv5 mobile (CPU)",
        "vision_model": False,
    }


def test_api_backend_uses_the_selected_gpu_memory_profile(tmp_path, monkeypatch):
    api = importlib.import_module("app")
    model_path = tmp_path / "Qwen3-8B-Q5_K_M.gguf"
    captured = {}

    class Backend:
        def __init__(self, path, **kwargs):
            captured["path"] = path
            captured["kwargs"] = kwargs

    monkeypatch.setattr(api, "_backend", None)
    monkeypatch.setattr(api, "ensure_model", lambda _: model_path)
    monkeypatch.setattr(api, "LocalQwenBackend", Backend)
    monkeypatch.setattr(
        api,
        "GPU_PROFILE",
        SimpleNamespace(
            profile_id="gtx-1050-ti",
            target_gpu="NVIDIA GeForce GTX 1050 Ti 4 GB",
            cuda_runtime="10.2",
            n_ctx=8192,
            n_gpu_layers=16,
            n_threads=8,
            n_threads_batch=16,
            n_batch=128,
            n_ubatch=64,
            flash_attn=False,
            offload_kqv=False,
            op_offload=False,
        ),
        raising=False,
    )

    backend = api.get_backend()

    assert backend.__class__ is Backend
    assert captured == {
        "path": model_path,
        "kwargs": {
            "n_ctx": 8192,
            "n_gpu_layers": 16,
            "n_threads": 8,
            "n_threads_batch": 16,
            "n_batch": 128,
            "n_ubatch": 64,
            "flash_attn": False,
            "offload_kqv": False,
            "op_offload": False,
            "require_gpu": True,
            "target_gpu": "NVIDIA GeForce GTX 1050 Ti 4 GB",
            "cuda_runtime": "10.2",
            "temperature": 0,
            "seed": 42,
        },
    }
    api._backend = None


def test_load_gpu_profile_uses_packaged_profile_id(tmp_path):
    api = importlib.import_module("app")
    (tmp_path / "gpu-profile.json").write_text(
        '{"profile_id": "gtx-1050-ti"}', encoding="utf-8"
    )

    profile = api.load_gpu_profile(tmp_path)

    assert profile.profile_id == "gtx-1050-ti"
    assert profile.target_gpu == "NVIDIA GeForce GTX 1050 Ti 4 GB"
    assert profile.cuda_runtime == "10.2"
    assert profile.n_ctx == 8192
    assert profile.n_gpu_layers == 16
    assert profile.n_batch == 128
    assert profile.n_ubatch == 64
    assert profile.flash_attn is False
    assert profile.offload_kqv is False
    assert profile.op_offload is False


def test_load_gpu_profile_defaults_to_rtx_3060_when_unpacked(tmp_path):
    api = importlib.import_module("app")

    profile = api.load_gpu_profile(tmp_path)

    assert profile.profile_id == "rtx-3060"
    assert profile.target_gpu == "NVIDIA GeForce RTX 3060 12 GB"
    assert profile.cuda_runtime == "13.2"
    assert profile.n_ctx == 8192
    assert profile.n_gpu_layers == -1
    assert profile.n_batch == 512
    assert profile.n_ubatch == 512
    assert profile.flash_attn is True
    assert profile.offload_kqv is True
    assert profile.op_offload is True


def test_pdf_input_api_accepts_multipart_uploads(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    api = importlib.import_module("app")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    monkeypatch.setattr(api, "INPUT_DIR", input_dir)
    monkeypatch.setattr(api, "OUTPUT_DIR", output_dir)
    calls = []

    def process(paths, destination, args):
        calls.append((paths, destination, args))
        return {"outputs": [{"pdf": paths[0].name}], "errors": []}

    monkeypatch.setattr(api, "process_pdf_paths", process)

    with TestClient(api.app) as client:
        response = client.post(
            "/process",
            data={"course_code": "CS101"},
            files={"files": ("module.pdf", b"%PDF-test", "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {"outputs": [{"pdf": "module.pdf"}], "errors": []}
    assert (input_dir / "CS101" / "module.pdf").read_bytes() == b"%PDF-test"
    assert calls[0][0] == [input_dir / "CS101" / "module.pdf"]
    assert calls[0][1] == output_dir / "CS101"


def test_queue_upload_returns_a_job_status_url_without_processing(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from folder_queue import FolderQueue

    api = importlib.import_module("app")
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    queue = FolderQueue(
        input_dir,
        output_dir,
        lambda paths, destination: pytest.fail("upload endpoint processed synchronously"),
        settle_seconds=5,
    )
    monkeypatch.setattr(api, "INPUT_DIR", input_dir)
    monkeypatch.setattr(api, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(api, "_folder_queue", queue)
    monkeypatch.setattr(api, "start_folder_queue", lambda: queue)
    monkeypatch.setattr(api, "stop_folder_queue", lambda: None)
    monkeypatch.setattr(api, "close_backend", lambda: None)

    with TestClient(api.app) as client:
        response = client.post(
            "/queue/upload",
            data={"folder_name": "Module 7"},
            files=[
                ("files", ("part-1.pdf", b"%PDF-one", "application/pdf")),
                ("files", ("part-2.pdf", b"%PDF-two", "application/pdf")),
            ],
        )
        status_response = client.get("/queue/Module%207")

    assert response.status_code == 202
    assert response.json() == {
        "job_id": "Module 7",
        "state": "queued",
        "pdf_count": 2,
        "status_url": "/queue/Module%207",
        "queue_url": "/queue",
    }
    assert status_response.status_code == 200
    assert status_response.json() == {
        "job_id": "Module 7",
        "folder": "Module 7",
        "pdf_count": 2,
        "state": "queued",
    }
    assert (input_dir / "Module 7" / "part-1.pdf").read_bytes() == b"%PDF-one"
    assert (input_dir / "Module 7" / "part-2.pdf").read_bytes() == b"%PDF-two"


def test_parse_args_exposes_local_qwen_and_ocr_options():
    args = slides_pdf_to_txt.parse_args([])

    assert Path(args.model_dir) == slides_pdf_to_txt.BASE_DIR / "models"
    assert args.n_ctx == 8192
    assert args.n_gpu_layers == -1
    assert args.n_threads == 8
    assert args.n_threads_batch == 16
    assert args.n_batch == 512
    assert args.n_ubatch == 512
    assert args.flash_attn is True
    assert args.require_gpu is True
    assert args.seed == 42
    assert args.ocr_min_chars == 40
    assert args.ocr_dpi == 200
    assert args.request_delay == 0
    assert not hasattr(args, "visual_dpi")
    assert not hasattr(args, "format_model")
    assert not hasattr(args, "ocr_model")
    assert not hasattr(args, "keep_uploads")


def test_model_download_uses_the_locked_text_only_qwen3_model(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        downloaded = tmp_path / kwargs["filename"]
        downloaded.write_bytes(b"gguf")
        return str(downloaded)

    monkeypatch.setattr(local_qwen, "hf_hub_download", fake_download)
    monkeypatch.setattr(local_qwen, "MODEL_SIZE", 4, raising=False)
    monkeypatch.setattr(
        local_qwen,
        "MODEL_SHA256",
        hashlib.sha256(b"gguf").hexdigest(),
        raising=False,
    )
    model_path = local_qwen.ensure_model(tmp_path)
    assert model_path == tmp_path / local_qwen.MODEL_FILENAME
    assert calls == [
        {
            "repo_id": local_qwen.MODEL_REPO,
            "revision": local_qwen.MODEL_REVISION,
            "filename": local_qwen.MODEL_FILENAME,
            "local_dir": str(tmp_path),
        }
    ]


def test_downloaded_model_is_verified_before_returning(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    downloaded = tmp_path / local_qwen.MODEL_FILENAME

    def fake_download(**kwargs):
        downloaded.write_bytes(b"nope")
        return str(downloaded)

    monkeypatch.setattr(local_qwen, "hf_hub_download", fake_download)
    monkeypatch.setattr(local_qwen, "MODEL_SIZE", 4, raising=False)
    monkeypatch.setattr(
        local_qwen,
        "MODEL_SHA256",
        hashlib.sha256(b"gguf").hexdigest(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        local_qwen.ensure_model(tmp_path)


def test_python_model_identity_matches_the_committed_lock():
    local_qwen = importlib.import_module("local_qwen")
    lock_path = Path(__file__).parents[1] / "model-lock.json"
    lock = __import__("json").loads(lock_path.read_text(encoding="utf-8"))["qwen"]

    assert lock == {
        "repo_id": local_qwen.MODEL_REPO,
        "revision": local_qwen.MODEL_REVISION,
        "model": {
            "filename": local_qwen.MODEL_FILENAME,
            "size": local_qwen.MODEL_SIZE,
            "sha256": local_qwen.MODEL_SHA256,
        },
    }


def test_cuda_dll_search_is_configured_only_once(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    cuda_bin = tmp_path / "site-packages" / "nvidia" / "cublas" / "bin"
    cuda_bin.mkdir(parents=True)
    registrations = []

    monkeypatch.setattr(local_qwen.os, "name", "nt")
    monkeypatch.setattr(local_qwen.os, "add_dll_directory", registrations.append)
    monkeypatch.setattr(
        local_qwen.site,
        "getsitepackages",
        lambda: [str(tmp_path / "site-packages")],
    )
    monkeypatch.delattr(local_qwen.sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(local_qwen, "_CUDA_DLL_SEARCH_CONFIGURED", False)
    local_qwen._DLL_DIRECTORY_HANDLES.clear()

    local_qwen._configure_cuda_dll_search()
    local_qwen._configure_cuda_dll_search()

    assert registrations == [str(cuda_bin)]


def test_existing_model_with_wrong_hash_is_rejected(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    target = tmp_path / local_qwen.MODEL_FILENAME
    target.write_bytes(b"nope")
    monkeypatch.setattr(local_qwen, "MODEL_SIZE", 4, raising=False)
    monkeypatch.setattr(
        local_qwen,
        "MODEL_SHA256",
        hashlib.sha256(b"gguf").hexdigest(),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="SHA-256"):
        local_qwen.ensure_model(tmp_path)


def test_local_qwen_preserves_prompts_and_disables_thinking(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def create_chat_completion(self, **kwargs):
            captured["completion"] = kwargs
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, llama_supports_gpu_offload=lambda: True),
    )
    model_path = tmp_path / "Qwen3-8B-Q5_K_M.gguf"
    backend = local_qwen.LocalQwenBackend(
        model_path,
        n_ctx=8192,
        n_gpu_layers=7,
        temperature=0,
        seed=42,
    )

    assert backend.complete("preserved system", "preserved user", max_tokens=321) == '{"ok": true}'
    assert captured == {
        "init": {
            "model_path": str(model_path),
            "n_ctx": 8192,
            "n_gpu_layers": 7,
            "main_gpu": 0,
            "n_batch": 512,
            "n_ubatch": 512,
            "n_threads": 8,
            "n_threads_batch": 16,
            "offload_kqv": True,
            "op_offload": True,
            "flash_attn": True,
            "use_mmap": True,
            "use_mlock": False,
            "seed": 42,
            "verbose": False,
        },
        "completion": {
            "messages": [
                {"role": "system", "content": "preserved system"},
                {"role": "user", "content": "preserved user\n\n/no_think"},
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": 321,
            "response_format": {"type": "json_object"},
        },
    }


def test_local_qwen_accepts_low_vram_offload_controls(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    captured = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=FakeLlama, llama_supports_gpu_offload=lambda: True),
    )

    local_qwen.LocalQwenBackend(
        tmp_path / "Qwen3-8B-Q5_K_M.gguf",
        n_gpu_layers=16,
        n_batch=128,
        n_ubatch=64,
        flash_attn=False,
        offload_kqv=False,
        op_offload=False,
        target_gpu="NVIDIA GeForce GTX 1050 Ti 4 GB",
        cuda_runtime="10.2",
    )

    assert captured["n_gpu_layers"] == 16
    assert captured["n_batch"] == 128
    assert captured["n_ubatch"] == 64
    assert captured["flash_attn"] is False
    assert captured["offload_kqv"] is False
    assert captured["op_offload"] is False


def test_backend_supports_legacy_llama_binding_without_op_offload(
    tmp_path, monkeypatch
):
    local_qwen = importlib.import_module("local_qwen")
    captured = {}

    class LegacyLlama:
        def __init__(
            self,
            model_path,
            n_ctx,
            n_gpu_layers,
            main_gpu,
            n_batch,
            n_ubatch,
            n_threads,
            n_threads_batch,
            offload_kqv,
            flash_attn,
            use_mmap,
            use_mlock,
            seed,
            verbose,
        ):
            captured.update(locals())

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=LegacyLlama, llama_supports_gpu_offload=lambda: True),
    )

    local_qwen.LocalQwenBackend(
        tmp_path / "Qwen3-8B-Q5_K_M.gguf",
        op_offload=False,
        require_gpu=True,
        cuda_runtime="10.2",
    )

    assert "op_offload" not in captured
    assert captured["n_gpu_layers"] == -1


def test_slide_formatting_uses_only_filtered_ocr_and_skips_image_descriptions():
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    page = pdf_ingestion.ExtractedPage(
        6,
        "PACPETTETTLL ANAS\ndv/dt = 9.8 - v/5",
        "ocr",
    )
    calls = []

    class Backend:
        def complete(self, system, user, *, max_tokens):
            calls.append((system, user, max_tokens))
            return (
                '{"title":"Not Specified","content":"dv/dt = 9.8 - v/5",'
                '"descriptions":[{"label":"Image/Diagram Description",'
                '"text":"PACPETTETTLL ANAS"}],'
                '"brief_explanation":"The equation is shown on the slide."}'
            )

    result = slides_pdf_to_txt.format_slide(Backend(), 6, page, 1, 1000)

    assert result["content"] == "dv/dt = 9.8 - v/5"
    assert result["descriptions"] == [
        {"label": "Image/Diagram Description", "text": "Not Specified"}
    ]
    assert "OCR page" in calls[0][0]
    assert "image_png" not in calls[0][1]


def test_local_qwen_rejects_a_cpu_only_runtime(tmp_path, monkeypatch):
    local_qwen = importlib.import_module("local_qwen")
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(
            Llama=object,
            llama_supports_gpu_offload=lambda: False,
        ),
    )
    with pytest.raises(RuntimeError, match="CUDA GPU offload is unavailable"):
        local_qwen.LocalQwenBackend(tmp_path / "model.gguf")


def test_paddle_ocr_disables_incompatible_onednn_and_preserves_line_order(
    tmp_path, monkeypatch
):
    from PIL import Image

    paddle_ocr = importlib.import_module("paddle_ocr")
    captured = {}
    monkeypatch.setenv("FLAGS_enable_pir_api", "1")

    class FakePipeline:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def predict(self, inputs):
            captured["input_shapes"] = [item.shape for item in inputs]
            return [
                SimpleNamespace(
                    json={"res": {"rec_texts": ["Slide title", "First bullet"]}}
                ),
                SimpleNamespace(json={"res": {"rec_texts": ["Second page"]}}),
            ]

    image_buffer = BytesIO()
    Image.new("RGB", (12, 8), "white").save(image_buffer, format="PNG")
    backend = paddle_ocr.PaddleOcrBackend(
        tmp_path / "paddleocr",
        pipeline_factory=FakePipeline,
    )

    texts = backend.recognize_many([image_buffer.getvalue()] * 2)

    assert texts == ["Slide title\nFirst bullet", "Second page"]
    assert captured["init"] == {
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "enable_mkldnn": False,
        "cpu_threads": paddle_ocr.DEFAULT_CPU_THREADS,
        "text_recognition_batch_size": 8,
    }
    assert os.environ["FLAGS_enable_pir_api"] == "0"
    assert captured["input_shapes"] == [(8, 12, 3), (8, 12, 3)]


def test_local_pdf_extraction_uses_text_then_paddle_fallback(tmp_path):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    events = []

    class FakePage:
        def __init__(self, text):
            self.text = text

        def get_text(self, mode, *, sort):
            assert (mode, sort) == ("text", True)
            return self.text

    class FakeDocument:
        page_count = 2

        def __init__(self):
            self.pages = [
                FakePage("A readable native-text slide with enough content."),
                FakePage("scan"),
            ]

        def load_page(self, index):
            return self.pages[index]

        def close(self):
            events.append("closed")

    def ocr(page, dpi):
        events.append((page.text, dpi))
        return "Readable words recovered from the scanned slide."

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        dpi=240,
        document_factory=lambda path: FakeDocument(),
        ocr=ocr,
    )

    assert pages == (
        pdf_ingestion.ExtractedPage(
            1, "A readable native-text slide with enough content.", "text"
        ),
        pdf_ingestion.ExtractedPage(
            2, "Readable words recovered from the scanned slide.", "ocr"
        ),
    )
    assert events == [("scan", 240), "closed"]


def test_default_paddle_ocr_batches_pages_and_reuses_lines_for_equations(
    tmp_path, monkeypatch
):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    batches = []

    class FakePixmap:
        def __init__(self, value):
            self.value = value

        def tobytes(self, output):
            assert output == "png"
            return self.value

    class FakePage:
        def __init__(self, number, text):
            self.number = number
            self.text = text

        def get_text(self, mode, *, sort):
            assert (mode, sort) == ("text", True)
            return self.text

        def get_pixmap(self, *, dpi, alpha):
            assert (dpi, alpha) == (200, False)
            return FakePixmap(f"page-{self.number}".encode())

    class FakeDocument:
        page_count = 2

        def __init__(self):
            self.pages = [
                FakePage(
                    1,
                    "Velocity Model\nm = 10 kg\nA readable slide with enough content.",
                ),
                FakePage(2, "scan"),
            ]

        def load_page(self, index):
            return self.pages[index]

        def close(self):
            pass

    def recognize_many(images):
        batches.append(list(images))
        return [
            "Velocity Model\nm = 10 kg",
            "Recovered scanned slide.\ndv/dt = 9.8 - v/5",
        ]

    monkeypatch.setattr(pdf_ingestion, "_ocr_images", recognize_many, raising=False)

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        document_factory=lambda path: FakeDocument(),
    )

    assert batches == [[b"page-1", b"page-2"]]
    assert pages == (
        pdf_ingestion.ExtractedPage(
            1,
            "Velocity Model\nA readable slide with enough content.",
            "text+equation-ocr",
            ("m = 10 kg",),
        ),
        pdf_ingestion.ExtractedPage(
            2,
            "Recovered scanned slide.",
            "ocr+equation-ocr",
            ("dv/dt = 9.8 - v/5",),
        ),
    )


def test_default_paddle_ocr_limits_page_batches_for_large_documents(
    tmp_path, monkeypatch
):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    batches = []

    class FakePixmap:
        def __init__(self, value):
            self.value = value

        def tobytes(self, output):
            assert output == "png"
            return self.value

    class FakePage:
        def __init__(self, number):
            self.number = number

        def get_text(self, mode, *, sort):
            assert (mode, sort) == ("text", True)
            return f"Readable native text for slide {self.number}."

        def get_pixmap(self, *, dpi, alpha):
            assert (dpi, alpha) == (200, False)
            return FakePixmap(f"page-{self.number}".encode())

    class FakeDocument:
        page_count = 17

        def load_page(self, index):
            return FakePage(index + 1)

        def close(self):
            pass

    def recognize_many(images):
        batch = list(images)
        batches.append(batch)
        return ["" for _ in batch]

    monkeypatch.setattr(pdf_ingestion, "_ocr_images", recognize_many)

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        document_factory=lambda path: FakeDocument(),
    )

    assert [len(batch) for batch in batches] == [8, 8, 1]
    assert [page.number for page in pages] == list(range(1, 18))


def test_corrupt_embedded_text_is_replaced_by_clean_paddle_ocr(tmp_path):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    corrupt_text = (
        "PACPETTETTLL ANAS\nLETPIL ELE TT VANS\n"
        "PPFETELEL UI AVVANAS\nZEPPAL ERA VV NAN\n"
        "CFEELI EPL IT VV VAN AASS"
    )

    class FakePage:
        def get_text(self, mode, *, sort):
            return corrupt_text

    class FakeDocument:
        page_count = 1

        def load_page(self, index):
            return FakePage()

        def close(self):
            pass

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        document_factory=lambda path: FakeDocument(),
        ocr=lambda page, dpi: "Suppose m = 10 kg\ndv/dt = 9.8 - v/5",
    )

    assert pages == (
        pdf_ingestion.ExtractedPage(
            1,
            "Suppose m = 10 kg\ndv/dt = 9.8 - v/5",
            "ocr",
        ),
    )


def test_gibberish_ocr_lines_are_dropped_but_equations_are_kept(tmp_path):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")

    class FakePage:
        def get_text(self, mode, *, sort):
            return "scan"

    class FakeDocument:
        page_count = 1

        def load_page(self, index):
            return FakePage()

        def close(self):
            pass

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        document_factory=lambda path: FakeDocument(),
        ocr=lambda page, dpi: (
            "Suppose m = 10 kg\n"
            "PACPETTETTLL ANAS\n"
            "dv/dt = 9.8 - v/5\n"
            "CFEELI EPL IT VV VAN AASS"
        ),
    )

    assert pages[0].text == "Suppose m = 10 kg\ndv/dt = 9.8 - v/5"


def test_equation_cleanup_preserves_direct_equations_missed_by_ocr():
    pdf_ingestion = importlib.import_module("pdf_ingestion")

    page = pdf_ingestion._build_extracted_page(
        1,
        "System model\nx = 1\ny = 2",
        "x = 1",
        ("x = 1",),
        min_chars=1,
    )

    assert page.text == "System model\ny = 2"
    assert page.equations == ("x = 1",)


def test_equations_use_separate_local_ocr_snapshots_even_with_readable_pdf_text(
    tmp_path,
):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    source.write_bytes(b"pdf")
    events = []

    class FakePage:
        def get_text(self, mode, *, sort):
            assert (mode, sort) == ("text", True)
            return (
                "Velocity Model\n"
                "Suppose m = 10 kg\n"
                "The slide introduces a falling-body example.\n"
                "dv/dt = 9.8 - v/5"
            )

    class FakeDocument:
        page_count = 1

        def load_page(self, index):
            return FakePage()

        def close(self):
            events.append("closed")

    def equation_ocr(page, dpi):
        events.append((page.__class__.__name__, dpi))
        return ("m = 10 kg", "dv/dt = 9.8 - v/5")

    pages = pdf_ingestion.extract_pdf_pages(
        source,
        min_chars=20,
        dpi=300,
        document_factory=lambda path: FakeDocument(),
        equation_ocr=equation_ocr,
    )

    assert pages == (
        pdf_ingestion.ExtractedPage(
            1,
            "Velocity Model\nThe slide introduces a falling-body example.",
            "text+equation-ocr",
            ("m = 10 kg", "dv/dt = 9.8 - v/5"),
        ),
    )
    assert events == [("FakePage", 300), "closed"]


def test_equation_snapshots_render_as_separate_local_ocr_text_entries():
    rendered = slides_pdf_to_txt.render_document(
        "2",
        "Differential Equations",
        [
            {
                "title": "Velocity Model",
                "content": "A falling-body example is introduced.",
                "equations": ("m = 10 kg", "dv/dt = 9.8 - v/5"),
                "descriptions": [
                    {
                        "label": "Image/Diagram Description",
                        "text": "Not Specified",
                    }
                ],
                "brief_explanation": "The slide presents a velocity model.",
            }
        ],
    )

    assert (
        "Equations (local OCR):\n"
        "Equation 1: m = 10 kg\n"
        "Equation 2: dv/dt = 9.8 - v/5\n\n"
    ) in rendered


def test_slide_formatter_keeps_snapshot_equations_out_of_model_generated_content():
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    page = pdf_ingestion.ExtractedPage(
        1,
        "A falling-body example is introduced.",
        "text+equation-ocr",
        ("dv/dt = 9.8 - v/5",),
    )
    calls = []

    class Backend:
        def complete(self, system, user, *, max_tokens):
            calls.append((system, user))
            return (
                '{"title":"Velocity Model","content":"A falling-body example is introduced.",'
                '"descriptions":[],"brief_explanation":"The slide presents a velocity model."}'
            )

    result = slides_pdf_to_txt.format_slide(Backend(), 1, page, 1, 1000)

    assert result["equations"] == ("dv/dt = 9.8 - v/5",)
    assert "Do not copy equation OCR into content" in calls[0][0]
    assert '"equations": [' in calls[0][1]


def test_paddle_ocr_uses_bundled_model_directories(tmp_path):
    paddle_ocr = importlib.import_module("paddle_ocr")
    model_root = tmp_path / "paddleocr"
    detection_dir = model_root / paddle_ocr.DETECTION_MODEL
    recognition_dir = model_root / paddle_ocr.RECOGNITION_MODEL
    detection_dir.mkdir(parents=True)
    recognition_dir.mkdir()
    captured = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    paddle_ocr.PaddleOcrBackend(model_root, pipeline_factory=FakePipeline)

    assert captured["text_detection_model_dir"] == str(detection_dir)
    assert captured["text_recognition_model_dir"] == str(recognition_dir)


def test_frozen_paddle_ocr_requires_bundled_models(tmp_path, monkeypatch):
    paddle_ocr = importlib.import_module("paddle_ocr")
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with pytest.raises(RuntimeError, match="bundled PaddleOCR models are missing"):
        paddle_ocr.PaddleOcrBackend(
            tmp_path / "paddleocr",
            pipeline_factory=lambda **kwargs: object(),
        )


def test_prepare_paddle_models_replaces_and_reuses_only_complete_bundles(tmp_path):
    paddle_ocr = importlib.import_module("paddle_ocr")
    prepare_paddle_models = importlib.import_module("prepare_paddle_models")
    destination = tmp_path / "bundle"
    stale_model = destination / paddle_ocr.DETECTION_MODEL
    stale_model.mkdir(parents=True)
    (stale_model / "partial.bin").write_bytes(b"partial")

    cache_root = tmp_path / "cache"
    for model_name in (paddle_ocr.DETECTION_MODEL, paddle_ocr.RECOGNITION_MODEL):
        model_dir = cache_root / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "inference.bin").write_bytes(model_name.encode())

    calls = []

    class FakePipeline:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    prepare_paddle_models.prepare_models(
        destination,
        cache_root=cache_root,
        pipeline_factory=FakePipeline,
    )
    prepare_paddle_models.prepare_models(
        destination,
        cache_root=cache_root,
        pipeline_factory=lambda **kwargs: pytest.fail("complete bundle rebuilt"),
    )

    assert len(calls) == 1
    assert not (destination / paddle_ocr.DETECTION_MODEL / "partial.bin").exists()
    assert paddle_ocr.model_bundle_is_complete(destination)


def test_process_pdf_uses_local_pages_with_preserved_prompts_and_output(
    tmp_path, monkeypatch
):
    pdf_ingestion = importlib.import_module("pdf_ingestion")
    source = tmp_path / "module.pdf"
    output = tmp_path / "module.txt"
    source.write_bytes(b"pdf")
    page = pdf_ingestion.ExtractedPage(
        1, "Native slide text about operating systems.", "text"
    )
    monkeypatch.setattr(
        slides_pdf_to_txt,
        "extract_pdf_pages",
        lambda path, **kwargs: (page,),
        raising=False,
    )

    calls = []

    class Backend:
        def complete(self, system, user, *, max_tokens):
            calls.append((system, user, max_tokens))
            if len(calls) == 1:
                return '{"module_number": "4", "module_title": "Operating Systems"}'
            return (
                '{"title": "Processes", "content": "A process is a running program.", '
                '"descriptions": [{"label": "Image/Diagram Description", '
                '"text": "Not Specified"}], "brief_explanation": '
                '"A process is an executing program. The operating system manages it."}'
            )

    args = slides_pdf_to_txt.parse_args(
        [
            str(source),
            "--attempts",
            "1",
            "--request-delay",
            "0",
            "--keep-inputs",
        ]
    )

    slides_pdf_to_txt.process_pdf(Backend(), source, output, args)

    assert "title slide" in calls[0][0]
    assert "text-only OCR pipeline" in calls[1][0]
    assert "Never infer what an image" in calls[1][0]
    assert '"text": "Native slide text about operating systems."' in calls[0][1]
    assert '"method": "text"' in calls[1][1]
    assert output.read_text(encoding="utf-8") == (
        "Module #: 4\n\n"
        "Module Title: Operating Systems\n\n"
        "---\n\n"
        "Slide 1:\n"
        "{\n"
        "Title:\n"
        "Processes\n\n"
        "Content:\n"
        "A process is a running program.\n\n"
        "Image/Diagram Description:\n"
        "Not Specified\n"
        "}\n\n"
        "Brief Explanation:\n"
        "A process is an executing program. The operating system manages it.\n\n"
        "---\n"
    )
    assert source.is_file()


def test_main_loads_one_local_backend_for_the_whole_batch(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    model_dir = tmp_path / "models"
    input_dir.mkdir()
    first = input_dir / "first.pdf"
    second = input_dir / "second.pdf"
    first.write_bytes(b"pdf")
    second.write_bytes(b"pdf")
    monkeypatch.setattr(slides_pdf_to_txt, "BASE_DIR", tmp_path)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    events = []

    def ensure(path):
        events.append(("model", Path(path)))
        return model_dir / "Qwen3-8B-Q5_K_M.gguf"

    class Backend:
        def __init__(self, model_path, **kwargs):
            events.append(("backend", model_path, kwargs))

        def close(self):
            events.append(("close",))

    def process(backend, source, output, args, progress):
        events.append(("process", backend, source.name))
        output.write_text("ok\n", encoding="utf-8")

    monkeypatch.setattr(slides_pdf_to_txt, "ensure_model", ensure)
    monkeypatch.setattr(slides_pdf_to_txt, "LocalQwenBackend", Backend)
    monkeypatch.setattr(slides_pdf_to_txt, "process_pdf", process)

    exit_code = slides_pdf_to_txt.main(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--model-dir",
            str(model_dir),
            "--n-gpu-layers",
            "3",
            "--keep-inputs",
        ]
    )

    backend = events[1]
    assert exit_code == 0
    assert events[0] == ("model", model_dir)
    assert backend == (
        "backend",
        model_dir / "Qwen3-8B-Q5_K_M.gguf",
        {
            "n_ctx": 8192,
            "n_gpu_layers": 3,
            "n_threads": 8,
            "n_threads_batch": 16,
            "n_batch": 512,
            "n_ubatch": 512,
            "flash_attn": True,
            "require_gpu": True,
            "temperature": 0,
            "seed": 42,
        },
    )
    assert events[2][0] == "process"
    assert events[2][1].__class__ is Backend
    assert events[2][2] == "first.pdf"
    assert events[3][0] == "process"
    assert events[3][1] is events[2][1]
    assert events[3][2] == "second.pdf"
    assert events[4] == ("close",)


@pytest.mark.parametrize(
    ("archive_name", "folder_name", "profile_id", "cuda_major"),
    [
        (
            "PDFSlideTextExtractor-GTX1050Ti.zip",
            "PDFSlideTextExtractor-GTX1050Ti",
            "gtx-1050-ti",
            "10",
        ),
        (
            "PDFSlideTextExtractor-RTX3060.zip",
            "PDFSlideTextExtractor-RTX3060",
            "rtx-3060",
            "13",
        ),
    ],
)
def test_portable_archives_match_their_gpu_profiles(
    archive_name, folder_name, profile_id, cuda_major
):
    archive_path = Path(__file__).parents[1] / "dist" / archive_name
    if not archive_path.is_file():
        pytest.skip(f"portable archive has not been built: {archive_name}")
    root = folder_name.casefold()

    with zipfile.ZipFile(archive_path) as archive:
        names = [name.replace("\\", "/").casefold() for name in archive.namelist()]
        profile = __import__("json").loads(
            archive.read(f"{folder_name}/gpu-profile.json").decode("utf-8")
        )

    cudart_name = {
        "10": "cudart64_102.dll",
        "11": "cudart64_110.dll",
    }.get(cuda_major, f"cudart64_{cuda_major}.dll")
    required_paths = {
        f"{root}/pdfslidetextextractor.exe",
        f"{root}/gpu-profile.json",
        f"{root}/models/qwen3-8b-q5_k_m.gguf",
        f"{root}/runtime/llama_cpp/lib/ggml-cuda.dll",
        f"{root}/runtime/cublas64_{cuda_major}.dll",
        f"{root}/runtime/cublaslt64_{cuda_major}.dll",
        f"{root}/runtime/{cudart_name}",
    }
    assert required_paths.issubset(names)
    assert any(
        name.startswith(f"{root}/runtime/nvrtc") and name.endswith(".dll")
        for name in names
    )
    assert profile == {"profile_id": profile_id}
    assert not any("qwen3vl" in name or "mmproj" in name for name in names)
    if profile_id == "gtx-1050-ti":
        assert not any(
            Path(name).name in {"llava.dll", "llava.lib", "mtmd.dll", "mtmd.lib"}
            for name in names
        )

    forbidden_names = {
        ".env",
        "progress.json",
        "progress.log",
        "pdfslidetextextractor-folder.exe",
    }
    leaked = [name for name in names if Path(name).name in forbidden_names]
    assert leaked == []

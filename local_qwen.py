from __future__ import annotations

import ctypes
import hashlib
import inspect
import os
from pathlib import Path
import site
import sys
from typing import Any

from huggingface_hub import hf_hub_download


MODEL_REPO = "Qwen/Qwen3-8B-GGUF"
MODEL_REVISION = "4f02e7c52b572082828edf5058a87e2e7dc3e4d5"
MODEL_FILENAME = "Qwen3-8B-Q5_K_M.gguf"
MODEL_SIZE = 5_851_112_224
MODEL_SHA256 = "068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738"
DEFAULT_N_CTX = 8192
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_THREADS = 8
DEFAULT_N_THREADS_BATCH = 16
DEFAULT_N_BATCH = 512
DEFAULT_N_UBATCH = 512
_DLL_DIRECTORY_HANDLES: list[Any] = []
_CUDA_DLL_SEARCH_CONFIGURED = False


def _configure_cuda_dll_search() -> None:
    """Make CUDA DLLs installed by NVIDIA's Python wheels visible on Windows."""
    global _CUDA_DLL_SEARCH_CONFIGURED

    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    if _CUDA_DLL_SEARCH_CONFIGURED:
        return

    candidates: list[Path] = []
    bundled_runtime = getattr(sys, "_MEIPASS", None)
    if bundled_runtime:
        bundle_root = Path(bundled_runtime)
        candidates.extend(
            [
                bundle_root / "nvidia" / "cu13" / "bin" / "x86_64",
                bundle_root / "nvidia" / "cu12" / "bin",
                bundle_root,
            ]
        )
    for root in site.getsitepackages():
        site_root = Path(root)
        candidates.extend(
            [
                site_root / "nvidia" / "cu13" / "bin" / "x86_64",
                site_root / "nvidia" / "cu12" / "bin",
                site_root / "nvidia" / "cublas" / "bin",
                site_root / "nvidia" / "cuda_runtime" / "bin",
                site_root / "nvidia" / "cuda_nvrtc" / "bin",
            ]
        )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            handle = os.add_dll_directory(str(candidate))
        except OSError:
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        dll_paths = sorted(candidate.glob("cublasLt64_*.dll"))
        dll_paths.extend(sorted(candidate.glob("cublas64_*.dll")))
        for dll_path in dll_paths:
            try:
                dll_handle = ctypes.CDLL(str(dll_path), winmode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            _DLL_DIRECTORY_HANDLES.append(dll_handle)
    _CUDA_DLL_SEARCH_CONFIGURED = True


def cuda_gpu_available() -> bool:
    _configure_cuda_dll_search()
    try:
        from llama_cpp import llama_supports_gpu_offload
    except (ImportError, OSError, RuntimeError):
        return False
    return bool(llama_supports_gpu_offload())


def _verify_existing_model(path: Path, expected_size: int, expected_sha256: str) -> None:
    size = path.stat().st_size
    if size != expected_size:
        raise RuntimeError(
            f"local Qwen model has size {size}, expected {expected_size}: {path}"
        )

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "local Qwen model failed SHA-256 verification: "
            f"expected {expected_sha256}, received {actual_sha256}: {path}"
        )


def _ensure_model_file(
    model_dir: Path,
    filename: str,
    expected_size: int,
    expected_sha256: str,
    *,
    allow_download: bool,
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    target = model_dir / filename
    if target.is_file():
        _verify_existing_model(target, expected_size, expected_sha256)
        return target

    if not allow_download:
        raise FileNotFoundError(f"bundled Qwen model component not found: {target}")

    downloaded = hf_hub_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        filename=filename,
        local_dir=str(model_dir),
    )
    downloaded_path = Path(downloaded)
    _verify_existing_model(downloaded_path, expected_size, expected_sha256)
    return downloaded_path


def ensure_model(model_dir: Path, *, allow_download: bool = True) -> Path:
    """Return the verified, locked text-only Qwen3 model path."""
    return _ensure_model_file(
        model_dir,
        MODEL_FILENAME,
        MODEL_SIZE,
        MODEL_SHA256,
        allow_download=allow_download,
    )


class LocalQwenBackend:
    def __init__(
        self,
        model_path: Path,
        *,
        n_ctx: int = DEFAULT_N_CTX,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        n_threads: int = DEFAULT_N_THREADS,
        n_threads_batch: int = DEFAULT_N_THREADS_BATCH,
        n_batch: int = DEFAULT_N_BATCH,
        n_ubatch: int = DEFAULT_N_UBATCH,
        flash_attn: bool = True,
        offload_kqv: bool = True,
        op_offload: bool = True,
        require_gpu: bool = True,
        target_gpu: str = "NVIDIA GPU",
        cuda_runtime: str = "packaged",
        temperature: float = 0.2,
        seed: int = 42,
    ) -> None:
        _configure_cuda_dll_search()
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except (ImportError, OSError, RuntimeError) as exc:
            raise RuntimeError(
                "The CUDA llama.cpp runtime could not load. Install an NVIDIA "
                f"driver compatible with CUDA {cuda_runtime} for {target_gpu}."
            ) from exc

        if require_gpu and not llama_supports_gpu_offload():
            raise RuntimeError(
                f"CUDA GPU offload is unavailable. Confirm {target_gpu} is enabled "
                f"and its NVIDIA driver supports CUDA {cuda_runtime}."
            )

        llama_kwargs = {
            "model_path": str(model_path),
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "main_gpu": 0,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_threads": n_threads,
            "n_threads_batch": n_threads_batch,
            "offload_kqv": offload_kqv,
            "flash_attn": flash_attn,
            "use_mmap": True,
            "use_mlock": False,
            "seed": seed,
            "verbose": False,
        }
        llama_parameters = inspect.signature(Llama).parameters
        if "op_offload" in llama_parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in llama_parameters.values()
        ):
            llama_kwargs["op_offload"] = op_offload

        try:
            self._llm = Llama(**llama_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Could not initialize Qwen on CUDA. Confirm {target_gpu} is enabled "
                f"in Device Manager and its driver supports CUDA {cuda_runtime}."
            ) from exc
        if require_gpu:
            print(
                f"Local Qwen acceleration: {target_gpu}, CUDA {cuda_runtime}, "
                f"{n_gpu_layers} GPU layers, {n_ctx}-token context, "
                f"{n_batch}-token batch.",
                file=sys.stderr,
            )
        self._temperature = temperature
        self._seed = seed
        self._completion_index = 0

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

    def _complete_messages(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        call_seed = self._seed + self._completion_index
        self._completion_index += 1
        response: Any = self._llm.create_chat_completion(
            messages=messages,
            temperature=self._temperature,
            seed=call_seed,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen returned an unexpected response shape") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Qwen returned empty assistant content")
        return content.strip()

    def complete(self, system: str, user: str, *, max_tokens: int) -> str:
        return self._complete_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"{user}\n\n/no_think"},
            ],
            max_tokens,
        )

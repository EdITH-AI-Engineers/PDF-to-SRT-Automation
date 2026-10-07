from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class GpuProfile:
    profile_id: str
    target_gpu: str
    cuda_runtime: str
    n_ctx: int
    n_gpu_layers: int
    n_threads: int
    n_threads_batch: int
    n_batch: int
    n_ubatch: int
    flash_attn: bool
    offload_kqv: bool
    op_offload: bool


GPU_PROFILES = {
    "gtx-1050-ti": GpuProfile(
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
    "rtx-3060": GpuProfile(
        profile_id="rtx-3060",
        target_gpu="NVIDIA GeForce RTX 3060 12 GB",
        cuda_runtime="13.2",
        n_ctx=8192,
        n_gpu_layers=-1,
        n_threads=8,
        n_threads_batch=16,
        n_batch=512,
        n_ubatch=512,
        flash_attn=True,
        offload_kqv=True,
        op_offload=True,
    ),
}


def load_gpu_profile(base_dir: Path) -> GpuProfile:
    profile_path = Path(base_dir) / "gpu-profile.json"
    profile_id = "rtx-3060"
    if profile_path.is_file():
        data = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_id = str(data.get("profile_id", "")).strip()
    try:
        return GPU_PROFILES[profile_id]
    except KeyError as exc:
        supported = ", ".join(sorted(GPU_PROFILES))
        raise RuntimeError(
            f"Unsupported GPU profile {profile_id!r}; expected one of: {supported}"
        ) from exc

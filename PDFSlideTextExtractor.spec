# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import importlib.metadata
import os
import site

import paddlex
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)


excluded_llama_files = {
    "llava.dll",
    "llava.lib",
    "mtmd.dll",
    "mtmd.lib",
    "mtmd_shared.dll",
    "mtmd_shared.lib",
}
llama_binaries = [
    item
    for item in collect_dynamic_libs("llama_cpp")
    if Path(item[0]).name.casefold() not in excluded_llama_files
]
llama_data = [
    item
    for item in collect_data_files("llama_cpp")
    if Path(item[0]).name.casefold() not in excluded_llama_files
    and Path(item[0]).suffix.casefold() != ".lib"
]
paddle_binaries = collect_dynamic_libs("paddle")
paddle_data = collect_data_files("paddlex") + collect_data_files("paddleocr")
installed_distributions = {
    distribution.metadata["Name"]
    for distribution in importlib.metadata.distributions()
    if distribution.metadata["Name"]
}
paddle_metadata_names = {
    "paddleocr",
    "paddlepaddle",
    "paddlex",
    *(
        dependency
        for dependency in paddlex.utils.deps.BASE_DEP_SPECS
        if dependency in installed_distributions
    ),
}
paddle_metadata = [
    item
    for dependency in sorted(paddle_metadata_names)
    for item in copy_metadata(dependency)
]
cuda_major = os.environ.get("PDF_EXTRACTOR_CUDA_MAJOR", "13")
build_name = os.environ.get(
    "PDF_EXTRACTOR_BUILD_NAME", "PDFSlideTextExtractor-RTX3060"
)
site_root = Path(site.getsitepackages()[-1])
if cuda_major == "13":
    cuda_bin_dirs = [site_root / "nvidia" / "cu13" / "bin" / "x86_64"]
elif cuda_major in {"11", "12"}:
    cuda_bin_dirs = [
        site_root / "nvidia" / "cublas" / "bin",
        site_root / "nvidia" / "cuda_runtime" / "bin",
        site_root / "nvidia" / "cuda_nvrtc" / "bin",
    ]
elif cuda_major == "10":
    cuda_bin = Path(os.environ["PDF_EXTRACTOR_CUDA_BIN"])
    cuda_names = (
        "cublas64_10.dll",
        "cublasLt64_10.dll",
        "cudart64_102.dll",
        "nvrtc64_102_0.dll",
        "nvrtc-builtins64_102.dll",
    )
    missing_cuda = [name for name in cuda_names if not (cuda_bin / name).is_file()]
    if missing_cuda:
        raise RuntimeError(
            "Missing CUDA 10.2 runtime DLLs: " + ", ".join(missing_cuda)
        )
    cuda_binaries = [(str(cuda_bin / name), ".") for name in cuda_names]
    cuda_bin_dirs = []
else:
    raise ValueError(f"Unsupported CUDA major version: {cuda_major}")

if cuda_major != "10":
    cuda_binaries = [
        (str(path), ".")
        for directory in cuda_bin_dirs
        for path in sorted(directory.glob("*.dll"))
    ]
if not cuda_binaries:
    raise RuntimeError(
        f"No CUDA {cuda_major} runtime DLLs found beneath {site_root / 'nvidia'}"
    )

a = Analysis(
    ["run_api.py"],
    pathex=[],
    binaries=llama_binaries + paddle_binaries + cuda_binaries,
    datas=llama_data + paddle_data + paddle_metadata,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PDFSlideTextExtractor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    contents_directory="runtime",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=build_name,
)

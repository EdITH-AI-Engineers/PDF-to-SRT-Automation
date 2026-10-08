# -*- mode: python ; coding: utf-8 -*-

import importlib.metadata

import paddlex
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    copy_metadata,
)


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
a = Analysis(
    ["run_folder_queue.py"],
    pathex=[],
    binaries=paddle_binaries,
    datas=paddle_data + paddle_metadata,
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
    name="PDFSlideTextExtractor-OCR",
)

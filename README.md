# PDF Slide Text Extractor

The extractor reads PDFs from the `input` folder and writes text files to the
`output` folder. It does not start an HTTP server or open a network port.
Extraction, PaddleOCR, and Qwen formatting all run locally. Two portable GPU
builds are available:

| Build | CUDA runtime | Qwen execution |
| --- | --- | --- |
| `PDFSlideTextExtractor-GTX1050Ti.zip` | CUDA 10.2 | 16 GPU layers, CPU-hosted KV cache, 128/64 logical/physical batches |
| `PDFSlideTextExtractor-RTX3060.zip` | CUDA 13.2 | Full GPU layer/KV/operation offload, flash attention, 512-token batches |

Both builds use the same locked Qwen3-8B Q5_K_M model, lightweight English
PP-OCRv5 mobile detection/recognition models, 8192-token context, prompts, OCR
filtering, and deterministic generation settings. The 1050 Ti
build preserves output quality but is slower because its 4 GB VRAM cannot hold
the complete 5.85 GB model. It uses system RAM for the remaining model layers
and KV cache.

The app uses the locked text-only Qwen3-8B Q5_K_M model. PyMuPDF reads reliable
embedded text, while PaddleOCR handles scanned pages or pages whose embedded
text looks corrupted. Every page gets one PaddleOCR pass so equations can be
detected, while OCR body text is used only when embedded text is missing or
corrupt. Pages are processed in bounded batches of up to eight, and the
recognized lines are reused for equation detection instead of running a second
OCR pass. The output lists equations separately as `Equation 1`, `Equation 2`,
and so on. Repeated gibberish lines are removed. The app does not send slide
images to Qwen and does not invent image or diagram descriptions.

PaddleOCR runs on the CPU in both packages. Current PaddlePaddle GPU builds do
not support the GTX 1050 Ti's Pascal compute capability, so CPU OCR keeps both
packages compatible while Qwen continues to use the selected NVIDIA GPU.

Each portable package includes its matching CUDA runtime DLLs and PaddleOCR
models, so neither a separate CUDA Toolkit nor a first-run model download is
needed. An NVIDIA display driver is still required. Use the
GTX 1050 Ti build for its bundled CUDA 10.2 runtime built for the Pascal
architecture. The RTX 3060 build requires a driver compatible with CUDA 13.2.
The app reports an actionable error rather than silently reverting to CPU when
CUDA is unavailable.

## Building both portable packages

The build script creates isolated Python environments so CUDA 10.2 and CUDA
13.2 libraries are never mixed. The GTX build installs the locally compiled
`llama-cpp-python` CUDA 10.2 wheel from `.build\cuda102-toolchain\wheels`:

```powershell
.\build_portables.ps1
```

Build only one profile with `-Profile GTX1050Ti` or `-Profile RTX3060`. The
resulting folders and ZIP archives are written to `dist`.

## Automatic folder queue

While the folder watcher is running, it continuously watches `input` and
processes one folder at a time. You can keep copying module folders into
`input`; output keeps the same relative folder structure:

```text
input\CS101\Module 1\slides.pdf  ->  output\CS101\Module 1\slides.txt
input\CS101\Module 2\slides.pdf  ->  output\CS101\Module 2\slides.txt
```

A folder is queued after its PDF filenames, sizes, and modification times have
remained unchanged for five seconds. This prevents partially copied PDFs from
being opened. Successfully processed input PDFs are deleted. A failed PDF stays
in `input` and is not retried repeatedly until it changes or the watcher is
restarted.

## Running the folder watcher

Install the dependencies and start the watcher:

```powershell
python -m pip install -r requirements.txt
python .\run_folder_queue.py
```

For a portable build, extract the package and run
`PDFSlideTextExtractor.exe`. Leave the program open while copying PDFs or
module folders into `input`. Results appear under the matching path in
`output`:

```text
input\module.pdf                    -> output\module.txt
input\CS101\Module 1\slides.pdf    -> output\CS101\Module 1\slides.txt
```

Press `Ctrl+C` to stop the watcher cleanly. No browser, API, or port is used.

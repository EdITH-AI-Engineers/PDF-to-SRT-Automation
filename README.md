# PDF Slide Text Extractor

The extractor can accept PDFs over its local HTTP API on port `8001` or read
them directly from the `input` folder. Extraction, PaddleOCR, and Qwen
formatting all run locally. Two portable GPU builds are available:

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
CUDA is unavailable. `GET /health` reports the selected profile, target GPU,
CUDA runtime, and GPU-layer setting.

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

While the API is running, it continuously watches `input` and processes one
folder at a time. You can keep copying module folders into `input`; output keeps
the same relative folder structure:

```text
input\CS101\Module 1\slides.pdf  ->  output\CS101\Module 1\slides.txt
input\CS101\Module 2\slides.pdf  ->  output\CS101\Module 2\slides.txt
```

A folder is queued after its PDF filenames, sizes, and modification times have
remained unchanged for five seconds. This prevents partially copied PDFs from
being opened. Successfully processed input PDFs are deleted. A failed PDF stays
in `input` and is not retried repeatedly until it changes or `POST /queue/retry`
is called.

## PDF input API

Install the dependencies and start the API:

```powershell
python -m pip install -r requirements.txt
python .\run_api.py
```

The upload page and API are available at <http://127.0.0.1:8001>. Upload one or
more PDF files as multipart field `files`:

```powershell
curl.exe -F "files=@C:\path\module.pdf" http://127.0.0.1:8001/process
```

To return immediately and let the background queue process the PDFs, use the
queued upload endpoint. `folder_name` is optional:

```powershell
curl.exe -F "folder_name=Module 7" -F "files=@C:\path\part-1.pdf" -F "files=@C:\path\part-2.pdf" http://127.0.0.1:8001/queue/upload
```

The response includes a `job_id` and `status_url`. Open that URL to see whether
the folder is queued, processing, completed, or failed.

An optional `course_code` form field stores inputs under
`input\<course_code>` and results under `output\<course_code>`. You may also use
`POST /process/<course_code>`.

Other endpoints:

```text
GET  /health
GET  /queue
GET  /queue/{job_id}
POST /queue/upload
POST /queue/retry
POST /process-existing
GET  /outputs
GET  /download/{filename}
```

## One-time folder input

The API queue is recommended. For a single command-line batch instead, put PDFs
in `input`, then run:

```powershell
python .\slides_pdf_to_txt.py
```

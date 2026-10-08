# PDF Slide Text Extractor — OCR Only

The extractor watches the `input` folder, reads slide PDFs locally, and writes
UTF-8 text files to the matching path under `output`. It does not use an LLM,
download a Qwen/GGUF model, start an HTTP server, or open a network port.

PyMuPDF is used first for reliable embedded PDF text. PP-OCRv6-small handles
scanned pages, and PP-OCRv6-medium retries only empty or low-confidence OCR
results. The output is
deterministic: the first readable line becomes the slide title, the remaining
recognized lines become the content, locally detected equations are listed
separately, and the output contains no generated brief-explanation section.

## Automatic folder queue

Keep the watcher open and copy PDFs or complete module folders into `input`:

```text
input\module.pdf                  -> output\module.txt
input\CS101\Module 1\slides.pdf  -> output\CS101\Module 1\slides.txt
```

A folder enters the queue after its PDF names, sizes, and modification times
have stayed unchanged for five seconds. Folders are processed one at a time.
Successfully processed PDFs are deleted from `input`; failed PDFs remain in
place and are recorded in `progress.json` and `progress.log`.

## Running from source

```powershell
python -m pip install -r requirements.txt
python .\run_folder_queue.py
```

Press `Ctrl+C` to stop cleanly.

## OCR behavior

The bundled default is PP-OCRv6-small at 200 DPI. PP-OCRv6-medium is loaded
lazily and receives only pages whose small-model result is empty or below the
0.82 confidence threshold. If both models return text, the higher-confidence
result is kept.

OCR is deliberately CPU-only: no CUDA, GPU runtime, oneDNN, server, or open
port is required. Pages are handled in bounded batches of eight. Image decoding
and color conversion run in parallel across up to four workers, while PaddleOCR
uses up to eight CPU threads. Recognized lines are reused for local equation
detection.

## Deployment systems

There are two current target systems:

1. **GTX 1050 Ti 4 GB system:** Windows 11 Pro 64-bit, Intel Core i7-10700,
   and 16 GB RAM. OCR runs on the CPU because the GPU's Pascal compute
   capability 6.1 is below the greater-than-7.5 requirement of the current
   PaddlePaddle Windows GPU package.
2. **RTX 3060 Ti 12 GB system:** intended for a separate GPU-enabled package.
   Do not replace the CPU package with GPU-only dependencies; both system
   variants must remain usable.

The current portable is the CPU-compatible variant and therefore works on both
systems without CUDA. A future GPU package may accelerate the RTX system while
retaining automatic CPU fallback.

## Building the portable package

The build contains PaddleOCR and its verified local model files, so normal use
does not download models on first run:

```powershell
.\build_portables.ps1
```

The output is written to:

```text
dist\PDFSlideTextExtractor-OCR\
dist\PDFSlideTextExtractor-OCR.zip
```

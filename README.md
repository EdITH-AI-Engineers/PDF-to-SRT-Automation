# PDF Slide Text Extractor — OCR Only

The extractor watches the `input` folder, reads slide PDFs locally, and writes
UTF-8 text files to the matching path under `output`. It does not use an LLM,
download a Qwen/GGUF model, start an HTTP server, or open a network port.

PyMuPDF is used first for reliable embedded PDF text. PaddleOCR handles scanned
pages and pages whose embedded text is missing or corrupt. The output is
deterministic: the first readable line becomes the slide title, the remaining
recognized lines become the content, locally detected equations are listed
separately, and fields that would require interpretation are written as
`Not Specified`.

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

The bundled default is the English PP-OCRv5 mobile detector and recognizer at
200 DPI. OCR runs on the CPU for broad Windows compatibility. Pages are handled
in bounded batches of eight, and recognized lines are reused for local equation
detection.

For faster conversion, the next optimization is to skip rasterization and OCR
when a page already has trustworthy embedded text. For higher recognition
accuracy, use a larger PaddleOCR model or a document-structure pipeline; these
increase package size and resource usage.

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

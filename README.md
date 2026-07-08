# PDF Slide Extraction With Mistral

This folder contains a standalone Python script that processes PowerPoint-slide PDFs from `input` with the Mistral API and creates one UTF-8 `.txt` file per PDF in `output`.

## Quick Start: Run The Windows Executable (No Python Required)

The easiest way to run this is with the pre-built executable that doesn't require Python:

The built executable is:

```text
dist\PDFSlideExtractionAPI.exe
```

### Setup

1. Create the exe `.env` file:

```powershell
Copy-Item .env.example .\dist\.env
```

2. Add your Mistral API key to `dist\.env`:
   - Go to https://console.mistral.ai/
   - Sign in or create an account
   - Open the API keys area and create a new API key
   - Open `dist\.env` and replace the placeholder with your real key

3. Create the required directories in `dist`:

```powershell
mkdir .\dist\input
mkdir .\dist\output
```

4. Put your PDF files in `dist\input\`

5. Run the executable:

```powershell
.\dist\PDFSlideExtractionAPI.exe
```

6. Open your browser to `http://127.0.0.1:8000`

If Windows blocks the executable with an Application Control policy, contact your admin to allowlist it, or use the Python version below instead.

---

## Alternative: Run With Python (Requires Python Installation)

### 1. Create a Mistral API Key

1. Go to https://console.mistral.ai/
2. Sign in or create an account.
3. Open the API keys area.
4. Create a new API key.
5. Copy the key once. Treat it like a password.

### 2. Install Python Packages

From this folder, run:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and reopen PowerShell.

### 3. Add the API Key

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder:

```text
MISTRAL_API_KEY=your_real_key_here
```

Do not share or upload `.env`.

### 4. Run The Local FastAPI App

Start the local API on port `8000`:

```powershell
python .\run_api.py
```

Open this URL:

```text
http://127.0.0.1:8000
```

**Useful API endpoints:**

```text
GET  /health
POST /process
POST /process/{course_code}
POST /process-existing
POST /process-existing/{course_code}
GET  /outputs
GET  /outputs/{course_code}
GET  /download/{filename}
GET  /download/{course_code}/{filename}
```

For course batches, put PDFs in a course folder:

```text
input\CS101\Module1.pdf
input\CS101\Module2.pdf
```

Then call:

```text
POST /process-existing/CS101
```

The API writes the batch to:

```text
output\CS101\Module1.txt
output\CS101\Module2.txt
```

The upload endpoint also accepts a course code, either as the URL path
`POST /process/CS101` or as the `course_code` form field on `POST /process`.

### 5. Process All PDFs In The Input Folder

Run:

```powershell
python .\slides_pdf_to_txt.py
```

The script scans `input` for `.pdf` files and writes one `.txt` per PDF into:

```text
output
```

For example:

```text
input\Module1.pdf -> output\Module1.txt
```

### 6. Process Specific PDFs

Run:

```powershell
python .\slides_pdf_to_txt.py .\input\Module1.pdf .\input\Module2.pdf
```

### 7. Overwrite Existing Text Files

Run:

```powershell
python .\slides_pdf_to_txt.py --overwrite
```

Without `--overwrite`, the script avoids replacing existing files by creating names like `Module1_2.txt`.

### 8. If You Hit Mistral Rate Limits

If you see `Status 429` or `Rate limit exceeded`, rerun with a longer wait:

```powershell
python .\slides_pdf_to_txt.py --format-model mistral-large-2512 --slide-max-tokens 6000 --metadata-max-tokens 800 --rate-limit-wait 90 --request-delay 5 --attempts 8
```

If `mistral-large-2512` still fails from rate limits, use a cheaper fallback:

```powershell
python .\slides_pdf_to_txt.py --format-model mistral-small-2603 --slide-max-tokens 6000 --metadata-max-tokens 800 --rate-limit-wait 90 --request-delay 5 --attempts 8
```

## What The Script Does

- Processes every PDF independently.
- Creates one UTF-8 `.txt` file per PDF.
- Uses the original PDF filename for the output filename whenever possible.
- Runs Mistral OCR on each PDF.
- Uses mistral-large-2512 as the default formatting model for stronger slide handling.
- Treats each PDF page as one slide.
- Preserves slide order.
- Adds a strict 2-3 sentence brief explanation after every slide.
- Deletes uploaded Mistral files after processing unless `--keep-uploads` is used.

## Rebuilding The Executable

To rebuild the executable after code changes:

```powershell
python -m pip install pyinstaller
python -m PyInstaller --noconfirm --clean .\PDFSlideExtractionAPI.spec
```

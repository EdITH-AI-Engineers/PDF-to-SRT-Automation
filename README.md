# PDF Slide Extraction With Mistral

This folder contains a standalone Python script that processes PowerPoint-slide PDFs from `input` with the Mistral API and creates one UTF-8 `.txt` file per PDF in `output`.

## 1. Install Python Packages

From this folder, run:

```powershell
python -m pip install -r requirements.txt
```

If `python` is not recognized, install Python from https://www.python.org/downloads/ and reopen PowerShell.

## 2. Create a Mistral API Key

1. Go to https://console.mistral.ai/
2. Sign in or create an account.
3. Open the API keys area.
4. Create a new API key.
5. Copy the key once. Treat it like a password.

## 3. Add the API Key

Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

Open `.env` and replace the placeholder:

```text
MISTRAL_API_KEY=your_real_key_here
```

Do not share or upload `.env`.

## 4. Process All PDFs In The Input Folder

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

## 5. Process Specific PDFs

Run:

```powershell
python .\slides_pdf_to_txt.py .\input\Module1.pdf .\input\Module2.pdf
```

## 6. Overwrite Existing Text Files

Run:

```powershell
python .\slides_pdf_to_txt.py --overwrite
```

Without `--overwrite`, the script avoids replacing existing files by creating names like `Module1_2.txt`.


## 7. If You Hit Mistral Rate Limits

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





## Run The Local FastAPI App

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

Start the local API on port `8000`:

```powershell
python .\run_api.py
```

Open this URL:

```text
http://127.0.0.1:8000
```

Useful API endpoints:

```text
GET  /health
POST /process
POST /process-existing
GET  /outputs
GET  /download/{filename}
```

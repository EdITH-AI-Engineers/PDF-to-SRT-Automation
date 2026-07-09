PDF Slide Text Extractor - Portable Windows Build

This package does not require Python to be installed.

Setup:
1. Copy .env.example to .env in this same folder.
2. Open .env and replace the placeholder with your real MISTRAL_API_KEY.
3. Put PDFs in input, or in course folders such as input\CS101.
4. Run PDFSlideTextExtractor.exe.
5. Wait for the extraction to finish, then press Enter to close the window.

Mistral API key steps:
1. Go to https://console.mistral.ai/
2. Sign in or create an account.
3. Open the API keys area.
4. Create a new API key.
5. Copy .env.example and rename the copy to .env.
6. Open .env and replace your_real_key_here with the key you copied.
7. Save .env in the same folder as PDFSlideTextExtractor.exe.

Do not share .env because it contains your private API key.

PDFSlideTextExtractorCore.exe is used internally by PDFSlideTextExtractor.exe.
Run PDFSlideTextExtractor.exe, not the core file.

Progress is saved while the extractor runs:
progress.json shows the latest status, current PDF, current slide, and outputs.
progress.log keeps a timestamped text log of the run.

After a PDF is successfully converted, the finished PDF is deleted from input.
This prevents the same finished PDF from being processed again on the next run.

Course batch example:
input\CS101\Module1.pdf
input\CS101\Module2.pdf

Outputs will be created in:
output\CS101

If Windows blocks PDFSlideTextExtractor.exe, your device has a policy for
unsigned executables. The exe will need to be allowed by IT or signed with an
approved certificate.

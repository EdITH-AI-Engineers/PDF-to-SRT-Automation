PDF Slide Text Extractor - Portable Windows Build

This package does not require Python to be installed.

Setup:
1. Copy .env.example to .env in this same folder.
2. Open .env and replace the placeholder with your real MISTRAL_API_KEY.
3. Put PDFs in input, or in course folders such as input\CS101.
4. Double-click start_extraction.bat, or run PDFSlideTextExtractor.exe directly.
5. Wait for the extraction to finish.

Course batch example:
input\CS101\Module1.pdf
input\CS101\Module2.pdf

Outputs will be created in:
output\CS101

If Windows blocks PDFSlideTextExtractor.exe, your device has a policy for
unsigned executables. The exe will need to be allowed by IT or signed with an
approved certificate.

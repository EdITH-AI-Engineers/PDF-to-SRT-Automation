param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$buildRoot = Join-Path $projectRoot ".build\ocr-only"
$distRoot = Join-Path $projectRoot "dist"
$paddleModelRoot = Join-Path $buildRoot "paddle-models"
$venv = Join-Path $buildRoot "venv"
$venvPython = Join-Path $venv "Scripts\python.exe"
$buildName = "PDFSlideTextExtractor-OCR"

if (-not (Test-Path -LiteralPath $venvPython)) {
    & $Python -m venv $venv
}
& $venvPython -c "import pip" 2>$null
if ($LASTEXITCODE -ne 0) {
    & $venvPython -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) {
        throw "Could not bootstrap pip for the OCR-only build"
    }
}

& $venvPython -m pip install --disable-pip-version-check `
    -r (Join-Path $projectRoot "requirements-dev.txt") `
    -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed for the OCR-only build"
}

& $venvPython (Join-Path $projectRoot "prepare_paddle_models.py") $paddleModelRoot
if ($LASTEXITCODE -ne 0) {
    throw "PaddleOCR model preparation failed"
}

$pyinstallerDist = Join-Path $buildRoot "pyinstaller-dist"
$pyinstallerWork = Join-Path $buildRoot "pyinstaller-work"
$packageDir = Join-Path $distRoot $buildName
$archivePath = Join-Path $distRoot ($buildName + ".zip")
$stagingRoot = Join-Path $buildRoot "portable-staging"
$stagedPackageDir = Join-Path $stagingRoot $buildName
$stagedArchivePath = Join-Path $stagingRoot ($buildName + ".zip")

foreach ($target in @($pyinstallerDist, $pyinstallerWork, $stagingRoot)) {
    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null

& $venvPython -m PyInstaller --noconfirm --clean `
    --distpath $pyinstallerDist `
    --workpath $pyinstallerWork `
    (Join-Path $projectRoot "PDFSlideTextExtractor.spec")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed for the OCR-only build"
}

Move-Item -LiteralPath (Join-Path $pyinstallerDist $buildName) `
    -Destination $stagedPackageDir
New-Item -ItemType Directory -Force -Path `
    (Join-Path $stagedPackageDir "input"), `
    (Join-Path $stagedPackageDir "output"), `
    (Join-Path $stagedPackageDir "models") | Out-Null
Copy-Item -LiteralPath $paddleModelRoot `
    -Destination (Join-Path $stagedPackageDir "models\paddleocr") -Recurse
Copy-Item -LiteralPath (Join-Path $projectRoot "README.md") `
    -Destination (Join-Path $stagedPackageDir "README.txt")

& $venvPython (Join-Path $projectRoot "package_portable.py") `
    $stagedPackageDir $stagedArchivePath
if ($LASTEXITCODE -ne 0) {
    throw "Archive creation failed for the OCR-only build"
}

if (Test-Path -LiteralPath $packageDir) {
    Remove-Item -LiteralPath $packageDir -Recurse -Force
}
if (Test-Path -LiteralPath $archivePath) {
    Remove-Item -LiteralPath $archivePath -Force
}
Move-Item -LiteralPath $stagedPackageDir -Destination $packageDir
Move-Item -LiteralPath $stagedArchivePath -Destination $archivePath

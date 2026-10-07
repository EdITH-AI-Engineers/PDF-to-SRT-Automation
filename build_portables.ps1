param(
    [ValidateSet("All", "GTX1050Ti", "RTX3060")]
    [string]$Profile = "All",
    [string]$Python = "python",
    [string]$ModelPath = ".\models\Qwen3-8B-Q5_K_M.gguf"
)

$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$model = (Resolve-Path -LiteralPath (Join-Path $projectRoot $ModelPath)).Path
$buildRoot = Join-Path $projectRoot ".build"
$distRoot = Join-Path $projectRoot "dist"
$paddleModelRoot = Join-Path $buildRoot "paddle-models"

$profiles = @(
    @{
        Selector = "GTX1050Ti"
        Id = "gtx-1050-ti"
        Name = "PDFSlideTextExtractor-GTX1050Ti"
        CudaMajor = "10"
        VenvName = "venv-cu102"
        Requirements = "requirements-gtx1050ti.txt"
        WheelPattern = ".build\cuda102-toolchain\wheels\llama_cpp_python-0.3.9-*.whl"
        CudaBin = ".build\cuda102-toolchain\root\bin"
    },
    @{
        Selector = "RTX3060"
        Id = "rtx-3060"
        Name = "PDFSlideTextExtractor-RTX3060"
        CudaMajor = "13"
        VenvName = "venv"
        Requirements = "requirements-rtx3060.txt"
    }
)

foreach ($item in $profiles) {
    if ($Profile -ne "All" -and $Profile -ne $item.Selector) {
        continue
    }

    $profileBuildRoot = Join-Path $buildRoot $item.Id
    $venv = Join-Path $profileBuildRoot $item.VenvName
    $venvPython = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $venvPython)) {
        & $Python -m venv $venv
    }
    & $venvPython -c "import pip" 2>$null
    if ($LASTEXITCODE -ne 0) {
        & $venvPython -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) {
            throw "Could not bootstrap pip for $($item.Name)"
        }
    }

    & $venvPython -m pip install --disable-pip-version-check `
        -r (Join-Path $projectRoot "requirements-dev.txt") `
        -r (Join-Path $projectRoot $item.Requirements)
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed for $($item.Name)"
    }
    if ($item.WheelPattern) {
        $wheelMatches = @(Get-ChildItem -Path (Join-Path $projectRoot $item.WheelPattern))
        if ($wheelMatches.Count -ne 1) {
            throw "Expected exactly one CUDA 10.2 llama-cpp wheel at $($item.WheelPattern)"
        }
        & $venvPython -m pip install --disable-pip-version-check --force-reinstall `
            $wheelMatches[0].FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Custom CUDA 10.2 wheel installation failed for $($item.Name)"
        }
    }
    & $venvPython (Join-Path $projectRoot "prepare_paddle_models.py") `
        $paddleModelRoot
    if ($LASTEXITCODE -ne 0) {
        throw "PaddleOCR model preparation failed for $($item.Name)"
    }

    $pyinstallerDist = Join-Path $profileBuildRoot "pyinstaller-dist"
    $pyinstallerWork = Join-Path $profileBuildRoot "pyinstaller-work"
    $packageDir = Join-Path $distRoot $item.Name
    $archivePath = Join-Path $distRoot ($item.Name + ".zip")
    $stagingRoot = Join-Path $profileBuildRoot "portable-staging"
    $stagedPackageDir = Join-Path $stagingRoot $item.Name
    $stagedArchivePath = Join-Path $stagingRoot ($item.Name + ".zip")

    foreach ($target in @($pyinstallerDist, $pyinstallerWork, $stagingRoot)) {
        if (Test-Path -LiteralPath $target) {
            Remove-Item -LiteralPath $target -Recurse -Force
        }
    }
    New-Item -ItemType Directory -Force -Path $stagingRoot | Out-Null

    $env:PDF_EXTRACTOR_CUDA_MAJOR = $item.CudaMajor
    $env:PDF_EXTRACTOR_BUILD_NAME = $item.Name
    if ($item.CudaBin) {
        $env:PDF_EXTRACTOR_CUDA_BIN = (
            Resolve-Path -LiteralPath (Join-Path $projectRoot $item.CudaBin)
        ).Path
    }
    try {
        & $venvPython -m PyInstaller --noconfirm --clean `
            --distpath $pyinstallerDist `
            --workpath $pyinstallerWork `
            (Join-Path $projectRoot "PDFSlideTextExtractor.spec")
    }
    finally {
        Remove-Item Env:PDF_EXTRACTOR_CUDA_MAJOR -ErrorAction SilentlyContinue
        Remove-Item Env:PDF_EXTRACTOR_BUILD_NAME -ErrorAction SilentlyContinue
        Remove-Item Env:PDF_EXTRACTOR_CUDA_BIN -ErrorAction SilentlyContinue
    }
    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed for $($item.Name)"
    }

    Move-Item -LiteralPath (Join-Path $pyinstallerDist $item.Name) `
        -Destination $stagedPackageDir
    New-Item -ItemType Directory -Force -Path `
        (Join-Path $stagedPackageDir "input"), `
        (Join-Path $stagedPackageDir "output"), `
        (Join-Path $stagedPackageDir "models") | Out-Null
    Copy-Item -LiteralPath $model `
        -Destination (Join-Path $stagedPackageDir "models\Qwen3-8B-Q5_K_M.gguf")
    Copy-Item -LiteralPath $paddleModelRoot `
        -Destination (Join-Path $stagedPackageDir "models\paddleocr") -Recurse
    Copy-Item -LiteralPath (Join-Path $projectRoot "model-lock.json") `
        -Destination $stagedPackageDir
    Copy-Item -LiteralPath (Join-Path $projectRoot "profiles\$($item.Id).json") `
        -Destination (Join-Path $stagedPackageDir "gpu-profile.json")
    Copy-Item -LiteralPath (Join-Path $projectRoot "README.md") `
        -Destination (Join-Path $stagedPackageDir "README.txt")

    & $venvPython (Join-Path $projectRoot "package_portable.py") `
        $stagedPackageDir $stagedArchivePath
    if ($LASTEXITCODE -ne 0) {
        throw "Archive creation failed for $($item.Name)"
    }

    if (Test-Path -LiteralPath $packageDir) {
        Remove-Item -LiteralPath $packageDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }
    Move-Item -LiteralPath $stagedPackageDir -Destination $packageDir
    Move-Item -LiteralPath $stagedArchivePath -Destination $archivePath
}

# Builds dist\ProjectR-Setup.exe on this PC: the setup file to give friends.
#
#   VS Code:  Terminal > Run Task... > "Build setup file"
#   or:       powershell -ExecutionPolicy Bypass -File installer\build.ps1
#
#   ... -Offline          everything inside: friends download nothing after it
#                         (Python, PyTorch with CUDA, the offline voice). About
#                         3 GB, so it is a folder: dist\ProjectR-Offline\ with
#                         the Setup .exe and its .bin parts; give the whole folder.
#   ... -Offline -Torch cpu   the same without CUDA: about 600 MB, CPU only.
#
# The first time it fetches what it needs into the project folder (nothing
# onto C:): uv.exe (about 20 MB, put into the setup file) and Inno Setup
# (about 10 MB, the tool that makes the .exe) in .cache\tools. -Offline also
# downloads Python and the libraries once into .cache\stage (3 to 5 GB).
param(
    [string]$Version = (Get-Date -Format "yyyy.M.d"),
    [switch]$Offline,
    [ValidateSet("cuda", "cpu")][string]$Torch = "cuda"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root ".cache\tools"
New-Item -ItemType Directory -Force -Path $tools, (Join-Path $PSScriptRoot "bin") | Out-Null
$env:TEMP = $env:TMP = Join-Path $root ".cache\tmp"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

# 1. uv: it installs Python and the libraries on the friend's PC.
$UvVersion = "0.11.32"
$uv = Join-Path $PSScriptRoot "bin\uv.exe"
if (-not (Test-Path $uv)) {
    Write-Host "Fetching uv $UvVersion..."
    $zip = Join-Path $env:TEMP "uv.zip"
    Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip
    Expand-Archive $zip -DestinationPath (Join-Path $env:TEMP "uv") -Force
    Copy-Item (Join-Path $env:TEMP "uv\uv.exe") $uv
}

# 2. Inno Setup: already installed, else a private copy in .cache\tools.
$iscc = @(
    (Join-Path $tools "InnoSetup\ISCC.exe"),
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $iscc) {
    Write-Host "Fetching Inno Setup into $tools\InnoSetup ..."
    $setup = Join-Path $env:TEMP "innosetup.exe"
    Invoke-WebRequest "https://jrsoftware.org/download.php/is.exe" -OutFile $setup
    $dir = Join-Path $tools "InnoSetup"
    Start-Process $setup -Wait -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CURRENTUSER", "/NOICONS", "/DIR=`"$dir`""
    $iscc = Join-Path $dir "ISCC.exe"
    if (-not (Test-Path $iscc)) { throw "Inno Setup did not install into $dir." }
}

# 3. -Offline: Python with every library and the offline voice, ready to copy.
$defines = @("/DMyVersion=$Version")
if ($Offline) {
    $stage = Join-Path $root ".cache\stage"
    $runtime = Join-Path $stage "python-$Torch"
    $env:UV_CACHE_DIR = Join-Path $root ".cache\uv"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $stage "pythons"
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $env:UV_PYTHON_BIN_DIR = Join-Path $stage "bin"   # not C:\Users\...\.local\bin
    Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue  # an active .venv is not the Python to pack
    if (-not (Test-Path (Join-Path $runtime "Lib\site.py"))) {
        Write-Host "Python 3.11 for the setup file..."
        & $uv python install 3.11
        if ($LASTEXITCODE -ne 0) { throw "uv python install failed." }
        # The full Python that uv installed (not a venv's python.exe).
        $home3 = Get-ChildItem (Join-Path $stage "pythons") -Directory -Filter "cpython-3.11*" |
            Where-Object { Test-Path (Join-Path $_.FullName "Lib\site.py") } | Select-Object -First 1
        if (-not $home3) { throw "Python 3.11 not found in $stage\pythons after installing it." }
        if (Test-Path $runtime) { throw "$runtime is there but incomplete: delete that folder, then run me again." }
        Copy-Item $home3.FullName $runtime -Recurse
    }
    $py = Join-Path $runtime "python.exe"
    $index = if ($Torch -eq "cuda") { "https://download.pytorch.org/whl/cu132" } else { "https://download.pytorch.org/whl/cpu" }
    $tag = if ($Torch -eq "cuda") { "+cu132" } else { "" }
    Write-Host "PyTorch ($Torch) and the app's libraries into $runtime ..."
    & $uv pip install --python $py --break-system-packages "torch==2.13.0$tag" "torchvision==0.28.0$tag" --index-url $index
    if ($LASTEXITCODE -ne 0) { throw "Installing PyTorch failed." }
    # The project's one requirements.txt, without the torch lines (installed just above).
    $list = Join-Path $env:TEMP "requirements-setup.txt"
    Get-Content (Join-Path $root "requirements.txt") | Where-Object { $_ -notmatch '^\s*(torch|torchvision)==|extra-index-url' } | Set-Content $list
    & $uv pip install --python $py --break-system-packages -r $list
    if ($LASTEXITCODE -ne 0) { throw "Installing the libraries failed." }
    & $uv pip install --python $py --break-system-packages --no-deps easyocr boxmot==25.0.0
    if ($LASTEXITCODE -ne 0) { throw "Installing easyocr and boxmot failed." }
    # The offline Bengali voice (Meta MMS, about 150 MB), so no first-use download.
    $voice = Join-Path $stage "huggingface"
    $env:HF_HOME = $voice
    & $py -c "from transformers import AutoTokenizer, VitsModel; AutoTokenizer.from_pretrained('facebook/mms-tts-ben'); VitsModel.from_pretrained('facebook/mms-tts-ben')"
    if ($LASTEXITCODE -ne 0) { throw "Downloading the offline voice failed." }
    Remove-Item Env:HF_HOME
    $defines += "/DOffline=1", "/DRuntime=$runtime", "/DVoice=$voice", "/DTorch=$Torch"
}

# 4. The setup file.
Write-Host "Building Project R $Version ..."
& $iscc @defines (Join-Path $PSScriptRoot "ProjectR.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed (exit code $LASTEXITCODE)." }
Write-Host ""
if ($Offline) {
    $folder = Join-Path $root "dist\ProjectR-Offline$(if ($Torch -eq 'cpu') { '-CPU' })"
    $size = (Get-ChildItem $folder | Measure-Object Length -Sum).Sum
    Write-Host "Done: $folder ($([math]::Round($size / 1GB, 1)) GB)" -ForegroundColor Green
    Write-Host "Give friends the whole folder (the .exe needs its .bin parts next to it), e.g. zipped or"
    Write-Host "on a USB stick. They run ProjectR-Setup.exe in it; nothing is downloaded after that."
} else {
    $out = Join-Path $root "dist\ProjectR-Setup.exe"
    Write-Host "Done: $out ($([math]::Round((Get-Item $out).Length / 1MB)) MB)" -ForegroundColor Green
    Write-Host "Give this one file to friends. It downloads Python and the libraries while installing."
}
Write-Host "Windows may say 'Windows protected your PC' (the file is not signed): 'More info' > 'Run anyway'."

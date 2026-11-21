# Project R: sets up Python and the AI libraries inside the app folder.
# The setup file runs this once after copying the files; the Start menu entry
# "Repair Project R" runs it again. Everything it downloads (Python, PyTorch,
# the rest, their caches) goes into the app folder, nothing onto C: unless the
# app itself is installed there.
#
#   powershell -ExecutionPolicy Bypass -File installer\setup_runtime.ps1 -App <folder> [-Torch auto|cuda|cpu]
param(
    [string]$App = (Split-Path -Parent $PSScriptRoot),
    [ValidateSet("auto", "cuda", "cpu")][string]$Torch = "auto"
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Host.UI.RawUI.WindowTitle = "Project R set-up"

$App = (Resolve-Path $App).Path
$uv = Join-Path $App "installer\bin\uv.exe"
$venv = Join-Path $App ".venv"
$python = Join-Path $venv "Scripts\python.exe"
# An offline install brought its own Python (runtime\): repair that one.
$bundled = Join-Path $App "runtime\python.exe"
$extra = @()
if (Test-Path $bundled) { $python = $bundled; $extra = @("--break-system-packages") }
$log = Join-Path $App "setup.log"

# Downloads, caches and temporary files stay in the app folder.
$env:UV_CACHE_DIR = Join-Path $App ".cache\uv"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $App "python"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_PYTHON_BIN_DIR = Join-Path $App ".cache\uv-bin"
Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
$env:TEMP = $env:TMP = Join-Path $App ".cache\tmp"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }
function Run($exe) {
    & $exe @args
    if ($LASTEXITCODE -ne 0) { throw "$([IO.Path]::GetFileName($exe)) $($args -join ' ') failed (exit code $LASTEXITCODE)." }
}

Start-Transcript -Path $log -Force | Out-Null
$failed = $false
try {
    Write-Host "Project R set-up in $App"
    Write-Host "This downloads Python and the AI libraries: about 1 GB without an NVIDIA card,"
    Write-Host "about 3.5 GB with one. Keep this window open; it can take 5 to 20 minutes."
    if (-not (Test-Path $uv)) { throw "uv.exe is missing from $uv. Install Project R again." }

    Step "Python 3.11"
    if (-not (Test-Path $bundled)) {
        Run $uv python install 3.11
        Run $uv venv --python 3.11 --allow-existing $venv
    }

    # PyTorch: with CUDA for an NVIDIA card (the same build the project uses), else for the CPU.
    $smi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $smi -and (Test-Path "$env:SystemRoot\System32\nvidia-smi.exe")) { $smi = "$env:SystemRoot\System32\nvidia-smi.exe" }
    $gpu = $null
    if ($smi) { try { $gpu = (& $smi --query-gpu=name,driver_version --format=csv,noheader 2>$null | Select-Object -First 1) } catch { } }
    $cuda = ($Torch -eq "cuda") -or ($Torch -eq "auto" -and $gpu)
    if ($cuda) {
        Step "PyTorch for the NVIDIA card ($gpu), about 2.5 GB"
        Run $uv pip install --python $python @extra "torch==2.13.0+cu132" "torchvision==0.28.0+cu132" --index-url https://download.pytorch.org/whl/cu132
    } else {
        Step "PyTorch for the CPU (no NVIDIA card found: the commentary runs, but slower)"
        Run $uv pip install --python $python @extra "torch==2.13.0" "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cpu
    }

    Step "The rest of the app"
    Run $uv pip install --python $python @extra -r (Join-Path $App "requirements-app.txt")

    Step "Check"
    Push-Location $App
    try {
        Run $python "desktop\app.py" --check
        $device = & $python -c "import torch; print('GPU: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
        Write-Host "PyTorch runs on: $device"
        if ($cuda -and $device -eq "CPU") {
            Write-Host "The NVIDIA card is not used: update its driver (580 or newer, from nvidia.com)," -ForegroundColor Yellow
            Write-Host "then run 'Repair Project R' from the Start menu." -ForegroundColor Yellow
        }
    } finally { Pop-Location }
    Set-Content -Path (Join-Path (Split-Path -Parent $python) "project-r-ready.txt") -Value (Get-Date -Format s)
    Write-Host ""
    Write-Host "Project R is ready. Start it from the Start menu or the desktop icon." -ForegroundColor Green
} catch {
    $failed = $true
    Write-Host ""
    Write-Host "Set-up stopped: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Check the internet connection, then run 'Repair Project R' from the Start menu."
    Write-Host "Details: $log"
} finally {
    Stop-Transcript | Out-Null
}
if (-not $env:CI) { Write-Host ""; Read-Host "Press Enter to close" | Out-Null }
if ($failed) { exit 1 }
exit 0

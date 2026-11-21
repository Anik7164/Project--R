# Builds dist\ProjectR-Setup.exe on this PC: the setup file to give friends.
#
#   VS Code:  Terminal > Run Task... > "Build setup file"
#   or:       powershell -ExecutionPolicy Bypass -File installer\build.ps1
#
# The first time it fetches what it needs into the project folder (nothing
# onto C:): uv.exe (about 20 MB, put into the setup file) and Inno Setup
# (about 10 MB, the tool that makes the .exe) in .cache\tools.
param([string]$Version = (Get-Date -Format "yyyy.M.d"))
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

# 3. The setup file.
Write-Host "Building Project R $Version ..."
& $iscc "/DMyVersion=$Version" (Join-Path $PSScriptRoot "ProjectR.iss")
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed (exit code $LASTEXITCODE)." }
$out = Join-Path $root "dist\ProjectR-Setup.exe"
Write-Host ""
Write-Host "Done: $out ($([math]::Round((Get-Item $out).Length / 1MB)) MB)" -ForegroundColor Green
Write-Host "Give this one file to friends. Windows may say 'Windows protected your PC' (the file is not"
Write-Host "signed): they click 'More info' > 'Run anyway'."

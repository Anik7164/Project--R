# Makes a "Project R" shortcut with the app's icon that starts the desktop app.
#
#   powershell -ExecutionPolicy Bypass -File desktop\make_shortcut.ps1            # in the project folder
#   powershell -ExecutionPolicy Bypass -File desktop\make_shortcut.ps1 -Desktop   # on the Windows desktop
#   ... -Share   the shortcut also lets phones on the Wi-Fi join (iPhone app, viewers)
#
# The shortcut is a 2 KB file. Without -Desktop nothing is written outside the project folder.
param([switch]$Desktop, [switch]$Share)

$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $python)) { $python = (Get-Command pythonw -ErrorAction Stop).Source }
$folder = if ($Desktop) { [Environment]::GetFolderPath("Desktop") } else { $project }
$path = Join-Path $folder "Project R.lnk"

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
$shortcut.TargetPath = $python
$shortcut.Arguments = '"' + (Join-Path $project "desktop\app.py") + '"' + $(if ($Share) { " --share" } else { "" })
$shortcut.WorkingDirectory = $project
$shortcut.IconLocation = Join-Path $project "desktop\icon.ico"
$shortcut.Description = "Project R live Bengali football commentary"
$shortcut.Save()
Write-Host "Shortcut made: $path"

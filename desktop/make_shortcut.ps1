# Makes a "Project R" shortcut with the app's icon that starts the desktop app.
#
#   powershell -ExecutionPolicy Bypass -File desktop\make_shortcut.ps1            # in the project folder
#   powershell -ExecutionPolicy Bypass -File desktop\make_shortcut.ps1 -Desktop   # on the Windows desktop
#   ... -Share   the shortcut also lets phones on the Wi-Fi join (iPhone app, viewers)
#
# The shortcut is a 2 KB file. Without -Desktop nothing is written outside the project folder.
param([switch]$Desktop, [switch]$Share)

$project = Split-Path -Parent $PSScriptRoot
# The shortcut starts "Project R.bat", which finds the project's Python and
# says what is missing when something is.
$launcher = Join-Path $project "desktop\Project R.bat"
$folder = if ($Desktop) { [Environment]::GetFolderPath("Desktop") } else { $project }
$path = Join-Path $folder "Project R.lnk"

$shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
$shortcut.TargetPath = $launcher
$shortcut.Arguments = $(if ($Share) { "--share" } else { "" })
$shortcut.WindowStyle = 7     # minimised: only the app window shows
$shortcut.WorkingDirectory = $project
$shortcut.IconLocation = Join-Path $project "desktop\icon.ico"
$shortcut.Description = "Project R live Bengali football commentary"
$shortcut.Save()
Write-Host "Shortcut made: $path"

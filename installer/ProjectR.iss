; Project R setup file (Inno Setup 6). Build it with installer\build.ps1
; (VS Code: Terminal > Run Task > "Build setup file") or on GitHub
; (Actions > "Windows setup file"). Out comes dist\ProjectR-Setup.exe.
;
; The setup file holds the app (about 30 MB) and uv. Python and the AI
; libraries are downloaded on the friend's PC right after the files are
; copied (installer\setup_runtime.ps1), into the folder they chose: an NVIDIA
; card gets PyTorch with CUDA, any other PC the CPU build.

#ifndef MyVersion
  #define MyVersion "1.0.0"
#endif
#define Root ".."

[Setup]
AppId={{7B1E2C4A-5D3F-4A8B-9C61-2F0E8D7A9B13}
AppName=Project R
AppVersion={#MyVersion}
AppVerName=Project R {#MyVersion}
AppPublisher=Project R (NSU CSE498R)
DefaultDirName={localappdata}\Programs\Project R
DefaultGroupName=Project R
DisableProgramGroupPage=yes
; Anyone can install it, no administrator needed; any folder or drive can be chosen.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
DisableDirPage=no
OutputDir={#Root}\dist
OutputBaseFilename=ProjectR-Setup
SetupIconFile={#Root}\desktop\icon.ico
UninstallDisplayIcon={app}\desktop\icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; Python and PyTorch come on top of the files: room for them.
ExtraDiskSpaceRequired=6000000000

[Messages]
SelectDirDesc=Where should Project R go? Python and the AI libraries (up to 6 GB) go into the same folder.

[Tasks]
Name: "desktopicon"; Description: "Make a desktop icon"; GroupDescription: "Icons:"

[Files]
Source: "{#Root}\src\*"; DestDir: "{app}\src"; Flags: recursesubdirs ignoreversion; Excludes: "__pycache__,*.pyc"
Source: "{#Root}\webapp\*"; DestDir: "{app}\webapp"; Flags: recursesubdirs ignoreversion; Excludes: "__pycache__,*.pyc"
Source: "{#Root}\desktop\*"; DestDir: "{app}\desktop"; Flags: recursesubdirs ignoreversion; Excludes: "__pycache__,*.pyc"
Source: "{#Root}\models\football_yolo26s.pt"; DestDir: "{app}\models"; Flags: ignoreversion
Source: "{#Root}\yolo26n.pt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\dataset.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\botsort_custom.yaml"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\requirements-app.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\installer\setup_runtime.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "{#Root}\installer\bin\uv.exe"; DestDir: "{app}\installer\bin"; Flags: ignoreversion

[Dirs]
; Videos to play as a match, and the saved matches, are the user's: kept on uninstall.
Name: "{app}\data\videos"; Flags: uninsneveruninstall
Name: "{app}\outputs"; Flags: uninsneveruninstall

[Icons]
Name: "{group}\Project R"; Filename: "{app}\desktop\Project R.bat"; WorkingDir: "{app}"; IconFilename: "{app}\desktop\icon.ico"; Flags: runminimized
Name: "{group}\Project R (phones on the Wi-Fi can join)"; Filename: "{app}\desktop\Project R.bat"; Parameters: "--share"; WorkingDir: "{app}"; IconFilename: "{app}\desktop\icon.ico"; Flags: runminimized
Name: "{group}\Repair Project R (set up Python again)"; Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup_runtime.ps1"" -App ""{app}"""; WorkingDir: "{app}"
Name: "{group}\Uninstall Project R"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Project R"; Filename: "{app}\desktop\Project R.bat"; WorkingDir: "{app}"; IconFilename: "{app}\desktop\icon.ico"; Flags: runminimized; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup_runtime.ps1"" -App ""{app}"""; WorkingDir: "{app}"; StatusMsg: "Setting up Python and the AI libraries (a few GB, 5 to 20 minutes). Keep the black window open."; Flags: waituntilterminated
Filename: "{app}\desktop\Project R.bat"; WorkingDir: "{app}"; Description: "Start Project R now"; Flags: postinstall nowait skipifsilent runminimized shellexec

[UninstallDelete]
; What the set-up downloaded. Videos and saved matches (data, outputs) stay.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\.cache"
Type: files; Name: "{app}\setup.log"

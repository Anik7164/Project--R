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
; -Offline builds (build.ps1 -Offline) carry Python, the libraries and the
; offline voice inside; nothing is downloaded on the friend's PC.

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
#ifdef Offline
  #if Torch == "cpu"
OutputDir={#Root}\dist\ProjectR-Offline-CPU
  #else
OutputDir={#Root}\dist\ProjectR-Offline
  #endif
OutputBaseFilename=ProjectR-Setup
; Too big for one file: the .exe plus .bin parts of up to 2 GB, kept together.
DiskSpanning=yes
DiskSliceSize=max
#else
OutputDir={#Root}\dist
OutputBaseFilename=ProjectR-Setup
#endif
SetupIconFile={#Root}\desktop\icon.ico
UninstallDisplayIcon={app}\desktop\icon.ico
#ifdef Offline
Compression=lzma2/normal
LZMANumBlockThreads=4
#else
Compression=lzma2/max
#endif
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
#ifndef Offline
; Python and PyTorch come on top of the files: room for them.
ExtraDiskSpaceRequired=6000000000
#endif

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
Source: "{#Root}\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#Root}\installer\setup_runtime.ps1"; DestDir: "{app}\installer"; Flags: ignoreversion
Source: "{#Root}\installer\bin\uv.exe"; DestDir: "{app}\installer\bin"; Flags: ignoreversion
#ifdef Offline
Source: "{#Runtime}\*"; DestDir: "{app}\runtime"; Flags: recursesubdirs ignoreversion createallsubdirs; Excludes: "__pycache__,*.pyc"
Source: "{#Voice}\*"; DestDir: "{app}\.cache\huggingface"; Flags: recursesubdirs ignoreversion createallsubdirs
#endif

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
#ifndef Offline
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\setup_runtime.ps1"" -App ""{app}"""; WorkingDir: "{app}"; StatusMsg: "Setting up Python and the AI libraries (a few GB, 5 to 20 minutes). Keep the black window open."; Flags: waituntilterminated
#endif
Filename: "{app}\desktop\Project R.bat"; WorkingDir: "{app}"; Description: "Start Project R now"; Flags: postinstall nowait skipifsilent runminimized shellexec

[UninstallDelete]
; What the set-up downloaded. Videos and saved matches (data, outputs) stay.
Type: filesandordirs; Name: "{app}\.venv"
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\.cache"
Type: files; Name: "{app}\setup.log"

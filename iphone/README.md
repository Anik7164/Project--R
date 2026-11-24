# Project R iPhone app (native)

A SwiftUI app, built the same way as the ARGUS companion app. The match is still analysed on
the PC with the GPU; the phone connects over the same Wi-Fi.

What it does:

- **Finds the PC by itself.** The server announces `_projectr._tcp` on the Wi-Fi (Bonjour,
  `webapp/pairing.py`) with all its addresses. The app lists the PCs it hears.
- **Scan to connect.** On the PC, **Connect a phone** shows a QR code with the pairing link
  `projectr://connect?h=<addresses>&p=8000&n=<PC name>`. The iPhone's own camera opens it in
  the app, or the app's **Scan the QR code on the PC** reads it. The link only finds the PC.
- **Watch or operate.** Once the PC answers: **Watch the match** (no password), or
  **Operate** with the operator password, which **Connect a phone** on the PC shows (and can
  change there). The app checks the password with the PC before it opens the operator page.
- **Type the address** for Wi-Fi that hides PCs from each other (universities, hotels).
- **Remembers the last PC** and what was chosen there (watch, or operate while the password
  still works) and goes straight back when opened. A PC with several addresses (Wi-Fi,
  Ethernet, hotspot): each is tried in turn.
- **Match screen:** the PC's own page (the same one Safari shows): picture, Bengali lines,
  voice, and the operator buttons. **PC … change** (or **Operate** while watching) goes back
  to Watch or Operate.
- **This phone as a camera, nothing to type.** While the app is open, the PC lists the phone
  under Camera 1, 2 and 3 as "iPhone: <name> (app, starts by itself)". When the operator
  chooses it, the PC asks the app (over the page's socket, with a one-phone token) and the
  phone starts filming by itself, watching or operating. "Use this device as a camera" still
  starts it by hand. It sends JPEG pictures to the PC's `/ws/camera`, up to 15 a second,
  never more than 2 on their way (slow Wi-Fi drops pictures instead of building lag).
  Unlike Safari, this works over plain http. Only an IP camera needs an rtsp:// address.

## On the PC

Start Project R so phones can join: the Start menu shortcut **Project R (phones on the Wi-Fi
can join)**, or `desktop\Project R.bat --share`, or `python webapp/server.py --host 0.0.0.0`.
Windows asks about the firewall the first time: allow it on private networks. If it was
refused, in PowerShell as administrator:

```powershell
New-NetFirewallRule -DisplayName "Project R" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private
```

## Get the app (.ipa)

1. GitHub: **Actions → iPhone app (native) → Run workflow** (it also runs when `iphone/`
   changes). A few minutes on a GitHub Mac.
2. From the finished run, download **ProjectR-iPhone-ipa**; inside is `ProjectR-unsigned.ipa`.
3. On Windows, install [Sideloadly](https://sideloadly.io) (with iTunes and iCloud from
   Apple's website, not the Microsoft Store). Plug the iPhone in, drop the .ipa on Sideloadly,
   sign in with your Apple ID.
4. On the iPhone: Settings → General → VPN & Device Management → trust your Apple ID; switch
   on Developer Mode if iOS asks (Settings → Privacy & Security → Developer Mode).
5. Open Project R. When iOS asks to **find devices on your local network**, tap **Allow**.
   Then scan the PC's QR code.

A free Apple ID signs the app for 7 days; after that install it again the same way.
If the phone cannot reach the PC: Settings → Privacy & Security → Local Network → Project R on.

## Files

| File | What it does |
|---|---|
| `project.yml` | the Xcode project for XcodeGen (`xcodegen generate`); no `.xcodeproj` in git |
| `ProjectR/ProjectRApp.swift` | start: Bonjour search, last PC, `projectr://` links |
| `ProjectR/AppModel.swift` | which PC, trying its addresses, watch or operate, what is on screen |
| `ProjectR/ServerFinder.swift` | Bonjour browser (`_projectr._tcp`, TXT: hosts, port, path, name, v) |
| `ProjectR/Models.swift` | a PC; reading pairing links, operator addresses and typed addresses; what is remembered |
| `ProjectR/ContentView.swift` | the screens: PC list, connecting, watch or operate, match |
| `ProjectR/WebPage.swift` | the PC's match page in the app |
| `ProjectR/CameraSender.swift` | the phone camera to the PC (`/ws/camera`) |
| `ProjectR/CaptureViews.swift` | QR scanner, camera preview, camera screen |
| `.github/workflows/iphone.yml` | the cloud build |

The PC side: `webapp/pairing.py` (addresses, pairing link, Bonjour) and in
`webapp/server.py`: `GET /connect/info` (`{name, port, addresses, link, phones}`), the
`app`/`name` the page gives `/ws`, `POST /api/camera/wake`, and `POST /api/password` (the PC
itself only). The PC's own window is always the operator.

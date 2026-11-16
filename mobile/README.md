# Project R on an iPhone

The iPhone app is the web app's own pages (`webapp/static`) inside a small native shell
([Capacitor](https://capacitorjs.com)). The match is still analysed on the PC with the GPU;
the phone connects to it over the same Wi-Fi and can

- **watch**: the picture, the Bengali lines and the voice;
- **operate**: start/stop, kick-off, goal and card buttons (with the operator key);
- **be a camera**: "Use this device as a camera" sends the phone's camera to the PC, where it
  shows in the lists for Camera 1 and Camera 2 as "Phone camera: …".

On the PC start the server so phones can reach it. It prints the address and the key to type
into the app:

```powershell
python webapp/server.py --host 0.0.0.0      # or: desktop\"Project R.bat" --share
```

## Without installing anything (works today)

Open the address the server prints in Safari on the iPhone, then **Share → Add to Home Screen**.
"Project R" becomes an icon that opens full screen like an app. Open the operator address
(the one with `?key=…`) once and the phone stays the operator.

Watching and operating work this way. The phone's **camera** does not: Safari only gives the
camera to an https page. Either start the PC with `--https` (Safari then warns once about the
home-made certificate; choose to go on), or use the app below.

## The real app (.ipa)

Xcode only runs on a Mac, so the app is built on GitHub's Macs:

1. On GitHub: **Actions → iPhone app → Run workflow** (it also runs by itself when `mobile/`
   changes). About 5 to 10 minutes.
2. Open the finished run and download the artifact **ProjectR-iPhone-app**. Inside is
   `ProjectR-unsigned.ipa`.
3. Put it on the iPhone. The file is not signed, because signing needs an Apple account:
   - **Free Apple ID, from Windows:** install [Sideloadly](https://sideloadly.io), plug the
     iPhone in, drop the .ipa on it and sign in with an Apple ID. On the phone, trust the
     developer under Settings → General → VPN & Device Management (and switch on Developer Mode
     if iOS asks). An app signed with a free Apple ID stops opening after 7 days; install it
     again the same way.
   - **Apple Developer Program (99 USD a year):** the app can be signed properly, lasts a year
     and can go to TestFlight. This needs a Mac or a signing step added to the workflow.
4. Open the app, type the PC's address and the key, **Connect**. iOS asks for the local network
   and, on the camera page, for the camera: allow both.

Build minutes: on a private repository a Mac minute counts ten times against the free GitHub
Actions minutes, so one build costs about 50 to 100 of them. Public repositories are free.

## On a Mac

```bash
cd mobile
npm install
npm run ios          # copies the pages into www/, makes ios/, applies ios-settings.mjs
npx cap open ios     # Xcode: choose your team under Signing, then Run on the phone
```

## What is in here

| File | What it does |
|---|---|
| `build-www.mjs` | copies `webapp/static` into `www/` and marks it as "inside the app" (`shell.js`), so the pages ask for the PC's address |
| `capacitor.config.json` | the app's name and id (`com.projectr.livecommentary`) |
| `ios-settings.mjs` | after `cap add ios`: camera and local-network permission texts, plain http to the PC on the Wi-Fi, icon and launch screen |
| `assets/` | icon and launch screen (drawn by `webapp/make_icons.py`) |
| `.github/workflows/ios-app.yml` (repository root) | the cloud build |

`www/`, `ios/` and `node_modules/` are made by the commands above and are not in git.

Not tried on a real iPhone yet: the pages were tested in a desktop browser at an iPhone's
screen size against a running server (connect screen, watching, operator, camera page). The
same shell also builds for Android (`npx cap add android`), which was not set up here.

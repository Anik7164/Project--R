"""
Project R as a desktop app: the live commentary page in a window of its
own. It starts the commentary server by itself and stops it when the
window is closed, so there is no terminal to keep open and no browser.

    python desktop/app.py                 # this PC runs the commentary
    python desktop/app.py --share         # ... and phones / other PCs on the Wi-Fi may join
    python desktop/app.py --connect 192.168.0.5:8000    # only a window onto another PC

On Windows, double-click "desktop\\Project R.bat" (or the shortcut that
desktop\\make_shortcut.ps1 makes) instead of typing this.

Needs: pip install -r requirements.txt   (pywebview is in it).
Without it the page opens in an Edge or Chrome app window instead, and
failing that in the normal browser.

Everything it writes (the window's settings, the server log) stays in
<project>/.cache/desktop.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import project_cache  # noqa: E402,F401  keeps downloads/caches in <project>/.cache

import argparse  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

TITLE = "Project R · লাইভ ধারাভাষ্য"
SERVER = ROOT / "webapp" / "server.py"
FOLDER = ROOT / ".cache" / "desktop"
KEY_FILE = ROOT / ".cache" / "webapp-operator-key.txt"
PORT = 8000
START_SECONDS = 90.0     # a cold start on a slow disk


def answers(address, timeout=1.5):
    """Is a Project R server answering at http://<address>?"""
    try:
        with urllib.request.urlopen(f"{address}/api/state", timeout=timeout) as reply:
            return reply.status == 200 and b'"operator"' in reply.read(4096)
    except (urllib.error.URLError, OSError, ValueError):
        return False


def free_port(wanted=PORT):
    """`wanted` if nothing listens there, else any free port."""
    for port in (wanted, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
                return probe.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free port on this PC.")


def remote_address(text):
    """--connect 192.168.0.5, 192.168.0.5:8000 or a whole address."""
    text = text.strip().rstrip("/")
    if not text.startswith(("http://", "https://")):
        text = f"http://{text}"
    scheme, rest = text.split("://", 1)
    host = rest.split("/", 1)[0]
    if ":" not in host and scheme == "http":
        rest = rest.replace(host, f"{host}:{PORT}", 1)
    return f"{scheme}://{rest}"


def server_command(port, share):
    # --exit-with: the server also goes when this app is killed, not only closed.
    command = [sys.executable, str(SERVER), "--port", str(port), "--exit-with", str(os.getpid())]
    if share:
        command += ["--host", "0.0.0.0"]
    return command


def start_server(port, share, log_path):
    """The commentary server as a process of its own, its output in a log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8", errors="replace")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)     # Windows: no black console window
    environment = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    return subprocess.Popen(server_command(port, share), cwd=str(ROOT), stdout=log,
                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=flags, env=environment)


def wait_until_up(address, process, seconds=START_SECONDS):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if answers(address):
            return True
        if process is not None and process.poll() is not None:
            return False
        time.sleep(0.3)
    return False


def stop_server(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def operator_key():
    try:
        return KEY_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def app_browser():
    """Edge or Chrome, which can show a page as a window without the
    browser around it (--app=)."""
    names = ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser",
             "microsoft-edge"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    places = []
    for variable in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        base = os.environ.get(variable)
        if base:
            places += [Path(base) / "Microsoft" / "Edge" / "Application" / "msedge.exe",
                       Path(base) / "Google" / "Chrome" / "Application" / "chrome.exe"]
    places += [Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
               Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")]
    for place in places:
        if place.is_file():
            return str(place)
    return None


def show_window(address, title=TITLE):
    """Blocks until the window is closed. Returns how it was shown."""
    try:
        import webview
    except ImportError:
        webview = None
    if webview is not None:
        # The page's own settings (picture delay, chosen cameras) are kept, in the project.
        storage = FOLDER / "webview"
        storage.mkdir(parents=True, exist_ok=True)
        try:
            webview.create_window(title, address, width=1440, height=940, min_size=(900, 600),
                                  background_color="#0b1210", text_select=True)
            try:
                webview.start(private_mode=False, storage_path=str(storage),
                              icon=str(ROOT / "desktop" / "icon.png"))
            except TypeError:               # an older pywebview
                webview.start()
            return "window"
        except Exception as error:          # no window system it can use: the browser instead
            print(f"pywebview could not open a window ({type(error).__name__}: {error}). "
                  "Using a browser window.")
    browser = os.environ.get("PROJECT_R_BROWSER") or app_browser()
    if browser is not None:
        profile = FOLDER / "browser"
        profile.mkdir(parents=True, exist_ok=True)
        extra = os.environ.get("PROJECT_R_BROWSER_FLAGS", "").split()
        subprocess.run([browser, f"--app={address}", f"--user-data-dir={profile}",
                        "--no-first-run", "--no-default-browser-check",
                        "--window-size=1440,940", *extra], check=False)
        return "browser window"
    import webbrowser

    webbrowser.open(address)
    print(f"Opened {address} in the browser. Press Ctrl+C here to stop the commentary server.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return "browser"


def tell(text):
    """A message for the person, also when there is no console (pythonw)."""
    print(text)
    if sys.platform == "win32" and (sys.stdout is None or not sys.stdout.isatty()):
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, text, "Project R", 0x10)
        except Exception:
            pass


NEEDED = [("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"), ("multipart", "python-multipart"),
          ("cv2", "opencv-python"), ("numpy", "numpy"), ("ultralytics", "ultralytics"),
          ("psutil", "psutil")]


def check():
    """What "Project R.bat" runs first, in a console: is everything the app
    needs installed in this Python? Prints what is missing."""
    import importlib.util

    print(f"Python: {sys.executable}  ({sys.version.split()[0]})")
    missing = [package for module, package in NEEDED if importlib.util.find_spec(module) is None]
    if importlib.util.find_spec("webview") is None:
        print("pywebview is not installed: the page opens in an Edge or Chrome window instead "
              "(pip install -r requirements.txt for the app window).")
    if not SERVER.is_file():
        print(f"Missing: {SERVER}. Is this the whole project, on the branch with the web app?")
        return 1
    if missing:
        print("Missing in this Python: " + ", ".join(missing))
        return 2
    print("Everything the app needs is installed.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--share", action="store_true",
                        help="let phones and other PCs on the Wi-Fi join (iPhone app, viewers)")
    parser.add_argument("--connect", metavar="ADDRESS",
                        help="do not run the commentary here: show the one running on another PC")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--check", action="store_true",
                        help="only say whether everything the app needs is installed")
    args = parser.parse_args(argv)
    if args.check:
        return check()

    process = None
    if args.connect:
        address = remote_address(args.connect)
        if not answers(address, timeout=5):
            tell(f"No Project R server answers at {address}.\n"
                 "Start it there with --share (or: python webapp/server.py --host 0.0.0.0).")
            return 1
    else:
        address = f"http://127.0.0.1:{args.port}"
        if answers(address):
            pass                            # already running (a second window, or a terminal)
        else:
            port = free_port(args.port)
            address = f"http://127.0.0.1:{port}"
            log_path = FOLDER / "server.log"
            process = start_server(port, args.share, log_path)
            if not wait_until_up(address, process):
                stop_server(process)
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                                 .splitlines()[-12:])
                tell("The commentary server did not start.\n\n" + tail +
                     f"\n\nFull log: {log_path}\n"
                     "Is the web app installed?  pip install -r requirements.txt")
                return 1
        if args.share and operator_key():
            address += f"/?key={operator_key()}"    # this window is the operator's
    try:
        show_window(address)
    finally:
        stop_server(process)
    return 0


def run():
    """main(), but a crash is shown and written down: with pythonw (no
    console) it would otherwise vanish without a word."""
    import traceback

    if sys.stdout is None:                  # pythonw: keep what is printed
        FOLDER.mkdir(parents=True, exist_ok=True)
        sys.stdout = sys.stderr = (FOLDER / "app.log").open("w", encoding="utf-8",
                                                           errors="replace", buffering=1)
    try:
        return main()
    except SystemExit:
        raise
    except Exception:
        text = traceback.format_exc()
        print(text)
        tell("Project R could not start.\n\n" + text[-1500:] +
             f"\n\nLog: {FOLDER / 'app.log'}")
        return 1


if __name__ == "__main__":
    sys.exit(run())

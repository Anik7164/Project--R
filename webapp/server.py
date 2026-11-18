"""
Web app for live Bengali commentary.

    python webapp/server.py                  # this PC only: http://127.0.0.1:8000
    python webapp/server.py --host 0.0.0.0   # also phones/laptops on the same Wi-Fi

Open the page, choose the camera (or a stream address, or a video to play
as if it were live), press Start. The page shows the match a few seconds
late with the Bengali lines and the voice on the moment they are about.

With --host 0.0.0.0 the address printed with ?key=... is the operator's
page (start/stop, kick-off, goal and card buttons); everybody else gets
the plain address and can only watch and listen.

    python webapp/server.py --host 0.0.0.0 --https

--https is for a phone's browser that is to be a camera (its camera page
only works over https there). The certificate is made on this PC, so the
phone shows a warning the first time: choose to go on. The iPhone app
(mobile/) and the desktop app (desktop/) need no https.

Needs: pip install -r requirements-webapp.txt
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import project_cache  # noqa: E402,F401  keeps downloads/caches in <project>/.cache

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import secrets  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
import socket  # noqa: E402

from fastapi import FastAPI, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

VIDEO_DIR = ROOT / "data" / "videos"
TRACKING_DIR = ROOT / "outputs" / "tracking"
LIVE_DIR = ROOT / "outputs" / "live"
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}



@asynccontextmanager
async def lifespan(_):
    hub.loop = asyncio.get_running_loop()
    if hub.warm:
        # The models load now, so Start opens the camera and talks at once.
        from live_engine import warm_up_in_background

        warm_up_in_background()
    yield
    if hub.match is not None:
        hub.match.stop()


app = FastAPI(title="Project R live commentary", lifespan=lifespan)
# The iPhone app's page comes from the app itself, not from this server.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(capacitor|ionic|https?)://localhost(:\d+)?$",
    allow_methods=["GET", "POST"], allow_headers=["X-Key", "Content-Type"])


class Hub:
    """Sends what the engine produces to every open browser."""

    FRAME_LIMIT = 20      # a slow browser skips pictures rather than fall behind
    SOUND_LIMIT = 200

    def __init__(self):
        self.clients = {}  # queue -> True: this browser only gets the camera on air
        self.loop = None
        self.match = None
        self.key = None    # operator key; None: anyone who can open the page
        self.phones = {}   # id -> PushFeed: devices sending their camera in
        self.share = None  # {"address", "key"}: what a phone on the Wi-Fi types to join
        self.warm = False  # load the models when the server starts (main() sets it)

    def _deliver(self, item):
        from live_engine import CAMERA_OF_KIND

        camera = CAMERA_OF_KIND.get(item[0]) if isinstance(item, bytes) else None
        on_air = getattr(self.match, "on_air", 1)
        for client, air_only in list(self.clients.items()):
            if isinstance(item, bytes):
                if camera is not None and air_only:
                    # Viewers and the broadcast see one picture: the camera on air.
                    if camera != on_air:
                        continue
                    if camera != 1:
                        item_sent = b"\x01" + item[1:]
                        if client.qsize() < self.FRAME_LIMIT:
                            client.put_nowait(item_sent)
                        continue
                limit = self.SOUND_LIMIT if item[:1] == b"\x02" else self.FRAME_LIMIT
                if client.qsize() >= limit:
                    continue
            elif client.qsize() >= 2000:
                continue
            client.put_nowait(item)

    def publish(self, item):
        """Called from the engine's threads."""
        if self.loop is not None and self.clients:
            self.loop.call_soon_threadsafe(self._deliver, item)


hub = Hub()


def in_folder(folder, name):
    """folder/name, refusing anything that is not a plain file name."""
    if not name or Path(name).name != name:
        raise HTTPException(400, "bad file name")
    path = folder / name
    if not path.is_file():
        raise HTTPException(404, "no such file")
    return path


def need_operator(request: Request):
    if hub.key and not secrets.compare_digest(request.headers.get("x-key", ""), hub.key):
        raise HTTPException(403, "This page can only watch. Open the operator address "
                                 "printed in the terminal.")


def list_videos():
    videos = []
    if VIDEO_DIR.is_dir():
        for path in sorted(VIDEO_DIR.iterdir()):
            if path.suffix.lower() in VIDEO_TYPES:
                videos.append({
                    "name": path.name,
                    "megabytes": round(path.stat().st_size / 1e6, 1),
                    "finished": (TRACKING_DIR / f"{path.stem}_commentary.json").exists(),
                })
    return videos


def list_microphones():
    """Sound inputs ffmpeg can open on this PC (Windows), for the camera's
    microphone box."""
    if sys.platform != "win32":
        return []
    import re
    import subprocess

    import audio_cues

    ffmpeg = audio_cues.ffmpeg_path()
    if ffmpeg is None:
        return []
    try:
        listing = subprocess.run(
            [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, timeout=10).stderr.decode("utf-8", "replace")
    except (OSError, subprocess.TimeoutExpired):
        return []
    return re.findall(r'"([^"@][^"]*)"\s*\(audio\)', listing)


def camera_names():
    """{camera number: name} of the cameras plugged into this PC, as far
    as the system tells. The numbers are the ones OpenCV opens."""
    import re
    import subprocess

    names = {}
    try:
        if sys.platform == "win32":
            import audio_cues

            ffmpeg = audio_cues.ffmpeg_path()
            if ffmpeg is not None:
                listing = subprocess.run(
                    [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow",
                     "-i", "dummy"],
                    capture_output=True, timeout=10).stderr.decode("utf-8", "replace")
                # DirectShow lists them in the order OpenCV numbers them.
                found = re.findall(r'"([^"@][^"]*)"\s*\(video\)', listing)
                names = dict(enumerate(found))
        elif sys.platform.startswith("linux"):
            for node in sorted(Path("/sys/class/video4linux").glob("video*"),
                               key=lambda n: int(n.name[5:])):
                first = (node / "index").read_text().strip() == "0"   # not its metadata node
                if first:
                    names[int(node.name[5:])] = (node / "name").read_text().strip()
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    return names


def list_cameras():
    """What can be chosen as camera 1 or camera 2."""
    names = camera_names() or {number: "" for number in range(4)}
    cameras = [{"id": f"camera:{number}",
                "label": f"Camera {number}" + (f": {name}" if name else "")}
               for number, name in sorted(names.items())]
    cameras += [{"id": f"phone:{phone}", "label": f"Phone camera: {feed.name}"}
                for phone, feed in hub.phones.items()]
    return cameras


def source_spec(source, live_only=False):
    """What the page chose, checked, as the engine wants it."""
    kind = (source or {}).get("kind")
    if kind == "file" and not live_only:
        return {"kind": "file", "path": in_folder(VIDEO_DIR, source.get("name", ""))}
    if kind == "url":
        url = (source.get("url") or "").strip()
        if not url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://",
                                       "srt://")):
            raise HTTPException(400, "The stream address must start with rtsp://, rtmp://, "
                                     "http://, https://, udp:// or srt://")
        return {"kind": "url", "url": url}
    if kind == "camera":
        try:
            number = int(source.get("index", 0))
        except (TypeError, ValueError):
            raise HTTPException(400, "The camera number must be a number.")
        if not 0 <= number <= 63:
            raise HTTPException(400, "The camera number must be between 0 and 63.")
        name = camera_names().get(number)
        return {"kind": "camera", "index": number,
                "label": f"camera {number}" + (f" ({name})" if name else "")}
    if kind == "phone":
        feed = hub.phones.get(source.get("id"))
        if feed is None:
            raise HTTPException(400, "That phone camera is not connected any more. "
                                     "Open its camera page again and choose it again.")
        return {"kind": "phone", "feed": feed, "id": source.get("id"),
                "label": f"phone camera {feed.name}"}
    raise HTTPException(400, "Choose a camera, a stream or a video.")


def same_device(one, other):
    if one["kind"] != other["kind"]:
        return False
    return {"camera": one.get("index") == other.get("index"),
            "url": one.get("url") == other.get("url"),
            "phone": one.get("feed") is other.get("feed")}.get(one["kind"], False)


def in_use(spec):
    """Is the running match reading this camera?"""
    match = hub.match
    if match is None or match.state not in ("starting", "running"):
        return False
    used = [match.spec, *getattr(match, "other_specs", [])]
    return any(one is not None and same_device(spec, one) for one in used)




@app.get("/api/state")
def state(request: Request):
    from appearance import BENGALI_COLOUR_NAMES
    from live_engine import DEFAULTS

    operator = not hub.key or secrets.compare_digest(
        request.headers.get("x-key", ""), hub.key)
    return {
        "operator": operator,
        "match": hub.match.hello() if hub.match else None,
        "videos": list_videos() if operator else [],
        "mics": list_microphones() if operator else [],
        "cameras": list_cameras() if operator else [],
        "share": hub.share if operator else None,
        "colours": BENGALI_COLOUR_NAMES,
        "defaults": DEFAULTS,
    }


@app.post("/api/upload")
async def upload(request: Request, file: UploadFile):
    need_operator(request)
    name = Path(file.filename or "").name
    if Path(name).suffix.lower() not in VIDEO_TYPES:
        raise HTTPException(400, "Choose a video file (mp4, mov, mkv, avi, webm).")
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    target = VIDEO_DIR / name
    if target.exists():
        # Never write over a video that is already there.
        stem, number = target.stem, 2
        while target.exists():
            target = VIDEO_DIR / f"{stem} ({number}){target.suffix}"
            number += 1
    with target.open("wb") as out:
        while chunk := await file.read(1 << 20):
            out.write(chunk)
    return {"name": target.name, "videos": list_videos()}


@app.post("/api/start")
async def start(request: Request):
    need_operator(request)
    from live_engine import MAX_CAMERAS, LiveMatch

    body = await request.json()
    spec = source_spec(body.get("source"))
    # Cameras 2 and 3 are shown beside camera 1 and can be put on air; which
    # device is which is the operator's choice on the page.
    wanted = list(body.get("others") or [])
    if body.get("second") and not wanted:
        wanted = [body["second"]]
    if len(wanted) > MAX_CAMERAS - 1:
        raise HTTPException(400, f"At most {MAX_CAMERAS} cameras.")
    others = []
    for number, source in enumerate(wanted, start=2):
        other = source_spec(source, live_only=True)
        for earlier, used in enumerate([spec, *others], start=1):
            if same_device(used, other):
                raise HTTPException(400, f"Camera {earlier} and camera {number} are the same "
                                         "device. Choose different ones.")
        others.append(other)
    if hub.match is not None and hub.match.state in ("starting", "running"):
        raise HTTPException(409, "A match is already running. Stop it first.")
    hub.match = LiveMatch(spec, body.get("settings"), hub.publish, hub.publish, others=others)
    hub.match.start()
    hub.publish(hub.match.hello())
    return {"session": hub.match.id}


@app.get("/api/cameras")
def cameras(request: Request):
    need_operator(request)
    return {"cameras": list_cameras()}


@app.post("/api/camera/preview")
def camera_preview(request: Request, source: dict):
    """One picture from a camera, so the operator sees which device is
    which before deciding camera 1 and camera 2."""
    need_operator(request)
    import cv2

    from live_engine import FrameSource

    spec = source_spec(source, live_only=True)
    if in_use(spec):
        raise HTTPException(409, "The running match is using this camera.")
    reader = FrameSource(spec, 5.0)
    try:
        capture = reader._open()
    except RuntimeError as error:
        raise HTTPException(400, str(error))
    try:
        frame = None
        for _ in range(12):            # the first pictures of a webcam are dark
            ok, picture = capture.read()
            if ok and picture is not None:
                frame = picture
            elif frame is not None:
                break
    finally:
        capture.release()
    if frame is None:
        raise HTTPException(400, f"{reader.describe()} opened but sent no picture.")
    scale = min(1.0, 480 / frame.shape[1])
    if scale < 1.0:
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(jpeg.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/stop")
def stop(request: Request):
    need_operator(request)
    if hub.match is not None:
        hub.match.stop()
    return {"ok": True}


@app.post("/api/operator")
async def operator(request: Request):
    need_operator(request)
    body = await request.json()
    if body.get("action") == "on_air":
        # Which camera viewers and the stream see: also while the models load.
        if hub.match is None or hub.match.state not in ("starting", "running"):
            raise HTTPException(409, "No match is running.")
        try:
            hub.match.put_on_air(body.get("camera"))
        except ValueError as error:
            raise HTTPException(400, str(error))
        return {"ok": True, "on_air": hub.match.on_air}
    if hub.match is None or hub.match.state != "running":
        raise HTTPException(409, "No match is running.")
    try:
        if body.get("action") == "colour":
            hub.match.set_colour(body.get("team"), body.get("colour"))
            hub.publish(hub.match.stats())
        else:
            hub.match.operator(body.get("action"), body.get("team"))
    except ValueError as error:
        raise HTTPException(400, str(error))
    return {"ok": True}


@app.get("/live/{session}/audio/{name}")
def spoken_line(session: str, name: str):
    if Path(session).name != session:
        raise HTTPException(400, "bad session")
    return FileResponse(in_folder(LIVE_DIR / session / "audio", name))


@app.get("/api/finished/{name}")
def finished(name: str):
    """A match the offline scripts already did: its lines, for playing
    back with the video."""
    video = in_folder(VIDEO_DIR, name)
    lines = TRACKING_DIR / f"{video.stem}_commentary.json"
    if not lines.exists():
        raise HTTPException(404, "No finished commentary for this video yet.")
    spoken = TRACKING_DIR / f"{video.stem}_commentary.mp4"
    return {
        "lines": json.loads(lines.read_text(encoding="utf-8")),
        "video": f"/media/spoken/{spoken.name}" if spoken.exists() else f"/media/video/{name}",
        "spoken": spoken.exists(),
    }


@app.get("/media/video/{name}")
def media_video(name: str):
    return FileResponse(in_folder(VIDEO_DIR, name))


@app.get("/media/spoken/{name}")
def media_spoken(name: str):
    return FileResponse(in_folder(TRACKING_DIR, name))


@app.websocket("/ws")
async def feed(socket_: WebSocket):
    await socket_.accept()
    client = asyncio.Queue()
    # A viewer (no operator key) or the broadcast page (?view=air) only gets
    # the camera on air; the operator gets all of them, side by side.
    viewer = bool(hub.key) and not secrets.compare_digest(
        socket_.query_params.get("key", ""), hub.key)
    hub.clients[client] = viewer or socket_.query_params.get("view") == "air"
    try:
        if hub.match is not None:
            await socket_.send_text(json.dumps(hub.match.hello(), ensure_ascii=False))
        while True:
            item = await client.get()
            if isinstance(item, bytes):
                await socket_.send_bytes(item)
            else:
                await socket_.send_text(json.dumps(item, ensure_ascii=False))
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        hub.clients.pop(client, None)


@app.websocket("/ws/camera")
async def camera_in(socket_: WebSocket):
    """A phone (or laptop) sending its camera: JPEG pictures in, one after
    another. It then shows in the camera lists as "Phone camera: name"."""
    await socket_.accept()
    if hub.key and not secrets.compare_digest(socket_.query_params.get("key", ""), hub.key):
        await socket_.close(code=4403, reason="operator key needed")
        return
    from live_engine import PushFeed

    wanted = (socket_.query_params.get("name") or "phone").strip()[:40] or "phone"
    name, number = wanted, 2
    while any(feed.name == name for feed in hub.phones.values()):
        name, number = f"{wanted} {number}", number + 1
    phone, feed = secrets.token_hex(4), PushFeed(name)
    hub.phones[phone] = feed
    hub.publish({"type": "cameras"})
    try:
        await socket_.send_text(json.dumps({"id": phone, "name": name}))
        while True:
            message = await socket_.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data and len(data) <= 4_000_000:
                feed.push(data)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        feed.close()
        hub.phones.pop(phone, None)
        hub.publish({"type": "cameras"})


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/camera.html")
def camera_page():
    return FileResponse(STATIC / "camera.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return "this-pc"


def operator_key():
    """The same key every time the server starts, so a phone that was
    given the operator address once keeps working. Kept in .cache."""
    path = ROOT / ".cache" / "webapp-operator-key.txt"
    try:
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    except OSError:
        pass
    key = secrets.token_urlsafe(6)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(key, encoding="utf-8")
    except OSError:
        pass
    return key


def certificate(address, folder=None):
    """(certificate file, key file) for https on this PC's own address,
    made here once and kept in .cache."""
    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    folder = folder or ROOT / ".cache" / "webapp-https"
    cert_file, key_file, made_for = folder / "cert.pem", folder / "key.pem", folder / "address.txt"
    try:
        if cert_file.exists() and key_file.exists() and made_for.read_text().strip() == address:
            return cert_file, key_file
    except OSError:
        pass
    folder.mkdir(parents=True, exist_ok=True)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Project R live commentary")])
    names = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
    try:
        names.append(x509.IPAddress(ipaddress.ip_address(address)))
    except ValueError:
        names.append(x509.DNSName(address))
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=800))
            .add_extension(x509.SubjectAlternativeName(names), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256()))
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
        serialization.NoEncryption()))
    made_for.write_text(address)
    return cert_file, key_file


def exit_with(pid):
    """The desktop app started this server: when its window is gone (even
    if it was killed), the camera and the GPU are let go."""
    import os
    import time

    import psutil

    while psutil.pid_exists(pid):
        time.sleep(2)
    if hub.match is not None:
        hub.match.stop()
        time.sleep(1)                   # the session is saved
    os._exit(0)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 lets other devices on the network watch")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--exit-with", type=int, metavar="PID",
                        help="stop when this process is gone (the desktop app's window)")
    parser.add_argument("--https", action="store_true",
                        help="for a phone browser that is to be a camera "
                             "(needs: pip install cryptography)")
    parser.add_argument("--no-preload", action="store_true",
                        help="load the models at the first Start, not when the server starts")
    args = parser.parse_args()
    hub.warm = not args.no_preload

    local = args.host in ("127.0.0.1", "localhost", "::1")
    scheme, secure = ("https", {}) if args.https else ("http", {})
    address = "127.0.0.1" if local else lan_address()
    if args.https:
        try:
            cert_file, key_file = certificate(address)
        except ImportError:
            sys.exit("--https needs:  pip install cryptography")
        secure = {"ssl_certfile": str(cert_file), "ssl_keyfile": str(key_file)}
    if local:
        print(f"\nOpen  {scheme}://127.0.0.1:{args.port}")
        print(f"Broadcast (OBS browser source, the camera on air only): "
              f"{scheme}://127.0.0.1:{args.port}/?view=air\n")
    else:
        hub.key = operator_key()
        hub.share = {"address": f"{address}:{args.port}", "key": hub.key,
                     "url": f"{scheme}://{address}:{args.port}"}
        print(f"\nOperator (you):      {scheme}://127.0.0.1:{args.port}/?key={hub.key}")
        print(f"Operator on a phone: {scheme}://{address}:{args.port}/?key={hub.key}")
        print(f"Viewers:             {scheme}://{address}:{args.port}")
        print(f"Broadcast (OBS):     {scheme}://127.0.0.1:{args.port}/?view=air")
        if args.https:
            print("The iPhone app cannot use this https server (its certificate is home-made): "
                  "for the app, start without --https.\n")
        else:
            print(f"In the iPhone app:   address {address}:{args.port}   key {hub.key}\n")
    if args.exit_with:
        import threading

        threading.Thread(target=exit_with, args=(args.exit_with,), daemon=True).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **secure)


if __name__ == "__main__":
    main()

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

Needs: pip install fastapi "uvicorn[standard]" python-multipart edge-tts
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
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

VIDEO_DIR = ROOT / "data" / "videos"
TRACKING_DIR = ROOT / "outputs" / "tracking"
LIVE_DIR = ROOT / "outputs" / "live"
STATIC = Path(__file__).resolve().parent / "static"
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}



@asynccontextmanager
async def lifespan(_):
    hub.loop = asyncio.get_running_loop()
    yield
    if hub.match is not None:
        hub.match.stop()


app = FastAPI(title="Project R live commentary", lifespan=lifespan)


class Hub:
    """Sends what the engine produces to every open browser."""

    FRAME_LIMIT = 20      # a slow browser skips pictures rather than fall behind
    SOUND_LIMIT = 200

    def __init__(self):
        self.clients = set()
        self.loop = None
        self.match = None
        self.key = None    # operator key; None: anyone who can open the page

    def _deliver(self, item):
        for client in list(self.clients):
            if isinstance(item, bytes):
                limit = self.FRAME_LIMIT if item[:1] == b"\x01" else self.SOUND_LIMIT
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
    from live_engine import LiveMatch

    body = await request.json()
    source = body.get("source") or {}
    kind = source.get("kind")
    if kind == "file":
        spec = {"kind": "file", "path": in_folder(VIDEO_DIR, source.get("name", ""))}
    elif kind == "url":
        url = (source.get("url") or "").strip()
        if not url.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://",
                                       "srt://")):
            raise HTTPException(400, "The stream address must start with rtsp://, rtmp://, "
                                     "http://, https://, udp:// or srt://")
        spec = {"kind": "url", "url": url}
    elif kind == "camera":
        spec = {"kind": "camera", "index": int(source.get("index", 0))}
    else:
        raise HTTPException(400, "Choose a camera, a stream or a video.")
    if hub.match is not None and hub.match.state in ("starting", "running"):
        raise HTTPException(409, "A match is already running. Stop it first.")
    hub.match = LiveMatch(spec, body.get("settings"), hub.publish, hub.publish)
    hub.match.start()
    hub.publish(hub.match.hello())
    return {"session": hub.match.id}


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
    hub.clients.add(client)
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
        hub.clients.discard(client)


@app.get("/")
def page():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def lan_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return "this-pc"


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 lets other devices on the network watch")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    local = args.host in ("127.0.0.1", "localhost", "::1")
    if local:
        print(f"\nOpen  http://127.0.0.1:{args.port}\n")
    else:
        hub.key = secrets.token_urlsafe(6)
        address = lan_address()
        print(f"\nOperator (you):  http://127.0.0.1:{args.port}/?key={hub.key}")
        print(f"Viewers:         http://{address}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

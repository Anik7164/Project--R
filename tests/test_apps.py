"""The desktop app's launcher and the iPhone app's page bundle."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "desktop"))

import app as desktop  # noqa: E402


def test_addresses_and_the_server_command():
    assert desktop.remote_address("192.168.0.5") == "http://192.168.0.5:8000"
    assert desktop.remote_address("192.168.0.5:9000/") == "http://192.168.0.5:9000"
    assert desktop.remote_address("https://match.example") == "https://match.example"
    command = desktop.server_command(8123, share=True)
    assert command[0] == sys.executable and command[1].endswith("server.py")
    assert command[2:] == ["--port", "8123", "--exit-with", str(os.getpid()),
                           "--host", "0.0.0.0"]
    assert "--host" not in desktop.server_command(8123, share=False)
    assert not desktop.answers("http://127.0.0.1:9", timeout=0.5)


def test_free_port_moves_on_when_the_usual_one_is_taken():
    import socket

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        taken = busy.getsockname()[1]
        assert desktop.free_port(taken) != taken
    assert desktop.free_port(taken) == taken


def test_the_desktop_app_starts_the_server_and_stops_it(tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    port = desktop.free_port(0)
    address = f"http://127.0.0.1:{port}"
    process = desktop.start_server(port, share=False, log_path=tmp_path / "server.log")
    try:
        assert desktop.wait_until_up(address, process, seconds=60)
        assert desktop.answers(address)
    finally:
        desktop.stop_server(process)
    assert process.poll() is not None and not desktop.answers(address, timeout=0.5)


def test_a_server_that_cannot_start_is_reported_not_waited_for(tmp_path, monkeypatch):
    monkeypatch.setattr(desktop, "SERVER", tmp_path / "missing.py")
    process = desktop.start_server(desktop.free_port(0), False, tmp_path / "server.log")
    assert not desktop.wait_until_up("http://127.0.0.1:9", process, seconds=20)
    assert "missing.py" in (tmp_path / "server.log").read_text()


def test_the_iphone_app_ships_the_same_pages_and_asks_for_the_pc(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    mobile = tmp_path / "mobile"
    mobile.mkdir()
    shutil.copy(ROOT / "mobile" / "build-www.mjs", mobile)
    (tmp_path / "webapp").mkdir()
    shutil.copytree(ROOT / "webapp" / "static", tmp_path / "webapp" / "static")
    subprocess.run([node, "build-www.mjs"], cwd=mobile, check=True, capture_output=True)
    www = mobile / "www"
    for page in ("index.html", "camera.html"):
        assert (www / page).read_text(encoding="utf-8") == \
            (ROOT / "webapp" / "static" / page).read_text(encoding="utf-8")
    assert (www / "static" / "app.js").exists() and (www / "static" / "common.js").exists()
    assert "PROJECT_R_SHELL = true" in (www / "static" / "shell.js").read_text()
    # ... while the copy the PC serves says it is not the app.
    assert "PROJECT_R_SHELL = false" in (ROOT / "webapp" / "static" / "shell.js").read_text()

// Project R: what the match page and the camera page share.
//
// Where is the server? Opened from the PC's own address (browser, desktop
// app) it is that address. Inside the iPhone app the page is part of the
// app, so the app asks once for the PC's address and remembers it.

const SHELL = Boolean(window.PROJECT_R_SHELL) ||
  ["capacitor:", "ionic:", "file:"].includes(location.protocol);

function remembered(name) {
  try { return localStorage.getItem(name); } catch { return null; }
}
function remember(name, value) {
  try {
    if (value === null || value === "") localStorage.removeItem(name);
    else localStorage.setItem(name, value);
  } catch { /* private window: works until the page closes */ }
}

let SERVER = SHELL ? (remembered("server") || "") : "";
let key = new URLSearchParams(location.search).get("key") || "";
if (key) remember(`key ${SERVER}`, key);          // the operator address was opened once
else key = remembered(`key ${SERVER}`) || "";

// A server path as an address this page can fetch.
const serverUrl = (path) => (/^(https?:|blob:|data:)/.test(path) ? path : SERVER + path);
const socketUrl = (path) => (SERVER || location.origin).replace(/^http/, "ws") + path;

async function api(path, body, raw) {
  const options = { headers: { "X-Key": key } };
  if (raw) {
    options.method = "POST";
    options.body = raw;
  } else if (body !== undefined) {
    options.method = "POST";
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }
  const response = await fetch(serverUrl(path), options);
  if (response.ok && (response.headers.get("content-type") || "").startsWith("image/")) {
    return response.blob();
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

// "192.168.0.5", "192.168.0.5:8000" or the whole operator address
// "http://192.168.0.5:8000/?key=abc"  ->  {server, key}
function readAddress(text) {
  let value = String(text || "").trim();
  if (!value) return null;
  if (!/^https?:\/\//i.test(value)) value = `http://${value}`;
  let parsed;
  try { parsed = new URL(value); } catch { return null; }
  if (!parsed.hostname) return null;
  const port = parsed.port || (parsed.protocol === "http:" ? "8000" : "");
  return {
    server: `${parsed.protocol}//${parsed.hostname}${port ? `:${port}` : ""}`,
    key: parsed.searchParams.get("key") || "",
  };
}

// A script of our own (webapp/static/vendor), loaded only when needed.
const loadedScripts = {};
function loadScript(src) {
  loadedScripts[src] ||= new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.onload = resolve;
    script.onerror = () => { delete loadedScripts[src]; reject(new Error(`${src} did not load`)); };
    document.head.append(script);
  });
  return loadedScripts[src];
}

// The phone's back camera until a QR code shows up; resolves with its text
// (null when cancelled). The PC shows the code in Settings when it shares.
async function scanQrCode() {
  await loadScript("/static/vendor/jsQR.js");
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("this page has no camera access");
  const stream = await navigator.mediaDevices.getUserMedia(
    { video: { facingMode: { ideal: "environment" } }, audio: false });
  const box = document.createElement("div");
  box.className = "scan";
  box.innerHTML = `
    <video playsinline muted autoplay></video>
    <p>Point the camera at the QR code on the PC<br>
      <small>Project R on the PC: Settings tab</small></p>
    <button type="button">Cancel</button>`;
  document.body.append(box);
  const video = box.querySelector("video");
  video.srcObject = stream;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d", { willReadFrequently: true });
  return new Promise((resolve) => {
    let done = false;
    const finish = (text) => {
      if (done) return;
      done = true;
      stream.getTracks().forEach((track) => track.stop());
      box.remove();
      resolve(text);
    };
    box.querySelector("button").onclick = () => finish(null);
    const look = () => {
      if (done) return;
      if (video.readyState >= 2 && video.videoWidth) {
        const scale = Math.min(1, 720 / Math.max(video.videoWidth, video.videoHeight));
        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const picture = context.getImageData(0, 0, canvas.width, canvas.height);
        const code = window.jsQR(picture.data, canvas.width, canvas.height,
                                 { inversionAttempts: "attemptBoth" });
        if (code && readAddress(code.data)) { finish(code.data); return; }
      }
      requestAnimationFrame(look);
    };
    look();
  });
}

// The iPhone app's first screen: which PC runs the commentary?
function askForServer(message) {
  return new Promise((resolve) => {
    document.querySelector(".connect")?.remove();
    const box = document.createElement("form");
    box.className = "connect";
    box.innerHTML = `
      <div class="connect-card">
        <span class="mark" aria-hidden="true"></span>
        <h2 class="bn">লাইভ ধারাভাষ্য</h2>
        <p>Connect to the PC that runs Project R, on the same Wi-Fi. On the PC, start
          <b>Project R (phones on the Wi-Fi can join)</b>: its Settings tab then shows a QR code.</p>
        <button class="primary" type="button" data-scan>Scan the QR code on the PC</button>
        <p class="or">or type what the PC shows</p>
        <label>PC address
          <input type="text" name="address" inputmode="url" autocapitalize="off" autocorrect="off"
            spellcheck="false" placeholder="192.168.0.5:8000" required></label>
        <label>Operator key <small>(leave empty to only watch)</small>
          <input type="text" name="key" autocapitalize="off" autocorrect="off" spellcheck="false"></label>
        <button type="submit">Connect</button>
        <p class="error" role="alert"></p>
      </div>`;
    const error = box.querySelector(".error");
    error.textContent = message || "";
    box.elements.address.value = SERVER.replace(/^http:\/\//, "");
    box.elements.key.value = key;
    box.querySelector("[data-scan]").onclick = async () => {
      error.textContent = "";
      let text;
      try {
        text = await scanQrCode();
      } catch (problem) {
        error.textContent = `The camera could not scan (${problem.message || problem}). ` +
          "Allow the camera for Project R in the iPhone's Settings, or type the address.";
        return;
      }
      const found = text && readAddress(text);
      if (!found) return;
      box.elements.address.value = found.server.replace(/^http:\/\//, "");
      box.elements.key.value = found.key;
      box.requestSubmit();
    };
    box.onsubmit = async (event) => {
      event.preventDefault();
      const found = readAddress(box.elements.address.value);
      if (!found) { error.textContent = "That does not look like an address."; return; }
      const wanted = found.key || box.elements.key.value.trim();
      error.textContent = "Connecting…";
      try {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 6000);
        const response = await fetch(`${found.server}/api/state`,
          { headers: { "X-Key": wanted }, signal: controller.signal });
        clearTimeout(timer);
        if (!response.ok) throw new Error(response.statusText);
      } catch {
        error.textContent = `No answer from ${found.server}. Was Project R started on the PC ` +
          "with \"phones on the Wi-Fi can join\", and is this phone on the same Wi-Fi?";
        return;
      }
      SERVER = found.server;
      key = wanted;
      remember("server", SERVER);
      remember(`key ${SERVER}`, key);
      box.remove();
      resolve();
    };
    document.body.append(box);
    box.elements.address.focus();
  });
}

// Resolves once the server is known (at once outside the app).
async function findServer() {
  if (!SHELL) return;
  if (SERVER) {
    try {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 4000);
      const response = await fetch(`${SERVER}/api/state`,
        { headers: { "X-Key": key }, signal: controller.signal });
      clearTimeout(timer);
      if (response.ok) return;
    } catch { /* ask again below */ }
    await askForServer(`No answer from ${SERVER}.`);
  } else {
    await askForServer();
  }
}

// In the app: a small "PC: 192.168.0.5:8000 · change" in the header.
function showServer(element) {
  if (!SHELL || !element) return;
  element.hidden = false;
  element.textContent = `PC ${SERVER.replace(/^https?:\/\//, "")} · change`;
  element.onclick = async () => { await askForServer(); location.reload(); };
}

// A link to another page of the app, keeping the operator key.
function pageLink(page) {
  return !SHELL && key ? `${page}?key=${encodeURIComponent(key)}` : page;
}

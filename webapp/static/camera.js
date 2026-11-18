// Project R: this device's camera as a match camera.
//
// The camera's picture is drawn SEND_RATE times a second, packed as a
// JPEG and sent to the PC over a WebSocket (/ws/camera). The PC then
// lists it as "Phone camera: <name>" for Camera 1 or Camera 2.
// (SERVER, key and friends: common.js.)

const $ = (id) => document.getElementById(id);
const SEND_RATE = 15;                 // pictures a second: the PC analyses up to 12 and shows them all
const WAITING_LIMIT = 1_500_000;      // bytes not sent yet: skip pictures, do not queue them

let stream = null;
let socket = null;
let timer = null;
let wakeLock = null;
let wanted = false;                   // the camera should be sending
let sent = 0;
let countedAt = 0;

function setStatus(state, text) {
  $("status").className = `status ${state}`;
  $("statusText").textContent = text;
}
function showError(error) {
  $("error").textContent = error ? (error.message || String(error)) : "";
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("This browser only gives its camera to an https page or to the Project R " +
      "app. On the PC, start the server with --https and open the https address it prints.");
  }
  const width = Number($("size").value);
  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: $("lens").value },
        width: { ideal: width }, height: { ideal: Math.round(width * 9 / 16) },
        frameRate: { ideal: 30 },
      },
    });
  } catch (error) {
    if (error.name === "NotAllowedError") {
      throw new Error("The camera was not allowed. Allow it for this page in the settings and try again.");
    }
    if (error.name === "NotFoundError") throw new Error("No camera found on this device.");
    throw error;
  }
}

function connect() {
  const name = $("name").value.trim() || "Phone";
  const address = socketUrl(`/ws/camera?name=${encodeURIComponent(name)}&key=${encodeURIComponent(key)}`);
  socket = new WebSocket(address);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") return;
    const told = JSON.parse(event.data);
    if (told.name) {
      $("shownName").textContent = told.name;
      setStatus("running", `Sending as “${told.name}”`);
      $("liveBadge").hidden = false;
    }
  };
  socket.onclose = (event) => {
    $("liveBadge").hidden = true;
    if (!wanted) return;
    if (event.code === 4403) {
      stop();
      showError(new Error("Only the operator can add a camera. Open this page from the operator " +
        "address (the one with ?key=…), or type the operator key when connecting."));
      return;
    }
    setStatus("starting", "Lost the PC, connecting again…");
    setTimeout(() => { if (wanted) connect(); }, 1500);
  };
}

function sendPicture() {
  const video = $("view");
  if (!socket || socket.readyState !== WebSocket.OPEN || !video.videoWidth) return;
  if (socket.bufferedAmount > WAITING_LIMIT) return;      // slow Wi-Fi: newest picture wins
  const grab = $("grab");
  const scale = Math.min(1, Number($("size").value) / video.videoWidth);
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  if (grab.width !== width || grab.height !== height) {
    grab.width = width;
    grab.height = height;
  }
  grab.getContext("2d").drawImage(video, 0, 0, width, height);
  grab.toBlob((blob) => {
    if (!blob || !socket || socket.readyState !== WebSocket.OPEN) return;
    socket.send(blob);
    sent += 1;
    const now = performance.now();
    if (now - countedAt >= 2000) {
      $("rate").textContent = `${(sent * 1000 / (now - countedAt)).toFixed(0)} pictures/s`;
      sent = 0;
      countedAt = now;
    }
  }, "image/jpeg", 0.72);
}

async function keepAwake() {
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch { wakeLock = null; }
}

async function start() {
  showError();
  $("start").disabled = true;
  try {
    stream = await openCamera();
  } catch (error) {
    $("start").disabled = false;
    showError(error);
    return;
  }
  remember("camera name", $("name").value.trim());
  const video = $("view");
  video.srcObject = stream;
  await video.play().catch(() => {});
  $("idle").hidden = true;
  $("stop").disabled = false;
  for (const id of ["name", "lens", "size"]) $(id).disabled = true;
  wanted = true;
  sent = 0;
  countedAt = performance.now();
  setStatus("starting", "Connecting to the PC…");
  connect();
  timer = setInterval(sendPicture, 1000 / SEND_RATE);
  keepAwake();
}

function stop() {
  wanted = false;
  clearInterval(timer);
  socket?.close();
  socket = null;
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  $("view").srcObject = null;
  wakeLock?.release().catch(() => {});
  wakeLock = null;
  $("idle").hidden = false;
  $("liveBadge").hidden = true;
  $("start").disabled = false;
  $("stop").disabled = true;
  for (const id of ["name", "lens", "size"]) $(id).disabled = false;
  setStatus("", "Camera off");
}

async function boot() {
  $("name").value = remembered("camera name") || "";
  $("matchPage").href = pageLink("./");
  $("start").onclick = start;
  $("stop").onclick = stop;
  // The screen lock is given back when the page was hidden for a moment.
  document.addEventListener("visibilitychange", () => {
    if (wanted && document.visibilityState === "visible") keepAwake();
  });
  await findServer();
  showServer($("serverName"));
}

boot();

// Project R live commentary: the page.
//
// The server sends, over one WebSocket:
//   text   {type: hello|status|stats|line|audio|note}
//   bytes  1 byte kind (1 picture JPEG of camera 1, 2 match sound s16le
//          16 kHz mono, 3 picture of camera 2, 4 picture of camera 3),
//          8 bytes little-endian seconds into the match feed, then the data
// The operator gets every camera and sees them side by side; viewers and
// the broadcast page (?view=air) get only the camera on air, as kind 1.
// The picture is shown `delay` seconds late; a line is shown and spoken
// when the delayed picture reaches the moment the line is about.
// (SERVER, key, api() and friends: common.js.)

const $ = (id) => document.getElementById(id);

const COLOUR_CSS = {
  red: "#e23b3b", orange: "#ff8c1a", yellow: "#ffd426", green: "#3aa655",
  "sky blue": "#6ebeeb", blue: "#1e5adc", purple: "#963caa", pink: "#ff78b4",
  white: "#f5f5f5", grey: "#969696", black: "#1e1e1e",
};
const TYPE_LABEL = {
  goal: "Goal", foul: "Foul", freekick: "Free kick", goal_kick: "Goal kick",
  corner_kick: "Corner", yellow_card: "Yellow card", red_card: "Red card",
  penalty: "Penalty", penalty_kick: "Penalty", shootout_start: "Shootout",
  shootout_kick: "Shootout kick", big_chance: "Crowd roar", whistle_stop: "Whistle",
  possession_spell: "Possession", pass: "Pass", long_pass: "Long pass",
  run_with_ball: "Run", tackle_won: "Tackle", interception: "Interception",
  attack: "Attack", match_intro: "Teams", filler: "Play", kickoff: "Kick-off",
  half_time: "Half-time", second_half: "Second half", full_time: "Full time",
};
const SOURCE_LABEL = {
  crowd: "from the crowd", whistle: "from the whistle", "crowd+huddle": "roar and celebration",
  possession: "from the ball", stillness: "players standing still", rules: "from the ball",
  operator: "operator", teams: "kit colours", filler: "",
};
const BIG = new Set(["goal", "penalty", "penalty_kick", "shootout_start", "shootout_kick",
  "kickoff", "half_time", "second_half", "full_time"]);
const QUIET = new Set(["filler", "pass", "long_pass", "run_with_ball", "possession_spell"]);
const BN_DIGITS = "০১২৩৪৫৬৭৮৯";
const bn = (n) => String(n).replace(/\d/g, (d) => BN_DIGITS[d]);

let operator = false;
let stats = null;
let colours = {};
let sessionId = null;

// ---------------------------------------------------------------- server

function showError(error) {
  $("error").textContent = error ? (error.message || String(error)) : "";
}

// ---------------------------------------------------------------- timing

// The operator sees the match as it happens (they press the buttons on
// time); viewers see it late, with each line on its moment. Each keeps
// their own setting.
// Seconds always held back to smooth the pictures: a little over Wi-Fi, next
// to nothing for the operator on the PC that runs the commentary.
const LOCAL = !SHELL && ["127.0.0.1", "localhost", "[::1]"].includes(location.hostname);
let JITTER = 0.4;
const delayKey = () => (BROADCAST ? "delay broadcast" : operator ? "delay operator" : "delay");
let delay = 10;
let base = null;            // performance seconds at match second 0, delay included
let newest = 0;
const nowSeconds = () => performance.now() / 1000;
const playhead = () => (base === null ? -1 : nowSeconds() - base);

function retime(t) {
  // Keep the picture `delay` seconds behind what arrives. A jump (start,
  // a slow PC working through a video, a sleeping tab) moves the clock.
  const target = t - Math.max(delay, JITTER);
  if (base === null || Math.abs(playhead() - target) > 1.5) {
    base = nowSeconds() - target;
    audioBase = null;
  }
}

// ---------------------------------------------------------------- picture

// Up to 3 cameras. The operator sees them side by side and chooses which
// one is on air; a double-click makes one fill the picture, and another
// double-click brings them all back.
const views = $("views");
const viewList = [...views.querySelectorAll(".view")];
const cams = viewList.map((view) => {
  const target = view.querySelector("canvas");
  return { view, canvas: target, context: target.getContext("2d"), frames: [], drawn: -1 };
});
const BROADCAST = new URLSearchParams(location.search).get("view") === "air";
let cameraCount = 1;
let onAir = 1;

function onPicture(t, blob, camera = 1) {
  if (camera === 1) {
    newest = t;
    retime(t);
    $("idle").hidden = true;
  } else if (base === null) {
    return;                           // camera 1 sets the clock
  }
  const list = cams[camera - 1].frames;
  list.push({ t, blob });
  if (list.length > 1200) list.splice(0, list.length - 1200);
}

function setFullView(camera) {
  if (camera && cameraCount > 1) views.dataset.full = String(camera);
  else delete views.dataset.full;
  viewList.forEach((view, index) => {
    view.title = cameraCount < 2 ? "" : views.dataset.full
      ? "Double-click: all cameras side by side"
      : `Camera ${index + 1}. Double-click: full view`;
  });
}

function setCameras(cameras, air) {
  const count = Math.max(1, Math.min(3, (cameras && cameras.length) || 1));
  if (count !== cameraCount) {
    cameraCount = count;
    views.dataset.count = String(count);
    cams.forEach((cam, index) => {
      if (index >= count) {
        cam.frames = [];
        cam.context.clearRect(0, 0, cam.canvas.width, cam.canvas.height);
      }
    });
    if (count < 2 || Number(views.dataset.full) > count) setFullView(null);
    else setFullView(Number(views.dataset.full) || null);
  }
  onAir = air || 1;
  viewList.forEach((view, index) => {
    const number = index + 1;
    const camera = (cameras || [])[index];
    view.hidden = number > count;
    view.classList.toggle("on-air", count > 1 && number === onAir);
    view.querySelector(".view-name").textContent = camera
      ? `Camera ${number}${camera.name ? ` · ${camera.name}` : ""}${camera.fps ? ` · ${Math.round(camera.fps)}/s` : ""}`
      : `Camera ${number}`;
    view.querySelector(".view-air").textContent = camera && camera.error
      ? "Stopped" : number === onAir ? "On air" : "Put on air";
    view.querySelector(".view-air").disabled = Boolean(camera && camera.error) || number === onAir;
  });
}

async function putOnAir(number) {
  if (!operator || number > cameraCount || number === onAir) return;
  try {
    await api("/api/operator", { action: "on_air", camera: number });
    toast(`Camera ${bn(number)} on air`);
  } catch (error) {
    toast(error.message || String(error));
  }
}

function draw(target, targetContext, blob) {
  createImageBitmap(blob).then((image) => {
    if (target.width !== image.width || target.height !== image.height) {
      target.width = image.width;
      target.height = image.height;
    }
    targetContext.drawImage(image, 0, 0);
    image.close();
  }).catch(() => {});
}

function render() {
  requestAnimationFrame(render);
  const at = playhead();
  if (base === null) return;
  cams.forEach((cam) => {
    let pick = null;
    while (cam.frames.length && cam.frames[0].t <= at) pick = cam.frames.shift();
    if (pick && pick.t !== cam.drawn) {
      cam.drawn = pick.t;
      cam.shown = (cam.shown || 0) + 1;
      draw(cam.canvas, cam.context, pick.blob);
    }
  });
  releaseLines(at);
  releaseScoreboard(at);
}
requestAnimationFrame(render);

// ---------------------------------------------------------------- sound

let audio = null;           // AudioContext, made on the first click
let matchGain = null;
let audioBase = null;       // audio clock at match second 0
let soundWanted = false;

function soundOn() {
  if (!audio) {
    audio = new (window.AudioContext || window.webkitAudioContext)();
    matchGain = audio.createGain();
    matchGain.gain.value = $("matchVolume").value / 100;
    matchGain.connect(audio.destination);
  }
  audio.resume();
  soundWanted = true;
  $("soundOn").hidden = true;
}

function onMatchSound(t, buffer) {
  if (!audio || !soundWanted || base === null) return;
  if (stats && !stats.live_source && !stats.realtime) return;   // slow test run: no sound
  if (audioBase === null) audioBase = audio.currentTime - playhead();
  const when = audioBase + t;
  if (when < audio.currentTime) return;
  const pcm = new Int16Array(buffer);
  if (!pcm.length) return;
  const clip = audio.createBuffer(1, pcm.length, 16000);
  const data = clip.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) data[i] = pcm[i] / 32768;
  const node = audio.createBufferSource();
  node.buffer = clip;
  node.connect(matchGain);
  node.start(when);
}

// ---------------------------------------------------------------- lines

const waiting = [];         // lines not yet reached by the delayed picture
const byId = new Map();     // id -> {line, element}
let speaking = null;
const speech = [];          // lines to speak, in order
let captionTimer = null;

function onLine(line, old) {
  if (byId.has(line.id)) return;
  byId.set(line.id, { line, element: null });
  if (old) showLine(line, true);
  else waiting.push(line);
}

function releaseLines(at) {
  while (waiting.length && waiting[0].time <= Math.max(at, 0)) showLine(waiting.shift(), false);
  // A line the system only knew late (its moment already passed) comes at once.
  for (let i = 0; i < waiting.length; i++) {
    if (waiting[i].time <= at) showLine(waiting.splice(i--, 1)[0], false);
  }
}

function showLine(line, old) {
  const entry = byId.get(line.id);
  const item = document.createElement("li");
  if (BIG.has(line.type)) item.classList.add("big");
  if (QUIET.has(line.type)) item.classList.add("quiet");
  if (line.type === "yellow_card") item.classList.add("card-yellow");
  if (line.type === "red_card") item.classList.add("card-red");
  if (line.type === "whistle_stop" || line.type === "foul") item.classList.add("whistle");
  const minute = document.createElement("span");
  minute.className = "min";
  minute.textContent = bn(line.minute) + "'";
  const body = document.createElement("div");
  const text = document.createElement("p");
  text.textContent = line.text;
  const meta = document.createElement("small");
  const parts = [TYPE_LABEL[line.type] || line.type];
  if (SOURCE_LABEL[line.source]) parts.push(SOURCE_LABEL[line.source]);
  if (line.source !== "operator" && line.source !== "filler" && line.delay >= 0.5) {
    parts.push(`known ${line.delay.toFixed(0)} s after`);
  }
  meta.textContent = parts.join(" · ");
  body.append(text, meta);
  item.append(minute, body);
  $("feed").prepend(item);
  entry.element = item;
  $("feedEmpty").hidden = true;
  $("lineCount").textContent = `${byId.size} line${byId.size === 1 ? "" : "s"}`;
  if (old) return;
  showCaption(line.text, 5000);
  speech.push({ line, since: nowSeconds() });
  speakNext();
}

function showCaption(text, milliseconds) {
  $("captionText").textContent = text;
  $("caption").hidden = false;
  clearTimeout(captionTimer);
  captionTimer = setTimeout(() => { $("caption").hidden = true; }, milliseconds);
}

function onVoice(id, url) {
  const entry = byId.get(id);
  if (entry) entry.line.audio = url;
  speakNext();
}

function speakNext() {
  if (speaking || !soundWanted) { if (!soundWanted) speech.length = 0; return; }
  while (speech.length) {
    const next = speech[0];
    const age = nowSeconds() - next.since;
    // Play lines give way when the voice is behind; nothing waits for ever.
    if ((QUIET.has(next.line.type) && speech.length > 2) || age > 20) { speech.shift(); continue; }
    if (!next.line.audio) {
      if (age > 8) { speech.shift(); continue; }
      setTimeout(speakNext, 300);
      return;
    }
    speech.shift();
    const entry = byId.get(next.line.id);
    const player = new Audio(serverUrl(next.line.audio));
    player.volume = $("voiceVolume").value / 100;
    speaking = player;
    const done = () => {
      if (speaking !== player) return;
      speaking = null;
      entry?.element?.classList.remove("speaking");
      speakNext();
    };
    player.onended = done;
    player.onerror = done;
    player.onloadedmetadata = () => {
      if (isFinite(player.duration)) showCaption(next.line.text, player.duration * 1000 + 800);
    };
    entry?.element?.classList.add("speaking");
    player.play().catch(done);
    return;
  }
}

// ---------------------------------------------------------------- state

function kit(element, colour) {
  element.style.background = COLOUR_CSS[colour] || "";
}

// The operator pressed Stop, or the camera broke: what this page was still
// holding back (the picture runs `delay` seconds late) is dropped, so
// nothing more is shown or said once there is no match feed.
function cutNow() {
  cams.forEach((cam) => { cam.frames = []; });
  if (scoreQueue.length) showScoreboard(scoreQueue[scoreQueue.length - 1]);
  scoreQueue.length = 0;
  base = null;
  audioBase = null;
  waiting.length = 0;
  speech.length = 0;
  if (speaking) {
    const player = speaking;
    speaking = null;
    player.pause();
  }
  document.querySelectorAll("#feed .speaking").forEach((item) => item.classList.remove("speaking"));
  clearTimeout(captionTimer);
  $("caption").hidden = true;
  cams.forEach((cam) => cam.context.clearRect(0, 0, cam.canvas.width, cam.canvas.height));
  setCameras(null, 1);
  $("idle").hidden = false;
}

// ---------------------------------------------------------------- scoreboard

const scoreQueue = [];      // stats not yet reached by the delayed picture

function releaseScoreboard(at) {
  let latest = null;
  while (scoreQueue.length && scoreQueue[0].clock <= at) latest = scoreQueue.shift();
  if (latest) showScoreboard(latest);
}

function showScoreboard(next) {
  const state = next.state;
  const teams = next.teams || {};
  const name = (team, fallback) => (team && team.colour_bn ? `${team.colour_bn} দল` : fallback);
  $("nameA").textContent = name(teams.team_a, "দল ক");
  $("nameB").textContent = name(teams.team_b, "দল খ");
  $("labelA").textContent = name(teams.team_a, "Team A");
  $("labelB").textContent = name(teams.team_b, "Team B");
  for (const [team, ids] of [["team_a", ["kitA", "kitA2"]], ["team_b", ["kitB", "kitB2"]]]) {
    const colour = teams[team]?.colour;
    ids.forEach((id) => kit($(id), colour));
    document.querySelectorAll(`select.colour[data-team=${team}]`).forEach((select) => {
      if (colour && document.activeElement !== select) select.value = colour;
    });
  }
  $("scoreA").textContent = bn(next.score.team_a);
  $("scoreB").textContent = bn(next.score.team_b);
  $("minute").textContent = state === "starting" ? "–" : bn(next.minute) + "'";
  $("phase").textContent = {
    play: "in play", before: "before kick-off", break: "break",
    shootout: "shootout", full_time: "full time", hold: "on hold",
  }[next.phase] || next.phase;

}

let lastShown = null;

function onStats(next) {
  const before = stats;
  stats = next;
  if (next.session !== sessionId) newSession(next.session);
  const state = next.state;
  $("status").className = `status ${state}`;
  $("statusText").textContent = next.message;
  const running = state === "running" || state === "starting";
  if (running) $("feedNote").hidden = true;
  if (!running && next.cut && !(before && before.session === next.session && before.cut && before.state === state)) {
    cutNow();
    $("idleText").textContent = state === "failed" ? next.message : "Stopped. Nothing is being said.";
  }
  // Why nothing is said just now: no picture, a black one, nobody in view.
  $("hold").hidden = !next.hold;
  $("hold").textContent = next.hold || "";
  // Viewers and the broadcast get one picture: the camera on air.
  setCameras(operator && !BROADCAST && running ? next.cameras : null, next.on_air);
  $("onAirText").textContent = next.cameras && next.cameras.length > 1
    ? `camera ${bn(next.on_air || 1)} on air` : "";
  $("liveBadge").hidden = state !== "running";
  $("liveText").textContent = next.live_source ? "LIVE" : "PLAYING AS LIVE";
  $("lagText").textContent = delay ? `${delay} s behind` : "";
  if (!running && !cams[0].frames.length && state !== "ended") $("idle").hidden = false;
  $("start").disabled = running;
  $("stop").disabled = !running;

  // The score and the clock go with the picture: a page that shows the
  // match 10 s late shows a goal 10 s late too, not when the button is pressed.
  if (base !== null && next.clock !== undefined && Math.max(delay, JITTER) > 0.5) {
    scoreQueue.push(next);
    if (scoreQueue.length > 600) scoreQueue.splice(0, scoreQueue.length - 600);
  } else {
    scoreQueue.length = 0;
    showScoreboard(next);
  }

  $("tRate").textContent = next.rate ? next.rate.toFixed(1) : "–";
  $("tRate").parentElement.classList.toggle("warn", next.rate > 0 && next.rate < 0.8 * next.target_rate);
  $("tPlayers").textContent = next.players.toFixed(0);
  $("tBall").textContent = Math.round(next.ball_share * 100) + "%";
  $("tWhistles").textContent = next.sound ? next.whistles : "no sound";
  $("tRoars").textContent = next.sound ? next.roars : "–";
  // How many pictures a second the camera gives and this page shows: a
  // slow picture is the camera's doing when the first number is low.
  const camera = (next.cameras || [])[0];
  const now = nowSeconds();
  const shown = lastShown ? (cams[0].shown - lastShown.count) / Math.max(0.5, now - lastShown.at) : 0;
  lastShown = { count: cams[0].shown || 0, at: now };
  $("tCamera").textContent = camera && camera.fps ? camera.fps.toFixed(0) : "–";
  $("tShown").textContent = running && shown ? `shown ${shown.toFixed(0)}` : "pictures/s";
  $("tCamera").parentElement.classList.toggle("warn", Boolean(camera && camera.fps && camera.fps < 14));
}

function newSession(id) {
  sessionId = id;
  cams.forEach((cam) => { cam.frames = []; });
  scoreQueue.length = 0;
  setCameras(null, 1);
  $("feedNote").hidden = true;
  base = null;
  audioBase = null;
  waiting.length = 0;
  speech.length = 0;
  byId.clear();
  $("feed").replaceChildren();
  $("notes").replaceChildren();
  $("feedEmpty").hidden = false;
  $("lineCount").textContent = "0 lines";
  $("caption").hidden = true;
  stopFinished();
}

function onNote(text) {
  const item = document.createElement("li");
  item.textContent = text;
  $("notes").append(item);
}

function onStatus(message) {
  $("status").className = `status ${message.state}`;
  $("statusText").textContent = message.message;
  if (message.state === "starting") {
    $("idleText").textContent = message.message + "…";
    // The camera's picture shows while the models load: say why nothing is said yet.
    $("hold").hidden = false;
    $("hold").textContent = message.message;
  }
}

// ---------------------------------------------------------------- over the picture

let toastTimer = null;
function toast(text) {
  $("toast").textContent = text;
  $("toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 1800);
}

// A side of the match control opens and closes like a mini player; each
// remembers how it was left.
function setOverlay(box, open) {
  box.dataset.open = String(open);
  box.querySelector(".ov-toggle").setAttribute("aria-expanded", String(open));
  if (box.classList.contains("ov-clock")) $("stage").classList.toggle("clock-open", open);
}

function wireOverlays() {
  const narrow = window.matchMedia("(max-width: 700px)").matches;
  document.querySelectorAll(".ov").forEach((box, number) => {
    const name = `overlay ${number}`;
    const saved = remembered(name);
    setOverlay(box, saved === null ? (!narrow && !box.classList.contains("ov-clock")) : saved === "1");
    box.querySelector(".ov-toggle").onclick = () => {
      const open = box.dataset.open !== "true";
      setOverlay(box, open);
      remember(name, open ? "1" : "0");
    };
  });
  // Full screen: the buttons stay on the picture.
  const stage = $("stage");
  if (stage.requestFullscreen) {
    $("fullscreen").hidden = false;
    $("fullscreen").onclick = () => {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
      else stage.requestFullscreen().catch(() => {});
    };
    document.addEventListener("fullscreenchange", () => {
      $("fullscreen").classList.toggle("on", document.fullscreenElement === stage);
    });
  }
}

function choosePanel(name) {
  document.querySelectorAll(".panel-tabs button").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.panel === name));
  });
  document.querySelectorAll("[data-panel-body]").forEach((body) => {
    body.hidden = body.dataset.panelBody !== name;
  });
}

// ---------------------------------------------------------------- socket

function connect() {
  // The key tells the server this is the operator (all cameras); the
  // broadcast page asks for the camera on air only.
  const query = new URLSearchParams();
  if (key) query.set("key", key);
  if (BROADCAST) query.set("view", "air");
  const socket = new WebSocket(socketUrl(`/ws${query.toString() ? `?${query}` : ""}`));
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") {
      const view = new DataView(event.data);
      const kind = view.getUint8(0);
      const t = view.getFloat64(1, true);
      const camera = { 1: 1, 3: 2, 4: 3 }[kind];
      if (camera) onPicture(t, new Blob([new Uint8Array(event.data, 9)], { type: "image/jpeg" }), camera);
      else if (kind === 2) onMatchSound(t, event.data.slice(9));
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      onStats(message.stats);
      message.lines.forEach((line) => onLine(line, true));
      message.notes.forEach(onNote);
      // Lines of a match that is over are history, not commentary.
      const over = message.stats.state !== "running" && message.stats.state !== "starting";
      $("feedNote").hidden = !(over && message.lines.length);
    } else if (message.type === "cameras") {
      if (operator) loadCameras();
    } else if (message.type === "stats") onStats(message);
    else if (message.type === "line") onLine(message.line, false);
    else if (message.type === "audio") onVoice(message.id, message.url);
    else if (message.type === "note") onNote(message.text);
    else if (message.type === "status") onStatus(message);
  };
  socket.onclose = () => {
    if (!stats || stats.state !== "running") $("statusText").textContent = "Reconnecting…";
    setTimeout(connect, 1500);
  };
}

// ---------------------------------------------------------------- finished matches

let finishedLines = [];
let finishedShown = 0;

async function playFinished() {
  showError();
  try {
    const name = $("videoName").value;
    const data = await api(`/api/finished/${encodeURIComponent(name)}`);
    newSession(null);
    finishedLines = data.lines.map((line, i) => ({ ...line, id: `f${i}`, source: "", delay: 0 }));
    finishedShown = 0;
    const video = $("finishedVideo");
    video.src = serverUrl(data.video);
    video.hidden = false;
    views.hidden = true;
    $("idle").hidden = true;
    $("statusText").textContent = `Finished commentary: ${name}` + (data.spoken ? "" : " (text only)");
    video.play().catch(() => {});
  } catch (error) { showError(error); }
}

function stopFinished() {
  const video = $("finishedVideo");
  if (video.hidden) return;
  video.pause();
  video.removeAttribute("src");
  video.hidden = true;
  views.hidden = false;
  finishedLines = [];
}

$("finishedVideo").addEventListener("timeupdate", () => {
  const at = $("finishedVideo").currentTime;
  while (finishedShown < finishedLines.length && finishedLines[finishedShown].time <= at) {
    const line = finishedLines[finishedShown++];
    byId.set(line.id, { line, element: null });
    showLine(line, true);
    showCaption(line.text, 5000);
  }
});
$("finishedVideo").addEventListener("seeked", () => {
  const at = $("finishedVideo").currentTime;
  $("feed").replaceChildren();
  byId.clear();
  finishedShown = 0;
  while (finishedShown < finishedLines.length && finishedLines[finishedShown].time <= at) {
    const line = finishedLines[finishedShown++];
    byId.set(line.id, { line, element: null });
    showLine(line, true);
  }
});

// ---------------------------------------------------------------- controls

let sourceKind = "camera";

function chooseTab(kind) {
  sourceKind = kind;
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.kind === kind));
  });
  document.querySelectorAll(".tab").forEach((tab) => { tab.hidden = tab.dataset.for !== kind; });
}

function fillVideos(videos, chosen) {
  const select = $("videoName");
  select.replaceChildren(...videos.map((video) => {
    const option = document.createElement("option");
    option.value = video.name;
    option.textContent = `${video.name} (${video.megabytes} MB)` + (video.finished ? " · finished" : "");
    option.dataset.finished = video.finished ? "1" : "";
    return option;
  }));
  if (!videos.length) {
    const option = document.createElement("option");
    option.textContent = "No videos in data/videos yet";
    option.value = "";
    select.append(option);
  }
  if (chosen) select.value = chosen;
  finishedButton();
}

function finishedButton() {
  const option = $("videoName").selectedOptions[0];
  $("playFinished").hidden = !(option && option.dataset.finished);
}

function seconds(text) {
  const parts = String(text || "").trim().split(":").map(Number);
  if (parts.some(Number.isNaN)) return 0;
  return parts.length === 2 ? parts[0] * 60 + parts[1] : parts[0] || 0;
}

// ---- cameras: which device is camera 1, 2 and 3

const slots = [...document.querySelectorAll(".camera-slot")];   // camera 1, 2, 3
const ADDRESS = "address";            // "a network camera: type its address"
let cameraList = [];
const chosen = (() => {
  try { return JSON.parse(remembered("cameras")) || {}; } catch { return {}; }
})();                                 // {count, 1: {device, address}, 2: {...}}

const slotParts = (slot) => ({
  device: slot.querySelector(".device"),
  address: slot.querySelector(".address"),
  image: slot.querySelector("img"),
});

function saveCameras() {
  slots.forEach((slot) => {
    const { device, address } = slotParts(slot);
    chosen[slot.dataset.slot] = { device: device.value, address: address.value };
  });
  chosen.count = Number($("cameraCount").value);
  remember("cameras", JSON.stringify(chosen));
}

function fillCameras(list) {
  cameraList = list;
  const taken = new Set();
  slots.forEach((slot) => {
    const { device, address } = slotParts(slot);
    const wanted = device.value || chosen[slot.dataset.slot]?.device;
    device.replaceChildren(...list.map((camera) => {
      const option = document.createElement("option");
      option.value = camera.id;
      option.textContent = camera.label;
      return option;
    }));
    const other = document.createElement("option");
    other.value = ADDRESS;
    other.textContent = "A network camera (type its address)";
    device.append(other);
    const values = [...device.options].map((option) => option.value);
    // Keep what was chosen; otherwise the first device the other camera does not have.
    device.value = values.includes(wanted) && (wanted === ADDRESS || !taken.has(wanted))
      ? wanted : (values.find((value) => value !== ADDRESS && !taken.has(value)) || ADDRESS);
    taken.add(device.value);
    device.dataset.was = device.value;
    if (!address.value) address.value = chosen[slot.dataset.slot]?.address || "";
    address.hidden = device.value !== ADDRESS;
  });
}

async function loadCameras() {
  try { fillCameras((await api("/api/cameras")).cameras); } catch (error) { showError(error); }
}

function clearPreview(slot) {
  const { image } = slotParts(slot);
  if (image.src) URL.revokeObjectURL(image.src);
  image.removeAttribute("src");
  image.hidden = true;
}

const usedSlots = () => slots.slice(0, Number($("cameraCount").value) || 1);

function onDeviceChosen(slot) {
  const mine = slotParts(slot);
  // Choosing another camera's device hands that camera this one's: a swap.
  const otherSlot = usedSlots().find((candidate) => candidate !== slot &&
    mine.device.value !== ADDRESS && slotParts(candidate).device.value === mine.device.value);
  if (otherSlot) {
    const theirs = slotParts(otherSlot);
    theirs.device.value = mine.device.dataset.was;
    theirs.device.dataset.was = theirs.device.value;
    theirs.address.hidden = theirs.device.value !== ADDRESS;
    clearPreview(otherSlot);
  }
  mine.device.dataset.was = mine.device.value;
  mine.address.hidden = mine.device.value !== ADDRESS;
  clearPreview(slot);
  saveCameras();
}

function swapCameras() {
  const [one, two] = slots.map(slotParts);
  [one.device.value, two.device.value] = [two.device.value, one.device.value];
  [one.address.value, two.address.value] = [two.address.value, one.address.value];
  slots.forEach((slot) => {
    const { device, address } = slotParts(slot);
    device.dataset.was = device.value;
    address.hidden = device.value !== ADDRESS;
    clearPreview(slot);
  });
  saveCameras();
}

function setCameraCount(count) {
  $("cameraCount").value = String(count);
  slots.forEach((slot, index) => { slot.hidden = index >= count; });
  $("swapCameras").hidden = count < 2;
  // Two cameras cannot be the same device.
  const taken = new Set();
  slots.slice(0, count).forEach((slot) => {
    const parts = slotParts(slot);
    if (parts.device.value !== ADDRESS && taken.has(parts.device.value)) {
      const free = [...parts.device.options].map((option) => option.value)
        .find((value) => value !== ADDRESS && !taken.has(value));
      parts.device.value = free || ADDRESS;
      parts.device.dataset.was = parts.device.value;
      parts.address.hidden = parts.device.value !== ADDRESS;
    }
    taken.add(parts.device.value);
  });
}

function cameraSource(slot) {
  const { device, address } = slotParts(slot);
  const value = device.value;
  if (value === ADDRESS) return { kind: "url", url: address.value };
  const [kind, id] = [value.slice(0, value.indexOf(":")), value.slice(value.indexOf(":") + 1)];
  return kind === "phone" ? { kind, id } : { kind: "camera", index: Number(id) };
}

async function lookAtCamera(slot) {
  showError();
  const { image } = slotParts(slot);
  const button = slot.querySelector(".look");
  button.disabled = true;
  button.textContent = "Opening…";
  try {
    const picture = await api("/api/camera/preview", cameraSource(slot));
    clearPreview(slot);
    image.src = URL.createObjectURL(picture);
    image.hidden = false;
  } catch (error) {
    clearPreview(slot);
    showError(error);
  } finally {
    button.disabled = false;
    button.textContent = "Show";
  }
}

async function start() {
  showError();
  const source = { kind: sourceKind };
  let others = [];
  if (sourceKind === "camera") {
    Object.assign(source, cameraSource(slots[0]));
    others = usedSlots().slice(1).map(cameraSource);
    saveCameras();
  }
  if (sourceKind === "url") source.url = $("streamUrl").value;
  if (sourceKind === "file") source.name = $("videoName").value;
  const settings = {
    voice: $("voice").value,
    speaker: $("speaker").value,
    process_fps: Number($("processFps").value),
    people_model: $("peopleModel").value,
    boxes: $("boxes").checked,
    realtime: $("realtime").checked,
    mic: $("mic").value,
    start: sourceKind === "file" ? seconds($("startAt").value) : 0,
    need_players: Number($("needPlayers").value),
  };
  soundOn();
  try {
    $("start").disabled = true;
    await api("/api/start", { source, others, settings });
  } catch (error) {
    $("start").disabled = false;
    showError(error);
  }
}

function upload(file) {
  if (!file) return;
  showError();
  const form = new FormData();
  form.append("file", file);
  const request = new XMLHttpRequest();
  const bar = $("uploadProgress");
  bar.hidden = false;
  request.upload.onprogress = (event) => { bar.value = (100 * event.loaded) / event.total; };
  request.onload = () => {
    bar.hidden = true;
    const data = JSON.parse(request.responseText || "{}");
    if (request.status !== 200) return showError(new Error(data.detail || "Upload failed"));
    fillVideos(data.videos, data.name);
  };
  request.onerror = () => { bar.hidden = true; showError(new Error("Upload failed")); };
  request.open("POST", serverUrl("/api/upload"));
  request.setRequestHeader("X-Key", key);
  request.send(form);
}

function wire() {
  document.querySelectorAll(".tabs button").forEach((button) => {
    button.onclick = () => chooseTab(button.dataset.kind);
  });
  $("start").onclick = start;
  $("stop").onclick = () => api("/api/stop", {}).catch(showError);
  $("uploadFile").onchange = (event) => upload(event.target.files[0]);
  $("videoName").onchange = finishedButton;
  $("playFinished").onclick = () => { soundOn(); playFinished(); };
  $("soundOn").onclick = soundOn;

  slots.forEach((slot) => {
    const { device, address } = slotParts(slot);
    device.onchange = () => onDeviceChosen(slot);
    address.onchange = saveCameras;
    slot.querySelector(".look").onclick = () => lookAtCamera(slot);
  });
  $("cameraCount").onchange = (event) => { setCameraCount(Number(event.target.value)); saveCameras(); };
  $("swapCameras").onclick = swapCameras;
  $("refreshCameras").onclick = loadCameras;
  $("copyBroadcast").onclick = () => {
    $("broadcastUrl").select();
    navigator.clipboard?.writeText($("broadcastUrl").value)
      .then(() => toast("Broadcast address copied"))
      .catch(() => document.execCommand && document.execCommand("copy"));
  };
  $("cameraPage").href = pageLink("camera.html");
  // The cameras side by side: a double-click fills the picture with one,
  // the next double-click shows them all again. "Put on air" (or the keys
  // 1, 2, 3) chooses what viewers and the stream see.
  viewList.forEach((view, index) => {
    view.ondblclick = (event) => {
      if (event.target.closest("button")) return;
      setFullView(views.dataset.full ? null : index + 1);
    };
    view.querySelector(".view-air").onclick = () => putOnAir(index + 1);
  });
  document.addEventListener("keydown", (event) => {
    if (!operator || BROADCAST || cameraCount < 2 || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest && event.target.closest("input, select, textarea")) return;
    if (["1", "2", "3"].includes(event.key)) putOnAir(Number(event.key));
    else if (event.key === "Escape" && views.dataset.full) setFullView(null);
  });

  document.querySelectorAll(".ov [data-action]").forEach((button) => {
    button.onclick = async () => {
      showError();
      const box = button.closest("[data-team]");
      const team = box?.dataset.team;
      const who = box ? box.querySelector(".ov-name").textContent + ": " : "";
      button.disabled = true;
      try {
        await api("/api/operator", { action: button.dataset.action, team });
        toast(`${who}${button.dataset.say || button.textContent}`);
      } catch (error) {
        toast(error.message || String(error));
        showError(error);
      } finally {
        button.disabled = false;
      }
    };
  });
  document.querySelectorAll(".panel-tabs button").forEach((button) => {
    button.onclick = () => choosePanel(button.dataset.panel);
  });
  wireOverlays();
  document.querySelectorAll("select.colour").forEach((select) => {
    select.onchange = () => api("/api/operator", {
      action: "colour", team: select.dataset.team, colour: select.value,
    }).catch(showError);
  });

  $("delay").oninput = (event) => {
    delay = Number(event.target.value);
    remember(delayKey(), String(delay));
    $("delayOut").textContent = `${delay} s`;
    if (newest) { base = null; retime(newest); }
    if (stats) $("lagText").textContent = delay ? `${delay} s behind` : "";
  };
  $("matchVolume").oninput = (event) => { if (matchGain) matchGain.gain.value = event.target.value / 100; };
  $("voiceVolume").oninput = (event) => { if (speaking) speaking.volume = event.target.value / 100; };
}

// The page is one screen high: the header's height is taken off.
function fitHeader() {
  const header = document.querySelector(".top");
  if (header) document.documentElement.style.setProperty("--top", `${header.offsetHeight}px`);
}

async function boot() {
  fitHeader();
  window.addEventListener("resize", fitHeader);
  wire();
  await findServer();               // the iPhone app asks for the PC's address
  showServer($("serverName"));
  try {
    const state = await api("/api/state");
    operator = state.operator;
    colours = state.colours;
    // The broadcast page (?view=air): only the camera on air, the score and
    // the lines, for a stream (OBS "browser source") or a big screen.
    document.body.classList.toggle("broadcast", BROADCAST);
    document.body.classList.toggle("viewer", !operator || BROADCAST);
    if (!operator || BROADCAST) choosePanel("view");
    delay = Number(remembered(delayKey()) ?? (operator && !BROADCAST ? 0 : 10));
    if (operator && !BROADCAST && LOCAL) JITTER = 0.12;
    $("delay").value = delay;
    $("delayOut").textContent = `${delay} s`;
    if (operator) {
      $("delayHint").textContent = "You see the match as it happens, so the buttons are pressed on time. " +
        "Lines come a few seconds after their moment. Viewers on other devices see it 10 s late, " +
        "with each line on its moment.";
    }
    document.querySelectorAll("select.colour").forEach((select) => {
      select.replaceChildren(...Object.entries(colours).map(([name, bengali]) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = `${bengali} (${name})`;
        return option;
      }));
    });
    fillVideos(state.videos);
    fillCameras(state.cameras || []);
    setCameraCount([2, 3].includes(chosen.count) ? chosen.count : 1);
    // What to put in OBS (or open on a big screen) for the stream.
    const air = `${state.share ? state.share.url : (SERVER || location.origin)}/?view=air`;
    $("broadcastUrl").value = air;
    if (state.mics) {
      $("mics").replaceChildren(...state.mics.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        return option;
      }));
    }
    if (state.share) {
      // What a phone on the same Wi-Fi needs to join (the iPhone app asks for both).
      $("share").hidden = false;
      $("share").textContent = `Phones and other PCs on this Wi-Fi: viewers open ${state.share.url} · ` +
        `the iPhone app and the operator use address ${state.share.address} with key ${state.share.key}`;
    }
    if (!operator) $("idleText").textContent = "The commentary starts when the operator starts the match.";
    $("soundOn").hidden = false;
    if (BROADCAST) {
      $("idleText").textContent = "The stream picture: the camera on air shows here when the match starts.";
      soundOn();                      // OBS plays sound without a click; a browser may still ask
      if (audio && audio.state === "running") $("soundOn").hidden = true;
      else $("soundOn").hidden = false;
    }
  } catch (error) { showError(error); }
  connect();
}

boot();

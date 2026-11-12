// Project R live commentary: the page.
//
// The server sends, over one WebSocket:
//   text   {type: hello|status|stats|line|audio|note}
//   bytes  1 byte kind (1 picture JPEG, 2 match sound s16le 16 kHz mono),
//          8 bytes little-endian seconds into the match feed, then the data
// The picture is shown `delay` seconds late; a line is shown and spoken
// when the delayed picture reaches the moment the line is about.

const $ = (id) => document.getElementById(id);
const key = new URLSearchParams(location.search).get("key") || "";

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
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

function showError(error) {
  $("error").textContent = error ? (error.message || String(error)) : "";
}

// ---------------------------------------------------------------- timing

let delay = Number(localStorage.getItem("delay") ?? 10);
let base = null;            // performance seconds at match second 0, delay included
let frames = [];            // {t, blob}, oldest first
let newest = 0;
let drawn = -1;
const nowSeconds = () => performance.now() / 1000;
const playhead = () => (base === null ? -1 : nowSeconds() - base);

function retime(t) {
  // Keep the picture `delay` seconds behind what arrives. A jump (start,
  // a slow PC working through a video, a sleeping tab) moves the clock.
  const target = t - delay;
  if (base === null || Math.abs(playhead() - target) > 1.5) {
    base = nowSeconds() - target;
    audioBase = null;
  }
}

// ---------------------------------------------------------------- picture

const canvas = $("picture");
const context = canvas.getContext("2d");

function onPicture(t, blob) {
  newest = t;
  retime(t);
  frames.push({ t, blob });
  if (frames.length > 1200) frames.splice(0, frames.length - 1200);
  $("idle").hidden = true;
}

function render() {
  requestAnimationFrame(render);
  const at = playhead();
  if (base === null) return;
  let pick = null;
  while (frames.length && frames[0].t <= at) pick = frames.shift();
  if (pick && pick.t !== drawn) {
    drawn = pick.t;
    createImageBitmap(pick.blob).then((image) => {
      if (canvas.width !== image.width || canvas.height !== image.height) {
        canvas.width = image.width;
        canvas.height = image.height;
      }
      context.drawImage(image, 0, 0);
      image.close();
    }).catch(() => {});
  }
  releaseLines(at);
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
    const player = new Audio(next.line.audio);
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

function onStats(next) {
  stats = next;
  if (next.session !== sessionId) newSession(next.session);
  const state = next.state;
  $("status").className = `status ${state}`;
  $("statusText").textContent = next.message;
  const running = state === "running" || state === "starting";
  $("liveBadge").hidden = state !== "running";
  $("liveText").textContent = next.live_source ? "LIVE" : "PLAYING AS LIVE";
  $("lagText").textContent = delay ? `${delay} s behind` : "";
  if (!running && !frames.length && state !== "ended") $("idle").hidden = false;
  $("start").disabled = running;
  $("stop").disabled = !running;

  const teams = next.teams || {};
  const name = (team, fallback) => (team && team.colour_bn ? `${team.colour_bn} দল` : fallback);
  $("nameA").textContent = name(teams.team_a, "দল ক");
  $("nameB").textContent = name(teams.team_b, "দল খ");
  $("labelA").textContent = name(teams.team_a, "Team A");
  $("labelB").textContent = name(teams.team_b, "Team B");
  for (const [team, ids] of [["team_a", ["kitA", "kitA2"]], ["team_b", ["kitB", "kitB2"]]]) {
    const colour = teams[team]?.colour;
    ids.forEach((id) => kit($(id), colour));
    const select = document.querySelector(`select.colour[data-team=${team}]`);
    if (select && colour && document.activeElement !== select) select.value = colour;
  }
  $("scoreA").textContent = bn(next.score.team_a);
  $("scoreB").textContent = bn(next.score.team_b);
  $("minute").textContent = state === "starting" ? "–" : bn(next.minute) + "'";
  $("phase").textContent = {
    play: "in play", before: "before kick-off", break: "break",
    shootout: "shootout", full_time: "full time",
  }[next.phase] || next.phase;

  $("tRate").textContent = next.rate ? next.rate.toFixed(1) : "–";
  $("tRate").parentElement.classList.toggle("warn", next.rate > 0 && next.rate < 0.8 * next.target_rate);
  $("tPlayers").textContent = next.players.toFixed(0);
  $("tBall").textContent = Math.round(next.ball_share * 100) + "%";
  $("tWhistles").textContent = next.sound ? next.whistles : "no sound";
  $("tRoars").textContent = next.sound ? next.roars : "–";
  $("tDropped").textContent = next.dropped;
}

function newSession(id) {
  sessionId = id;
  frames = [];
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
  if (message.state === "starting") $("idleText").textContent = message.message + "…";
}

// ---------------------------------------------------------------- socket

function connect() {
  const socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.binaryType = "arraybuffer";
  socket.onmessage = (event) => {
    if (typeof event.data !== "string") {
      const view = new DataView(event.data);
      const kind = view.getUint8(0);
      const t = view.getFloat64(1, true);
      if (kind === 1) onPicture(t, new Blob([new Uint8Array(event.data, 9)], { type: "image/jpeg" }));
      else if (kind === 2) onMatchSound(t, event.data.slice(9));
      return;
    }
    const message = JSON.parse(event.data);
    if (message.type === "hello") {
      onStats(message.stats);
      message.lines.forEach((line) => onLine(line, true));
      message.notes.forEach(onNote);
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
    video.src = data.video;
    video.hidden = false;
    canvas.hidden = true;
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
  canvas.hidden = false;
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

async function start() {
  showError();
  const source = { kind: sourceKind };
  if (sourceKind === "camera") source.index = Number($("cameraIndex").value);
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
  };
  soundOn();
  try {
    $("start").disabled = true;
    await api("/api/start", { source, settings });
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
  request.open("POST", "/api/upload");
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

  document.querySelectorAll("#control [data-action]").forEach((button) => {
    button.onclick = () => {
      showError();
      const team = button.closest("[data-team]")?.dataset.team;
      api("/api/operator", { action: button.dataset.action, team }).catch(showError);
    };
  });
  document.querySelectorAll("select.colour").forEach((select) => {
    select.onchange = () => api("/api/operator", {
      action: "colour", team: select.dataset.team, colour: select.value,
    }).catch(showError);
  });

  $("delay").value = delay;
  $("delayOut").textContent = `${delay} s`;
  $("delay").oninput = (event) => {
    delay = Number(event.target.value);
    localStorage.setItem("delay", delay);
    $("delayOut").textContent = `${delay} s`;
    if (frames.length || newest) retime(newest);
    if (stats) $("lagText").textContent = delay ? `${delay} s behind` : "";
  };
  $("matchVolume").oninput = (event) => { if (matchGain) matchGain.gain.value = event.target.value / 100; };
  $("voiceVolume").oninput = (event) => { if (speaking) speaking.volume = event.target.value / 100; };
}

async function boot() {
  wire();
  try {
    const state = await api("/api/state");
    operator = state.operator;
    colours = state.colours;
    $("control").hidden = !operator;
    document.querySelectorAll("select.colour").forEach((select) => {
      select.replaceChildren(...Object.entries(colours).map(([name, bengali]) => {
        const option = document.createElement("option");
        option.value = name;
        option.textContent = `${bengali} (${name})`;
        return option;
      }));
    });
    fillVideos(state.videos);
    if (state.mics) {
      $("mics").replaceChildren(...state.mics.map((name) => {
        const option = document.createElement("option");
        option.value = name;
        return option;
      }));
    }
    if (!operator) $("idleText").textContent = "The commentary starts when the operator starts the match.";
    $("soundOn").hidden = false;
  } catch (error) { showError(error); }
  connect();
}

boot();

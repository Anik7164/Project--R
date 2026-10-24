# Project--R: Bridging the Semantic Gap in Bengali Sports Commentary

![Status](https://img.shields.io/badge/Status-Active-success)
![Course](https://img.shields.io/badge/Course-CSE498R-blue)
![Language](https://img.shields.io/badge/Output-Bengali-green)

**Project--R** is an automated multimodal pipeline for real-time sports commentary and post-match journalism generation using Large Language Models (LLMs). Developed as a CSE498R Directed Research initiative, this system bridges the "semantic gap" between raw mathematical telemetry—such as millisecond timestamps and $X/Y$ coordinates—and engaging human broadcast consumption.

By decoupling visual perception from narrative generation, the system executes locally on edge devices to synthesize high-energy, colloquially accurate Bengali sports commentary for grassroots and amateur matches without the need for expensive human media teams.

---

## 🛠️ System Architecture

The pipeline is engineered for low-latency edge inference and is structured into four sequential phases:

1. **Phase 1: Data Ingestion and Event Structuring**
   Parses high-velocity JSON telemetry streams and utilizes rule-based event triggers to instantly initiate translation protocols for critical events like shots, fouls, or goals.
2. **Phase 2: Semantic Translation Middleware**
   Geometrically enriches triggered event data by calculating exact shooting distances and angles via Euclidean coordinates, which are then injected into strict, constraint-enforced prompt templates.
3. **Phase 3: Bilingual Natural Language Generation**
   A quantized open-weights LLM (e.g., Llama-3 or Mistral) generates short live commentary batches in English, while a secondary streaming prompt concurrently translates this text into broadcast-ready Bengali.
4. **Phase 4: Streaming Audio Synthesis and Synchronization**
   The Bengali stream routes into a multilingual Text-to-Speech (TTS) API (e.g., ElevenLabs). A 1.5-second video buffer delay synchronizes the synthesized voice track with the live visual action on-screen.

### Pipeline Visualization

```mermaid
flowchart TD
    A["Phase 1: Event Structuring<br/>(Action filters & trigger rules)"] --> B["Phase 2: Semantic Translation<br/>(Euclidean distance, xG, prompt injection)"]
    B --> C["Phase 3: LLM Generation Engine<br/>(Llama-3 / Mistral constraint generation)"]
    
    C --> D["English Live Micro-Batch"]
    
    D --> E["Bengali Translation Stream"]
    D --> F["Text Output"]
    
    E --> G["Phase 4: Streaming TTS Engine<br/>(ElevenLabs Multilingual v2)"]
    
    G --> H["Synced Audio Broadcast"]
    F --> H

---

## 🎥 Player Tracking

```bash
python src/track_players.py --video data/videos/test1.mp4   # --model defaults to ./yolo26n.pt
python src/evaluate_tracking.py outputs/tracking/test1_tracks.csv
```

BoT-SORT alone gives a player a new ID whenever it loses them: occlusion, leaving the view while the camera pans, or a broadcast cut. `src/identity.py` adds a stable player ID on top:

* **Camera motion** (`src/camera_motion.py`): the pan/zoom between frames is estimated from the background, so a player who left the view is still expected at the right place in current-frame pixels when they come back.
* **Scene cuts** (replays, close-ups, cuts between two wide cameras, dissolves) are detected; positions from before a cut are not trusted after it.
* **Re-linking**: a new BoT-SORT ID is pending for 3 frames, then matched to a currently missing player by position (gate grows with time missing) and shirt/shorts colour, or given a new ID. When only colour is available (after a cut, or missing for over ~2 s), ambiguous links between teammates are refused: a new ID is safer than a wrong one.
* **Players running into each other**: BoT-SORT sometimes swaps two IDs, or carries one onto the other player, when players cross. A player's colour model is frozen while the view disagrees with it; once two players of different kits separate, an ID that now shows the other player is handed back (or the two are exchanged); and a team colour model, fitted while the video runs, releases an ID whose player suddenly wears the other team's shirt and never re-links a player to someone of the other team. Two teammates who swap cannot be told apart by colour; that needs the shirt number.
* **Duplicates**: a partial-body box inside a bigger box of the same player is dropped.
* **Not players**: people whose feet are not on grass (crowd, staff, faces in close-ups) are left out.

`outputs/tracking/test1_tracks.csv` keeps both IDs: `track_id` (stable) and `tracker_id` (raw BoT-SORT), so `evaluate_tracking.py` can compare them on any clip. With MOTChallenge ground truth (for example SoccerNet tracking) add `--gt gt.txt` for IDF1/MOTA (`pip install motmetrics`).

Tests run on a synthetic broadcast clip (panning camera, cuts, occlusions): `pip install pytest && python -m pytest tests`.

### Teams

```bash
python src/identify_teams.py --video data/videos/test1.mp4
```

Writes `outputs/tracking/test1_teams.csv`: for every stable player ID its `team` (`team_a`/`team_b`), `role` (`player`, `goalkeeper`, `referee`) and shirt colour in English and Bengali. The teams are the two big kit-colour clusters; goalkeepers and referees fit neither. A goalkeeper is almost always the last person at one end of play and joins the nearer team (the two goalkeeper kits always go to different teams); a referee stays in the middle of play.

### Ball and match events

```bash
python src/ball_tracking.py --video data/videos/test1.mp4
python src/match_events.py  --video data/videos/test1.mp4 [--spotting results_spotting.json] [--start-minute 0]
```

`ball_tracking.py` keeps the YOLO "sports ball" detection that continues the ball's path (camera motion removed) and fills short gaps. `match_events.py` writes `outputs/tracking/test1_events.json` with each event's minute, type (names from `dataset.json`) and the players involved (scorer/assist, fouler/fouled, booked player, set-piece taker), using ball possession from the tracks:

* With `--spotting` the event types and times come from a SoccerNet action-spotting model's `results_spotting.json` (goal, foul, yellow/red card, corner, free kick, goal kick, penalty).
* Without it, simple rules are used: a yellow/red rectangle above the referee's head is a card, a ball that stood still and is then played is a set piece (a goal kick when the goalkeeper takes it), and a free kick implies a foul by the opponent nearest the fouled player. Goals need the spotting model.

### Who is who: name → number → colour

```bash
pip install easyocr
python src/extract_jersey_crops.py
python src/jersey_reading.py   --video data/videos/test1.mp4 [--roster roster.json]
python src/player_identity.py  --video data/videos/test1.mp4
```

`jersey_reading.py` reads the name and number on each player's shirt from their best torso crops; a value is only accepted when at least two crops agree. With a roster (`{"team_a": {"7": "Rashford"}, ...}`) a read number becomes the player's name and a misread name snaps to the closest roster name. `player_identity.py` then fills the Bengali templates in `dataset.json` for every event, naming each player by the best thing we know: their name, else "৭ নম্বর জার্সির খেলোয়াড়", else "লাল জার্সির খেলোয়াড়". Templates with a `{jersey}` slot are only used when both name and number are known.

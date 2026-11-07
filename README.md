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

## 💾 Disk use

Every script imports `src/project_cache.py` first, which points the PyTorch, Ultralytics, EasyOCR, Hugging Face and temp folders to `<project>/.cache`, so models and caches stay on the project's drive instead of the home folder (C: on Windows). For `pip` itself, run this once per terminal before installing:

```powershell
$env:PIP_CACHE_DIR = "$PWD\.cache\pip"; $env:TEMP = "$PWD\.cache\tmp"; $env:TMP = $env:TEMP
```

## 📦 Models

```powershell
python src/setup_models.py --install    # extra packages + every model, all inside the project folder
```

This installs `requirements-models.txt` (EasyOCR and BoxMOT without their dependencies, so they cannot replace the project's OpenCV 5) and downloads YOLO26, PARSeq, the EasyOCR text detector, LMBN ReID, ResNet-152, NetVLAD++ and the Bengali MMS-TTS voice into `.cache/`.

### Football detector (train once, on your GPU)

The COCO model only knows "person" and "sports ball": it cannot tell a referee from a player, and it often misses the small ball. `src/train_detector.py` trains YOLO26 on SoccerNet-Tracking, where every player, goalkeeper, referee and the ball is boxed in every frame of 57 broadcast clips:

```powershell
python src/train_detector.py download     # ~12 GB into data/SoccerNet
python src/train_detector.py prepare      # YOLO dataset (hard links: no extra space)
python src/train_detector.py train        # yolo26s at 1280 px; --model yolo26m for more accuracy
```

The result goes to `models/football_yolo26s.pt`, and every script then uses it automatically (newest `models/football_*.pt`, unless `--model` or `YOLO_MODEL` says otherwise). With it, tracks get the classes player / goalkeeper / referee, which then decide each person's role in `identify_teams.py`, and `ball_tracking.py` uses its ball class. The 49 SoccerNet test clips are never used for training, so they stay free for a fair evaluation later.

**Fine-tune on your own footage.** SoccerNet is TV broadcast; a match filmed from the stands looks different, and the ball is found far less often. Fine-tune on your own videos, either fully automatically or with hand-checked labels. Only the ball needs checking: player boxes are taken from the current model (already mAP50 0.97).

*Automatic (no hand work):* one command labels the ball in your videos and fine-tunes.

```powershell
python src/train_detector.py auto-finetune "data/videos/test1 5.mp4" data/videos/match2.mp4
#   --rounds 2 labels again with the fine-tuned model and trains once more (twice the time, usually more balls)
```

It runs the current model on 3-second windows (every 6 s) at a low confidence, links the ball candidates across frames with camera motion removed, and keeps only tracks that last, move like a ball, are confidently seen at least once and are on the grass. Faint detections on such a track and 1-2 frame misses where the ball goes straight on become labels; candidates off the track (caps in the crowd, socks, line marks) get none. See `src/auto_label.py`. On a 10-window test of a local match, 75 frames were labelled and about 68 of the ball boxes were right by eye. This is self-training: it helps less than hand labels, but it costs nothing. More different videos help more than more frames of one video. Frames you later correct by hand are never overwritten by a new auto-label run.

Tried on 3 local matches (592 frames), the automatic fine-tune found the ball in a held-out match only 0.6% more often (9,152 vs 9,098 frames): its labels are mostly balls the model already saw. The balls it misses completely need a person.

*Click the ball (about 10 minutes, recommended):* only the frames where the model is unsure are shown; click the ball (a magnifier helps), or press N if no ball is visible. The box is drawn for you.

```powershell
python src/click_balls.py data/videos/match1.mp4 data/videos/match2.mp4 data/videos/match3.mp4 --frames 300
python src/train_detector.py finetune     # validation is on your clicked frames only
```

Run `click_balls.py` again to label more; frames already done are not shown again.

*Hand-checked in a labelling tool (slowest):*

```powershell
python src/train_detector.py label-frames data/videos/match2.mp4 data/videos/match3.mp4 --limit 300
#   -> data/local_det/images + labels (only the model's ball boxes) and people/ (its player boxes, kept aside).
#      Upload images + labels + classes.txt to a labelling tool (e.g. Roboflow), fix the ball in each
#      frame (add a missed ball, delete a wrong one, nothing if no ball is visible), export as YOLO.
python src/train_detector.py import-labels path/to/unzipped/export   # handles renamed files and class order
python src/train_detector.py prepare-local          # your frames (+ player boxes) + as many SoccerNet frames
python src/train_detector.py train --data local --model models/football_yolo26s.pt --epochs 40
```

`--all-classes` on label-frames puts the player boxes in the labels to correct as well. The result is `models/football_yolo26s_local.pt`, which the scripts then use. Running `label-frames` again adds new frames and never overwrites a label file that is already there. Validation uses the last part of each video, so check that its ball mAP50 goes up.

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
* **Deep appearance (ReID)**: when BoxMOT is installed, an LMBN person re-identification network (`src/reid.py`) adds an appearance embedding to the colour model when re-linking. On test1.mp4 it separates teammates better than colour (AUC 0.85 vs 0.79); on a 30 s clip it cut IDs holding both teams from 5 to 4, with 88 instead of 85 IDs, so the effect is small so far. A model fine-tuned on SoccerNet-ReID should do better. `--no-reid` turns it off.
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
pip install tensorflow                                        # once, for action_spotting.py
python src/ball_tracking.py   --video data/videos/test1.mp4
python src/action_spotting.py --video data/videos/test1.mp4  # trained SoccerNet event model
python src/match_events.py    --video data/videos/test1.mp4 [--start-minute 0] [--start-score 0-0]
python src/scoreboard.py      --video data/videos/test1.mp4  # optional: just list the goals
```

`action_spotting.py` runs the trained SoccerNet-v2 **NetVLAD++** model (Giancola & Ghanem 2021, official weights from [sn-spotting](https://github.com/SoccerNet/sn-spotting), Apache-2.0). Two frames per second go through ResNet-152 exactly as SoccerNet's features were made; the model scores 17 event types for every half second from the 15 s around it, and the peaks are written to `outputs/tracking/test1_spotting.json`. Both models download once into `.cache/` (ResNet-152 ~240 MB, NetVLAD++ ~20 MB). On a CPU it takes about 1–2 minutes for the 2-minute test clip. `match_events.py` uses that file automatically when it exists (`--rules` forces the old rules).

On test1.mp4 the model finds the corner after the 0:54 save, the foul at 1:28 and the card at 1:35; it misses both goals (highlights cut the build-up and celebration it was trained on), which the scoreboard covers. Cards it spots take their colour and time from the card seen in the picture (it called the 1:35 red card yellow), and set pieces in the 20 s after a goal (really the kick-off) are dropped.

**The sound (local matches).** Filmed from the stands, the ball is often a few pixels wide, but the whistle and the crowd are clear. `match_events.py` reads them from the video's sound (`audio_cues.py`, about 3 s for a 10-minute video, written to `<name>_audio.json`):

* **Referee's whistle**: a flat tone in 2.8-4.5 kHz held for 0.25 s; the referee blows the same whistle all match, so only tones near the most-heard pitch count (voices and horns slide in pitch and are spread over the band). On match1 it found the whistles at 2:40, 3:30, 4:05, 6:47 and 7:19 and rejected the sliding voices and horns checked against the spectrogram.
* **Crowd roar**: 3 dB louder than the median of the last 30 s for 1.5 s. On match1 the roars were at 4:04 and 4:37 (attacks near goal) and 7:38-7:48 (the goal at 7:49).
* Model events are checked with them: a spotted foul or free kick needs a whistle nearby, a shot or corner a roar, and anything else (throw-in, goal kick) a confidence of 0.6, otherwise it is dropped. Two spots of the same type within 10 s are one event.
* Where nothing else is said, a roar becomes `big_chance` ("the crowd is up, orange are attacking"), a whistle `whistle_stop` ("the referee stops play"), and a spell where one team keeps the ball `possession_spell`. These name the team, not a player, so they stay right on weak footage. `--no-audio` turns all this off.

**How right is the commentary?** Write down what really happens, then score the events against it:

```powershell
python src/mark_events.py --video data/videos/match5.mp4 --teams orange white   # one key per event while watching
python src/score_commentary.py --video data/videos/match5.mp4
```

`mark_events.py` plays the video; press g goal, s shot, a attack, f foul, k free kick, c corner, t throw-in, o goal kick, x tackle, y card, w whistle (then 1 or 2 for the team), space to pause, j/l to jump 5 s, u to undo. Or write `outputs/tracking/<name>_truth.txt` by hand, one line per event (`7:49 goal orange`). `score_commentary.py` prints each line as right, wrong or wrong team (an event of a fitting type within a few seconds, e.g. "the referee stops play" fits a whistle, foul or free kick) and the share of right lines; passes, runs and possession lines are not checked.

`ball_tracking.py` keeps the YOLO "sports ball" detection that continues the ball's path (camera motion removed) and fills short gaps. `match_events.py` writes `outputs/tracking/test1_events.json` with each event's minute, type (names from `dataset.json`) and the players involved (scorer/assist, fouler/fouled, booked player, set-piece taker), using ball possession from the tracks:

* **Goals** come from the broadcast score graphic (`src/scoreboard.py`): it finds the part of the top of the frame that stays the same, the two score digits in it, and reports a goal whenever one digit changes and stays changed for 2 seconds. On test1.mp4 it finds both goals (1:02.8 United 1-0, 1:49.6 Tottenham 1-1). The colours of the board's two halves are matched to the team colours to know which team scored; the scorer is the last player of that team on the ball, which is only as good as the ball tracking (weak on highlights). Use `--no-scoreboard` to turn it off.
* **Cards in close-ups**: besides looking above a tracked referee, `card_detection.py` looks for a very bright, solid red/yellow card anywhere in the frame that stands alone (not part of an advert) and has a person right below it (YOLO person check on those frames only), seen for about half a second. On test1.mp4 this finds the red card at 1:35.1 and nothing else.
* With `--spotting` the event types and times come from a SoccerNet action-spotting model's `results_spotting.json` (goal, foul, yellow/red card, corner, free kick, goal kick, penalty).
* Without it, simple rules are used: a yellow/red rectangle above the referee's head is a card, a ball that stood still and is then played is a set piece (a goal kick when the goalkeeper takes it), and a free kick implies a foul by the opponent nearest the fouled player. These rules are unreliable on highlight reels, which cut away before set pieces are taken: on test1.mp4 none of the fouls/free kicks they found were real.

### Research benchmarks

`benchmarks/` compares the pipeline with published models on SoccerNet, using the official evaluation code. Event spotting compares MaxPool, NetVLAD, CALF and NetVLAD++ on the 100 test matches. Tracking compares 8 trackers and 9 detectors, each with and without our ID layer, scored with HOTA, IDF1 and MOTA. Setup, downloads (all inside the project folder) and run order are in [benchmarks/README.md](benchmarks/README.md).

### Who is who: name → number → colour

```bash
python src/extract_jersey_crops.py
python src/jersey_reading.py   --video data/videos/test1.mp4 [--roster roster.json] [--ocr easyocr]
python src/player_identity.py  --video data/videos/test1.mp4
python src/speak_commentary.py --video data/videos/test1.mp4 [--voice edge]
```

`jersey_reading.py` reads the name and number on each player's shirt from their best torso crops; a value is only accepted when at least two crops agree. With a roster (`{"team_a": {"7": "Rashford"}, ...}`) a read number becomes the player's name and a misread name snaps to the closest roster name. `player_identity.py` then fills the Bengali templates in `dataset.json` for every event, naming each player by the best thing we know: their name, else "৭ নম্বর জার্সির খেলোয়াড়", else "লাল জার্সির খেলোয়াড়". Templates with a `{jersey}` slot are only used when both name and number are known.

Shirt text is read with **PARSeq** (Bautista & Atienza, ECCV 2022, `src/parseq_reader.py`), the scene-text recogniser the best published shirt-number readers build on; EasyOCR's CRAFT detector finds the text regions first, and only digits (or only letters) can be predicted. `--ocr easyocr` uses EasyOCR alone.

`speak_commentary.py` speaks each line in Bengali at its event's video time (a line waits until the previous one is finished) and writes `outputs/tracking/test1_commentary.wav` plus `test1_commentary.mp4`, the video with the commentary over the quieter match sound. The default voice is Meta's MMS-TTS Bengali, offline (CC-BY-NC: research use); `--voice edge` uses Microsoft's neural Bengali voices (`bn-BD-PradeepNeural`, `--speaker female` for Nabanita), which sound more natural but need the internet.

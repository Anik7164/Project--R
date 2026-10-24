"""
Synthetic broadcast clip for testing the identity layer without real video.

A textured pitch is filmed by a panning, slowly zooming camera. Two teams
and a referee move around. A fake tracker reproduces BoT-SORT's failure
modes: a new track ID whenever a player is lost (left the view, occluded)
or randomly, all new IDs after a scene cut, and occasional duplicate
partial-body boxes.
"""

import numpy as np
import cv2

PITCH_W, PITCH_H = 3200, 1600
VIEW_W, VIEW_H = 960, 540
PLAYER_W, PLAYER_H = 22, 56

KITS = {
    "home": ((40, 40, 200), (240, 240, 240)),    # red shirt, white shorts
    "away": ((200, 120, 30), (60, 40, 20)),      # blue shirt, dark shorts
    "home_gk": ((30, 220, 230), (30, 30, 30)),   # yellow
    "away_gk": ((150, 60, 150), (150, 60, 150)), # purple
    "referee": ((20, 20, 20), (20, 20, 20)),     # black
}


def make_pitch(rng):
    pitch = np.zeros((PITCH_H, PITCH_W, 3), np.uint8)
    for x in range(0, PITCH_W, 160):
        shade = 110 if (x // 160) % 2 else 135
        pitch[:, x:x + 160] = (40, shade, 40)
    noise = rng.integers(-18, 18, pitch.shape, dtype=np.int16)
    pitch = np.clip(pitch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Pitch markings and advertising boards give the flow corners.
    cv2.rectangle(pitch, (60, 60), (PITCH_W - 60, PITCH_H - 60), (230, 230, 230), 4)
    cv2.line(pitch, (PITCH_W // 2, 60), (PITCH_W // 2, PITCH_H - 60), (230, 230, 230), 4)
    cv2.circle(pitch, (PITCH_W // 2, PITCH_H // 2), 180, (230, 230, 230), 4)
    for x in range(0, PITCH_W, 90):
        colour = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(pitch, (x, 0), (x + 80, 50), colour, -1)
        cv2.putText(pitch, "AD", (x + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for _ in range(400):
        x, y = rng.integers(70, PITCH_W - 70), rng.integers(70, PITCH_H - 70)
        cv2.circle(pitch, (int(x), int(y)), 3, (200, 220, 200), -1)
    return pitch


def make_players(rng):
    players = []
    roles = (["home"] * 10 + ["home_gk"] + ["away"] * 10 + ["away_gk"] + ["referee"])
    for pid, role in enumerate(roles, start=1):
        players.append({
            "id": pid,
            "role": role,
            # Spot in the team shape, relative to the ball.
            "offset": np.array([rng.uniform(-700, 700), rng.uniform(-450, 450)]),
            "pos": np.array([PITCH_W / 2, PITCH_H / 2]) + rng.uniform(-600, 600, 2),
            "vel": np.zeros(2),
        })
    return players


def simulate(frames=600, seed=0, cut_ranges=((300, 330),), break_prob=0.004,
             duplicate_prob=0.01, occlusion_prob=0.004):
    """Yield (frame_index, image, detections) tuples.

    detections: list of dicts with tracker_id, box, conf, truth (true
    player ID, or None for a duplicate box).
    """
    rng = np.random.default_rng(seed)
    pitch = make_pitch(rng)
    crowd = rng.integers(0, 255, (VIEW_H, VIEW_W, 3), dtype=np.uint8)
    players = make_players(rng)

    next_tid = [1]
    current_tid = {}
    occluded_until = {}

    def new_tid():
        next_tid[0] += 1
        return next_tid[0] - 1

    for frame in range(frames):
        # Play moves up and down the pitch; the camera follows the ball
        # and zooms a little.
        ball = np.array([
            PITCH_W / 2 + 900 * np.sin(frame / 160),
            PITCH_H / 2 + 250 * np.sin(frame / 110),
        ])
        zoom = 1.0 + 0.15 * np.sin(frame / 200)
        view_w, view_h = VIEW_W / zoom, VIEW_H / zoom
        cam = np.clip(ball - [view_w / 2, view_h / 2], 0, [PITCH_W - view_w - 1, PITCH_H - view_h - 1])

        for p in players:
            target = ball + 0.6 * p["offset"]
            pull = np.clip((target - p["pos"]) * 0.02, -2.5, 2.5)
            p["vel"] = 0.9 * p["vel"] + 0.1 * pull + rng.normal(0, 0.3, 2)
            p["pos"] = np.clip(p["pos"] + p["vel"], [100, 100], [PITCH_W - 100, PITCH_H - 100])

        in_cut = any(a <= frame < b for a, b in cut_ranges)
        if in_cut:
            current_tid.clear()
            yield frame, crowd.copy(), []
            continue

        x0, y0 = cam
        src = pitch[int(y0):int(y0 + view_h), int(x0):int(x0 + view_w)]
        image = cv2.resize(src, (VIEW_W, VIEW_H))

        detections = []
        for p in players:
            px = (p["pos"][0] - x0) * zoom
            py = (p["pos"][1] - y0) * zoom
            w, h = PLAYER_W * zoom, PLAYER_H * zoom
            box = np.array([px - w / 2, py - h, px + w / 2, py])
            shirt, shorts = KITS[p["role"]]
            ix1, iy1, ix2, iy2 = box.astype(int)
            cv2.rectangle(image, (ix1, iy1), (ix2, int(iy1 + 0.12 * h)), (140, 170, 210), -1)
            cv2.rectangle(image, (ix1, int(iy1 + 0.12 * h)), (ix2, int(iy1 + 0.52 * h)), shirt, -1)
            cv2.rectangle(image, (ix1, int(iy1 + 0.52 * h)), (ix2, int(iy1 + 0.72 * h)), shorts, -1)
            cv2.rectangle(image, (ix1 + 2, int(iy1 + 0.72 * h)), (ix2 - 2, iy2), (140, 170, 210), -1)

            visible = box[0] >= 0 and box[1] >= 0 and box[2] <= VIEW_W and box[3] <= VIEW_H
            if visible and rng.random() < occlusion_prob:
                occluded_until[p["id"]] = frame + int(rng.integers(5, 40))
            if not visible or occluded_until.get(p["id"], -1) > frame:
                current_tid.pop(p["id"], None)
                continue
            if p["id"] not in current_tid or rng.random() < break_prob:
                current_tid[p["id"]] = new_tid()
            detections.append({
                "tracker_id": current_tid[p["id"]],
                "box": box,
                "conf": 0.85,
                "truth": p["id"],
            })
            if rng.random() < duplicate_prob:
                half = box.copy()
                half[3] = box[1] + 0.45 * (box[3] - box[1])
                detections.append({
                    "tracker_id": new_tid(),
                    "box": half,
                    "conf": 0.5,
                    "truth": None,
                })
        yield frame, image, detections

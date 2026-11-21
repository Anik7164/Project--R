"""
Events learnt from local matches. The SoccerNet model was trained on TV
broadcasts and is mostly wrong on a local match filmed from the stands,
so these two learn from local footage instead.

1. attacks_from_crowd(): a model that teaches itself on every match, with
   no hand-made labels. The crowd roars when the ball is near a goal, so
   the pictures around the roars show what "near a goal" looks like from
   this camera in this ground (the stand behind the goal, the box lines).
   A small model learns that look and then finds the other moments that
   look the same but had no roar. match_events.py uses it by itself.
   On the two matches with a hand-made truth list (match4, match5, open
   play): 8 new attack lines, 7 of them right, and the attacks mentioned
   in match4 went from 5 to 11 of 24. Small numbers, and the settings
   below were chosen on these same two matches; see README.

2. A model trained on hand-made truth lists (train_events.py). For every
   half second it says how likely an "attack" (attack, shot, goal,
   corner) or a "stoppage" (whistle, foul, free kick, card) is. With only
   two labelled matches it does not carry over to an unseen match yet
   (no better than guessing), so train_events.py does not save it until
   it does. It needs more labelled matches.

What it looks at, per half second:
- the picture: the ResNet-152 features action_spotting.py already saves
  (<name>_resnet152.npy), with the match's own average taken off (so the
  stadium and the camera place drop out), squeezed to a few numbers, for
  now, the last few seconds and the next few seconds, plus how fast the
  picture is changing;
- the sound: how loud it is against the last 30 seconds, how loud each
  pitch band is, and how clear a whistle tone is. A video with no sound
  still works (the sound numbers are left empty).

    from local_events import load_model, match_scores, spot
    scores = match_scores(load_model(), video)      # {group: one number per half second}
    for seconds, group, confidence in spot(model, scores): ...
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
from pathlib import Path

import numpy as np

from audio_cues import (FRAME, HOP, SAMPLE_RATE, WHISTLE_BAND, WHISTLE_PROMINENCE_DB,
                        load_audio)

ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = ROOT / "outputs" / "tracking"
MODEL_PATH = ROOT / "models" / "local_events.joblib"

STEP = 0.5                      # seconds; one ResNet feature per step
# Which real events (score_commentary.py types) each group stands for,
# and the commentary line a spotted group becomes.
GROUPS = {
    "attack": {"attack", "shot", "goal", "corner", "penalty"},
    "stoppage": {"whistle", "foul", "freekick", "card"},
    "throw_in": {"throw_in"},
    "goal_kick": {"goal_kick"},
}
LINE = {"attack": "big_chance", "stoppage": "whistle_stop",
        "throw_in": "throw_in", "goal_kick": "goal_kick"}
LABEL_SECONDS = 3.0             # steps this close to a real event are "yes"
WINDOWS = ((-6.0, -3.0), (-3.0, 0.0), (0.0, 3.0))   # seconds around a step
BASELINE_SECONDS = 30.0
BANDS = ((0, 500), (500, 1500), (1500, 3000), (3000, 6000))   # Hz
PEAK_GAP_SECONDS = 10.0         # peaks of one group closer than this are one event


def _window_mean(values, first, last):
    """Mean of values[i+first : i+last] for every i (steps), edges clipped."""
    n = len(values)
    total = np.vstack([np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)])
    index = np.arange(n)
    low = np.clip(index + first, 0, n)
    high = np.clip(index + last, 0, n)
    count = np.maximum(high - low, 1)[:, None]
    return (total[high] - total[low]) / count


def _steps(seconds):
    return int(round(seconds / STEP))


def picture_numbers(features, pca):
    """features: (N, 2048) ResNet-152, one per STEP. pca: fitted on the
    training matches."""
    centred = features - features.mean(axis=0)
    small = pca.transform(centred)
    change = np.linalg.norm(np.diff(features, axis=0, prepend=features[:1]), axis=1)
    change = (change / (np.median(change) + 1e-9))[:, None]
    parts = [small, change]
    for first, last in WINDOWS:
        parts.append(_window_mean(small, _steps(first), _steps(last)))
        parts.append(_window_mean(change, _steps(first), _steps(last)))
    return np.hstack(parts)


def sound_numbers(samples, count):
    """(count, k) sound numbers per STEP; all NaN if the video has no sound."""
    width = (2 + len(BANDS)) * (1 + len(WINDOWS))
    if samples is None or len(samples) < SAMPLE_RATE:
        return np.full((count, width), np.nan)
    size = int(STEP * SAMPLE_RATE)
    steps = min(count, len(samples) // size)
    freqs = np.fft.rfftfreq(FRAME, 1 / SAMPLE_RATE)
    whistle = (freqs >= WHISTLE_BAND[0]) & (freqs <= WHISTLE_BAND[1])
    window = np.hanning(FRAME)
    rows = np.zeros((steps, 2 + len(BANDS)))
    for i in range(steps):
        block = samples[i * size:(i + 1) * size]
        frames = np.lib.stride_tricks.sliding_window_view(block, FRAME)[::HOP]
        power = np.abs(np.fft.rfft(frames * window, axis=1)) ** 2
        rows[i, 0] = 10 * np.log10(np.mean(block ** 2) + 1e-12)
        spectrum = 10 * np.log10(power[:, whistle] + 1e-12)
        rows[i, 1] = np.max(spectrum.max(axis=1) - np.median(spectrum, axis=1))
        for b, (low, high) in enumerate(BANDS):
            band = (freqs >= low) & (freqs < high)
            rows[i, 2 + b] = 10 * np.log10(power[:, band].mean() + 1e-12)
    # Loudness against the last 30 seconds (past only, as audio_cues does).
    history = _steps(BASELINE_SECONDS)
    for column in [0] + list(range(2, 2 + len(BANDS))):
        level = rows[:, column].copy()
        for i in range(steps):
            past = level[max(0, i - history):i]
            rows[i, column] = level[i] - (np.median(past) if len(past) else level[i])
    rows[:, 1] -= WHISTLE_PROMINENCE_DB
    parts = [rows]
    for first, last in WINDOWS:
        parts.append(_window_mean(rows, _steps(first), _steps(last)))
    out = np.full((count, width), np.nan)
    out[:steps] = np.hstack(parts)
    return out


def match_numbers(features, samples, pca):
    return np.hstack([picture_numbers(features, pca),
                      sound_numbers(samples, len(features))])


def features_path(video):
    return TRACKING_DIR / f"{Path(video).stem}_resnet152.npy"


def load_features(video):
    path = features_path(video)
    if not path.exists():
        raise SystemExit(f"{path.name} is missing. Make it first:\n"
                         f"  python src/action_spotting.py --video {video}")
    return np.load(path)


# ----------------------------------------------------------------------
# 1. Taught by the crowd, match by match
# ----------------------------------------------------------------------
CROWD_ROAR_DB = 3.0            # roars this strong are the teacher
CROWD_AROUND = 5.0             # seconds before and after a roar that count as "attack"
CROWD_ROAR_SECONDS = 6.0       # of a long roar, only the start counts
CROWD_MIN_ROARS = 3            # fewer roars than this: nothing to learn from
CROWD_COMPONENTS = 16
CROWD_SMOOTH = (-2.0, 2.5)     # the picture is averaged over these seconds
CROWD_STRENGTH = 0.03          # LogisticRegression C: small = cautious
CROWD_SHARE = 0.15             # the clearest share of the match can become a line
CROWD_FLOOR = 0.5              # and only if it looks more like a roar moment than not
CROWD_GAP_SECONDS = 12.0


def crowd_lesson(count, cues, around=CROWD_AROUND):
    """yes/no per step: is a strong roar going on around this step."""
    times = np.arange(count) * STEP
    lesson = np.zeros(count, dtype=bool)
    roars = 0
    for cue in cues:
        if cue["type"] != "crowd_roar" or cue["strength"] < CROWD_ROAR_DB:
            continue
        roars += 1
        end = cue["time"] + min(cue["duration"], CROWD_ROAR_SECONDS)
        lesson |= (times >= cue["time"] - around) & (times <= end + around)
    return lesson, roars


def attacks_from_crowd(features, cues, quiet=None):
    """[(seconds, confidence)]: moments that look like the moments the
    crowd roared at. features: (N, 2048), one per STEP. cues: audio_cues.
    quiet(seconds) -> True for stretches that are not play (walkout,
    half-time, a shootout); they are neither learnt from nor reported."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    count = len(features)
    play = np.array([not (quiet and quiet(i * STEP)) for i in range(count)], dtype=bool)
    lesson, _ = crowd_lesson(count, cues)
    roars = sum(c["type"] == "crowd_roar" and c["strength"] >= CROWD_ROAR_DB
                and not (quiet and quiet(c["time"])) for c in cues)
    yes, no = int(lesson[play].sum()), int((~lesson[play]).sum())
    if roars < CROWD_MIN_ROARS or yes < 20 or no < 20:
        return []
    components = min(CROWD_COMPONENTS, int(play.sum()) - 1)
    small = PCA(n_components=components, random_state=0).fit_transform(
        features[play] - features[play].mean(axis=0))
    numbers = _window_mean(small, _steps(CROWD_SMOOTH[0]), _steps(CROWD_SMOOTH[1]))
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=CROWD_STRENGTH, max_iter=3000, class_weight="balanced"))
    scores = np.zeros(count)
    scores[play] = model.fit(numbers, lesson[play]).predict_proba(numbers)[:, 1]
    smooth = _window_mean(scores[:, None], -2, 3)[:, 0]
    limit = max(CROWD_FLOOR, np.quantile(smooth[play], 1 - CROWD_SHARE))
    taken = []
    for i in np.argsort(-smooth):
        if smooth[i] < limit:
            break
        if play[i] and all(abs(i - j) * STEP >= CROWD_GAP_SECONDS for j in taken):
            taken.append(int(i))
    return sorted((i * STEP, float(smooth[i])) for i in taken)


# ----------------------------------------------------------------------
# 2. Trained on hand-made truth lists (train_events.py)
# ----------------------------------------------------------------------


def load_model(path=MODEL_PATH):
    import joblib
    return joblib.load(path)


def model_scores(model, numbers):
    return {group: classifier.predict_proba(numbers)[:, 1]
            for group, classifier in model["classifiers"].items()}


def match_scores(model, video):
    numbers = match_numbers(load_features(video), load_audio(video), model["pca"])
    return model_scores(model, numbers)


def spot(model, scores, gap=PEAK_GAP_SECONDS):
    """[(seconds, group, confidence)]: the peaks of each group's score.
    A step counts if it is among the match's top share for that group
    (the share of "yes" steps seen in training), so a match the model is
    less sure about still gets its clearest moments."""
    found = []
    for group, values in scores.items():
        if len(values) == 0:
            continue
        smooth = _window_mean(values[:, None], -1, 2)[:, 0]
        limit = max(model["floor"][group],
                    np.quantile(smooth, 1 - model["share"][group]))
        order = np.argsort(-smooth)
        taken = []
        for i in order:
            if smooth[i] < limit:
                break
            if all(abs(i - j) * STEP >= gap for j in taken):
                taken.append(i)
                found.append((float(i * STEP), group, float(smooth[i])))
    return sorted(found)

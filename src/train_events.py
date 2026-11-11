"""
Train the local event model (local_events.py) on matches whose real
events are written down, and say honestly how good it is.

Each match needs, in outputs/tracking/:
    <name>_resnet152.npy   python src/action_spotting.py --video ...
    <name>_truth.txt       the real events, e.g. "7:47 goal orange"
                           (or <name>_truth.json from mark_events.py)

    python src/train_events.py --videos data/videos/match4.mp4 data/videos/match5.mp4

The test is "leave one match out": the model is trained on the other
matches and asked about a match it has never seen, for every match in
turn. That is the number to trust; a model tested on its own training
matches always looks good.

A group (attack, stoppage, ...) is kept only if, on the unseen matches,
its ranking is clearly better than guessing (MIN_LIFT). If no group
passes, nothing is saved: the model has only learnt what its training
matches look like, and more labelled matches are needed. With match4 and
match5 alone (2026-11) no group passes. match_events.py does not read
the saved model yet; that is the next step once a group passes.
"""

import project_cache  # noqa: F401  keeps downloads/caches in <project>/.cache
import argparse
from pathlib import Path

import numpy as np

from audio_cues import load_audio
from local_events import (GROUPS, LABEL_SECONDS, LINE, MODEL_PATH, STEP, load_features,
                          match_numbers, model_scores, spot)
from score_commentary import load_truth, score

COMPONENTS = 24          # numbers kept from the 2048 picture features
MIN_EVENTS = 8           # a group needs this many real events to be learnt
FLOOR = 0.0
MIN_LIFT = 1.5           # ranking on unseen matches / what guessing gives


def step_labels(truth, count):
    """{group: yes/no per step}: yes within LABEL_SECONDS of a real event."""
    times = np.arange(count) * STEP
    labels = {}
    for group, kinds in GROUPS.items():
        marks = np.zeros(count, dtype=bool)
        for event in truth:
            if event["type"] in kinds:
                marks |= np.abs(times - event["time"]) <= LABEL_SECONDS
        labels[group] = marks
    return labels


def load_match_data(video):
    name = Path(video).stem
    features = load_features(video)
    truth = load_truth(name)
    return {"name": name, "features": features, "samples": load_audio(video),
            "truth": truth, "labels": step_labels(truth, len(features))}


def fit(matches, components=COMPONENTS):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    centred = np.vstack([m["features"] - m["features"].mean(axis=0) for m in matches])
    pca = PCA(n_components=min(components, len(centred) - 1), random_state=0).fit(centred)
    numbers = np.vstack([match_numbers(m["features"], m["samples"], pca) for m in matches])
    model = {"pca": pca, "classifiers": {}, "share": {}, "floor": {},
             "matches": [m["name"] for m in matches]}
    for group, kinds in GROUPS.items():
        events = sum(e["type"] in kinds for m in matches for e in m["truth"])
        if events < MIN_EVENTS:
            continue
        labels = np.concatenate([m["labels"][group] for m in matches])
        # A video with no sound has empty sound numbers: filled with the
        # average, so they say nothing either way.
        classifier = make_pipeline(
            SimpleImputer(strategy="mean"), StandardScaler(),
            LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced"))
        model["classifiers"][group] = classifier.fit(numbers, labels)
        model["share"][group] = float(labels.mean())
        model["floor"][group] = FLOOR
    return model


def held_out_report(model, match):
    """How the model does on a match it was not trained on."""
    from sklearn.metrics import average_precision_score

    numbers = match_numbers(match["features"], match["samples"], model["pca"])
    scores = model_scores(model, numbers)
    report = {"steps": {}, "lines": None}
    for group, values in scores.items():
        labels = match["labels"][group]
        if labels.any():
            report["steps"][group] = (float(average_precision_score(labels, values)),
                                      float(labels.mean()))
    events = [{"time": seconds, "type": LINE[group]} for seconds, group, _ in spot(model, scores)]
    lines, used = score(events, match["truth"], {})
    right = sum(verdict == "right" for _, verdict, _ in lines)
    report["lines"] = (right, len(lines))
    report["found"] = {}
    for group in scores:
        indices = [i for i, e in enumerate(match["truth"]) if e["type"] in GROUPS[group]]
        report["found"][group] = (sum(i in used for i in indices), len(indices))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--save-anyway", action="store_true",
                        help="save the model even if it is no better than "
                             "guessing on an unseen match")
    args = parser.parse_args()

    matches = [load_match_data(video) for video in args.videos]
    for m in matches:
        sound = "with sound" if m["samples"] is not None else "no sound"
        print(f"{m['name']}: {len(m['features']) * STEP / 60:.1f} min, "
              f"{len(m['truth'])} real events, {sound}")

    lifts = {}
    if len(matches) >= 2:
        print("\nTest on a match the model has not seen (trained on the others):")
        total_right = total_lines = 0
        for held in matches:
            model = fit([m for m in matches if m is not held])
            report = held_out_report(model, held)
            right, lines = report["lines"]
            total_right, total_lines = total_right + right, total_lines + lines
            print(f"  {held['name']}: {right}/{lines} lines right"
                  + (f" = {100 * right / lines:.0f}%" if lines else ""))
            for group, (precision, chance) in report["steps"].items():
                found, real = report["found"][group]
                lifts.setdefault(group, []).append(precision / chance)
                print(f"      {group:<9} found {found}/{real} real events; "
                      f"ranking {precision:.2f} (guessing would give {chance:.2f})")
        if total_lines:
            print(f"  All together: {total_right}/{total_lines} = "
                  f"{100 * total_right / total_lines:.0f}% of lines right")
    else:
        print("\nOnly one match: it cannot be tested on an unseen match.")

    good = [group for group, values in lifts.items() if np.mean(values) >= MIN_LIFT]
    for group, values in lifts.items():
        verdict = "carries over" if group in good else "does NOT carry over"
        print(f"  {group}: {np.mean(values):.1f}x guessing on unseen matches, {verdict}")
    if not good and not args.save_anyway:
        print("\nNot saved: on a match it has not seen, the model is no better than "
              "guessing.\nIt has learnt what its training matches look like, not what "
              "an attack looks like.\nAdd more matches with a truth list (8-10 full "
              "matches is a fair start) and run this again.")
        return

    import joblib
    model = fit(matches)
    if not args.save_anyway:
        for key in ("classifiers", "share", "floor"):
            model[key] = {g: v for g, v in model[key].items() if g in good}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"\nModel trained on all {len(matches)} matches "
          f"({', '.join(model['classifiers'])}): {args.output}")


if __name__ == "__main__":
    main()

"""
Child-level evaluation: combine a child's recordings (chest spots) into one finding.

    python evaluate_children.py --sprsound-dir /path/to/SPRSound [--model sauti_mobilenet_v7.pth]

A child "has crackles" if paediatricians annotated crackles in ANY of the child's recordings, and likewise for
wheeze-type sounds. For each way of combining per-recording probabilities (analyzer.combine_spots supports these),
the decision threshold is chosen on VALIDATION children, then sensitivity / specificity are reported on TEST
children. Writes child_level_metrics.json.
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np
import torch

import train_robust as tr
from pediatric_data import sprsound_sound_recordings

WHEEZE, NORMAL, CRACKLE = 0, 1, 2           # indices in tr.CLASSES: Asthma, Normal, Pneumonia
STRATEGIES = ("max", "top2_mean", "mean")
MAX_SPOTS = 4                               # the app records up to 4 chest spots


def combine(values, strategy):
    values = sorted(values, reverse=True)
    if strategy == "max":
        return values[0]
    if strategy == "top2_mean":
        return float(np.mean(values[:2]))
    return float(np.mean(values))


@torch.no_grad()
def recording_probs(model, recordings):
    feats = tr.featurise(recordings)
    probs = tr.recording_probs(model, feats, torch.device("cpu"))
    return [p.numpy() if p is not None else None for p in probs]


def children(recordings, probs, rng):
    """child -> (truth {crackle, wheeze}, list of per-spot probs), at most MAX_SPOTS spots per child."""
    by_child = defaultdict(list)
    for r, p in zip(recordings, probs):
        if p is not None:
            by_child[r["patient"]].append((r, p))
    out = {}
    for child, items in by_child.items():
        rng.shuffle(items)
        items = items[:MAX_SPOTS]   # what one visit with the app would give
        truth = {"crackle": any(r["label"] == CRACKLE for r, _ in items),
                 "wheeze": any(r["label"] == WHEEZE for r, _ in items)}
        out[child] = (truth, [p for _, p in items])
    return out


def scores(kids, finding, strategy):
    idx = CRACKLE if finding == "crackle" else WHEEZE
    y = np.array([t[finding] for t, _ in kids.values()])
    s = np.array([combine([p[idx] for p in ps], strategy) for _, ps in kids.values()])
    return y, s


def sens_spec(y, s, threshold):
    pred = s >= threshold
    tp, fn = np.sum(pred & y), np.sum(~pred & y)
    tn, fp = np.sum(~pred & ~y), np.sum(pred & ~y)
    return (tp / (tp + fn) if tp + fn else None), (tn / (tn + fp) if tn + fp else None)


def pick_threshold(y, s, min_sensitivity):
    """Highest threshold (fewest false alarms) that keeps sensitivity >= min_sensitivity on validation."""
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        sens, _ = sens_spec(y, s, t)
        if sens is not None and sens >= min_sensitivity:
            return float(t)
    return 0.05


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=tr.MODEL_PATH)
    parser.add_argument("--sprsound-dir", required=True)
    parser.add_argument("--min-sensitivity", type=float, default=0.8,
                        help="screening target: thresholds keep at least this sensitivity on validation children")
    parser.add_argument("--out", default="child_level_metrics.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    model = tr.build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    splits = tr.split_by_patient(sprsound_sound_recordings(args.sprsound_dir, tr.CLASS_DIRS), seed=42)
    rng = random.Random(0)
    kids = {name: children(splits[name], recording_probs(model, splits[name]), rng) for name in ("val", "test")}

    results = {"model": args.model, "min_sensitivity_target": args.min_sensitivity,
               "children": {k: len(v) for k, v in kids.items()}, "findings": {}}
    for finding in ("crackle", "wheeze"):
        results["findings"][finding] = {}
        for strategy in STRATEGIES:
            threshold = pick_threshold(*scores(kids["val"], finding, strategy), args.min_sensitivity)
            y, s = scores(kids["test"], finding, strategy)
            sens, spec = sens_spec(y, s, threshold)
            results["findings"][finding][strategy] = {
                "threshold": threshold, "test_sensitivity": round(float(sens), 3) if sens is not None else None,
                "test_specificity": round(float(spec), 3) if spec is not None else None,
                "test_children_with_finding": int(y.sum()), "test_children": int(len(y))}
            print(finding, strategy, results["findings"][finding][strategy])
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

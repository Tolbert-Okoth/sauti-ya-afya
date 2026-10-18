"""
Noise-robustness benchmark on the held-out TEST patients.

    python evaluate_noise.py --noise-dir /path/to/noise [--model sauti_mobilenet_v4.pth] [--snr 10 0]

--noise-dir holds one sub-folder per noise group, e.g. voices/, animals/, impulsive/, steady/ with
audio files inside (we used ESC-50 folds 4-5 and the Free Spoken Digit Dataset; neither is used
for training). Two synthetic groups are always added: white hiss and 50 Hz mains hum.

Each test recording is mixed with a noise clip at the given signal-to-noise ratio and run through
the same pipeline and recording-level decision as the live engine. Reported per condition:
macro-recall and per-class sensitivity over usable recordings, and the share of recordings the
engine rejected as "record again".
"""
import argparse
import importlib.util
import json
import os
import random

import numpy as np
import torch

import train_robust as tr
from noise_mix import mix, noise_groups, synthetic_noise

SR = 16000


def load_pipeline(path):
    if not path:
        import audio_pipeline as ap
        return ap
    spec = importlib.util.spec_from_file_location("legacy_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def usable_windows(ap, y):
    if hasattr(ap, "analysis_windows"):
        return ap.analysis_windows(y)[0]
    return [w for w in ap.split_windows(y) if ap.window_quality(w)[0]]   # pipeline v2


@torch.no_grad()
def score(model, ap, signals, labels):
    confusion = np.zeros((3, 3), dtype=int)
    rejected = 0
    for y, label in zip(signals, labels):
        windows = usable_windows(ap, y)
        if not windows:
            rejected += 1
            continue
        probs = torch.softmax(model(torch.stack([ap.window_tensor(w) for w in windows])), 1).mean(0)
        confusion[label, int(probs.argmax())] += 1
    recall = {c: (confusion[i, i] / confusion[i].sum() if confusion[i].sum() else None)
              for i, c in enumerate(tr.CLASSES)}
    valid = [v for v in recall.values() if v is not None]
    return {"macro_recall": round(float(np.mean(valid)), 3) if valid else None,
            "recall": {k: (round(float(v), 3) if v is not None else None) for k, v in recall.items()},
            "rejected_share": round(rejected / len(signals), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=tr.MODEL_PATH)
    parser.add_argument("--noise-dir", required=True)
    parser.add_argument("--snr", type=float, nargs="*", default=[10, 0], help="pass no values for clean only")
    parser.add_argument("--data", default="../raw_data_phase4")
    parser.add_argument("--legacy-pipeline", default="", help="path to an older audio_pipeline.py")
    parser.add_argument("--out", default="noise_robustness.json")
    parser.add_argument("--sprsound-dir", default="", help="include SPRSound (paediatric) recordings")
    parser.add_argument("--age-group", default="", help="only test recordings of this group: child / adult / unknown")
    args = parser.parse_args()

    ap = load_pipeline(args.legacy_pipeline)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    assert checkpoint["pipeline_version"] == ap.PIPELINE_VERSION, "model / pipeline version mismatch"
    model = tr.build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    recordings = tr.collect_recordings(args.data)
    if args.sprsound_dir:
        recordings += tr.sprsound_recordings(args.sprsound_dir, tr.CLASS_DIRS)
    test = tr.split_by_patient(recordings, seed=42)["test"]
    if args.age_group:
        test = [r for r in test if r["age_group"] == args.age_group]
    clean = [ap.load_audio(r["path"]) for r in test]
    labels = [r["label"] for r in test]

    groups = noise_groups(args.noise_dir) if args.snr else {}
    rng = np.random.default_rng(0)
    results = {"model": os.path.basename(args.model), "pipeline_version": ap.PIPELINE_VERSION,
               "test_recordings": len(test), "clean": score(model, ap, clean, labels)}
    print("clean", results["clean"])
    for group in (list(groups) + ["white_hiss", "mains_hum"]) if args.snr else []:
        for snr in args.snr:
            noisy = []
            for y in clean:
                if group in groups:
                    noise = ap.load_audio(random.Random(len(noisy)).choice(groups[group]))
                else:
                    noise = synthetic_noise(group, len(y), rng)
                noisy.append(mix(y, noise, snr, rng))
            key = f"{group}@{snr:g}dB"
            results[key] = score(model, ap, noisy, labels)
            print(key, results[key])
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

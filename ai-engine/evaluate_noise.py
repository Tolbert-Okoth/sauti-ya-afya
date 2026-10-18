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
import glob
import importlib.util
import json
import os
import random

import numpy as np
import torch

import train_robust as tr

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


def synthetic_noise(kind, n, rng):
    t = np.arange(n) / SR
    if kind == "white_hiss":
        return rng.standard_normal(n).astype(np.float32)
    if kind == "mains_hum":
        return sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 8)).astype(np.float32)
    raise ValueError(kind)


def mix(y, noise, snr_db, rng):
    if len(noise) < len(y):
        noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))
    start = rng.integers(0, len(noise) - len(y) + 1)
    noise = noise[start:start + len(y)]
    p_signal, p_noise = np.mean(y ** 2) + 1e-12, np.mean(noise ** 2) + 1e-12
    mixed = y + noise * np.sqrt(p_signal / (p_noise * 10 ** (snr_db / 10)))
    peak = np.max(np.abs(mixed))
    return (mixed / peak * 0.9 if peak > 0.99 else mixed).astype(np.float32)   # no clipping from mixing


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
    parser.add_argument("--snr", type=float, nargs="+", default=[10, 0])
    parser.add_argument("--data", default="../raw_data_phase4")
    parser.add_argument("--legacy-pipeline", default="", help="path to an older audio_pipeline.py")
    parser.add_argument("--out", default="noise_robustness.json")
    args = parser.parse_args()

    ap = load_pipeline(args.legacy_pipeline)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    assert checkpoint["pipeline_version"] == ap.PIPELINE_VERSION, "model / pipeline version mismatch"
    model = tr.build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    test = tr.split_by_patient(tr.collect_recordings(args.data), seed=42)["test"]
    clean = [ap.load_audio(r["path"]) for r in test]
    labels = [r["label"] for r in test]

    groups = {g: sorted(glob.glob(os.path.join(args.noise_dir, g, "*")))
              for g in sorted(os.listdir(args.noise_dir)) if os.path.isdir(os.path.join(args.noise_dir, g))}
    rng = np.random.default_rng(0)
    results = {"model": os.path.basename(args.model), "pipeline_version": ap.PIPELINE_VERSION,
               "test_recordings": len(test), "clean": score(model, ap, clean, labels)}
    print("clean", results["clean"])
    for group in list(groups) + ["white_hiss", "mains_hum"]:
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

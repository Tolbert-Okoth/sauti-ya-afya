"""
Train the lung-sound classifier with an honest, patient-level evaluation.

    python train_robust.py [--data ../raw_data_phase4] [--epochs 12]

- Uses audio_pipeline.py, exactly like the live analyzer.
- Splits train / validation / test BY PATIENT, so no patient (and none of the
  three stethoscope-filter copies of a KAUH recording) appears in two splits.
- Picks the epoch with the best validation macro-recall and reports test
  metrics at recording level, aggregated the same way analyzer.py does.
- Writes the checkpoint (weights + classes + pipeline version + metrics) and
  model_metrics.json.
"""
import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision import models

import audio_pipeline as ap
from pediatric_data import sprsound_recordings
from noise_mix import SYNTHETIC, mix, noise_groups, synthetic_noise

CLASS_DIRS = ['asthma_wheeze', 'normal', 'pneumonia']
CLASSES = ['Asthma', 'Normal', 'Pneumonia']   # index order used by analyzer.py
AUDIO_EXTS = ('.wav', '.mp3', '.webm', '.m4a', '.flac', '.ogg', '.aac')
EXCLUSIONS_FILE = 'data_exclusions.txt'
MODEL_PATH = 'sauti_mobilenet_v6.pth'
METRICS_PATH = 'model_metrics.json'


def patient_key(filename):
    """Group recordings that come from the same person."""
    m = re.match(r'^(\d{3})_', filename)                      # ICBHI: 101_1b1_Al_sc_Meditron.wav
    if m:
        return f"icbhi:{m.group(1)}"
    m = re.match(r'^[BDE]P(\d+)_', filename)                  # KAUH: BP12_/DP12_/EP12_ = same recording
    if m:
        return f"kauh:{m.group(1)}"
    m = re.match(r'^P(\d+)(Pneumonia|Wheezing|Healthy)', filename)  # P14WheezingIE_69.wav
    if m:
        return f"pset:{m.group(2)}:{m.group(1)}"
    return f"file:{filename}"                                 # one-off clips


def source_and_age(filename):
    """(source, age_group) for raw_data_phase4 files. Only KAUH names carry an age."""
    key = patient_key(filename).split(':')[0]
    if key == 'kauh':
        m = re.search(r',(\d+),[MF]\.wav$', filename)
        if m:
            return 'kauh', 'child' if int(m.group(1)) < 18 else 'adult'
    return {'icbhi': 'icbhi', 'kauh': 'kauh', 'pset': 'pset'}.get(key, 'clips'), 'unknown'


def load_exclusions(path):
    excluded = set()
    if os.path.exists(path):
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.split('#', 1)[0].strip()
                if line:
                    excluded.add(line)
    return excluded


def collect_recordings(data_dir):
    excluded = load_exclusions(EXCLUSIONS_FILE)
    recordings, skipped = [], []
    for label, class_dir in enumerate(CLASS_DIRS):
        folder = os.path.join(data_dir, class_dir)
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(AUDIO_EXTS):
                continue
            rel = f"{class_dir}/{name}"
            if rel in excluded:
                skipped.append(rel)
                continue
            source, age_group = source_and_age(name)
            recordings.append({'path': os.path.join(folder, name), 'label': label, 'patient': patient_key(name),
                               'source': source, 'age_group': age_group})
    print(f"Excluded {len(skipped)} files listed in {EXCLUSIONS_FILE}")
    return recordings


def split_by_patient(recordings, seed, val_frac=0.15, test_frac=0.15):
    """Stratified group split: each (source, class) group's patients are divided train/val/test,
    so every dataset - in particular the paediatric one - has its own held-out patients."""
    rng = random.Random(seed)
    patients_by_class = defaultdict(set)
    for r in recordings:
        patients_by_class[(r['source'], r['label'])].add(r['patient'])
    assign = {}
    for _, patients in sorted(patients_by_class.items()):
        patients = sorted(patients)
        rng.shuffle(patients)
        n = len(patients)
        n_test = max(1, round(n * test_frac))
        n_val = max(1, round(n * val_frac))
        for i, p in enumerate(patients):
            assign[p] = 'test' if i < n_test else 'val' if i < n_test + n_val else 'train'
    splits = defaultdict(list)
    for r in recordings:
        splits[assign[r['patient']]].append(r)
    return splits


def featurise(recordings, noise_dir="", noise_copies=0, seed=0):
    """Decode each recording once and turn every usable 5 s window into a tensor.

    With noise_dir, each recording also gets `noise_copies` versions mixed with a random
    noise clip (or synthetic hiss / mains hum) at 0-20 dB SNR, run through the same
    pipeline. Only used for TRAINING recordings; validation and test stay clean.
    """
    groups = noise_groups(noise_dir) if noise_dir else {}
    choices = list(groups) + list(SYNTHETIC)
    rng = np.random.default_rng(seed)
    per_recording = []
    for i, r in enumerate(recordings):
        try:
            y = ap.load_audio(r['path'])
        except ap.AudioDecodeError as e:
            raise SystemExit(f"Failed to decode {r['path']}: {e}")
        versions = [y]
        for _ in range(noise_copies if noise_dir else 0):
            group = choices[rng.integers(len(choices))]
            noise = (ap.load_audio(groups[group][rng.integers(len(groups[group]))]) if group in groups
                     else synthetic_noise(group, len(y), rng))
            versions.append(mix(y, noise, rng.uniform(0, 20), rng))
        tensors = []
        for v in versions:
            usable, _, _, _ = ap.analysis_windows(v)
            tensors.extend(ap.window_tensor(w).half() for w in usable)   # float16 halves memory
        per_recording.append(tensors)
        if (i + 1) % 100 == 0:
            print(f"   featurised {i + 1}/{len(recordings)}")
    return per_recording


def windows_dataset(recordings, feats):
    xs, ys = [], []
    for r, tensors in zip(recordings, feats):
        for t in tensors:
            xs.append(t)
            ys.append(r['label'])
    return torch.stack(xs), torch.tensor(ys)


def spec_augment(batch):
    """Random frequency and time masks plus a small circular time shift."""
    batch = batch.clone()
    for img in batch:
        f0 = random.randint(0, 200); img[:, f0:f0 + random.randint(0, 20), :] = 0
        t0 = random.randint(0, 200); img[:, :, t0:t0 + random.randint(0, 20)] = 0
    return torch.roll(batch, shifts=random.randint(-20, 20), dims=3)


def build_model(pretrained=True):
    weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(model.last_channel, len(CLASSES)))
    return model


@torch.no_grad()
def recording_probs(model, feats, device):
    """Mean softmax over a recording's windows (same aggregation as analyzer.py)."""
    model.eval()
    out = []
    for tensors in feats:
        if not tensors:
            out.append(None)
            continue
        probs = torch.softmax(model(torch.stack(tensors).float().to(device)), dim=1).mean(dim=0)
        out.append(probs.cpu())
    return out


def evaluate_groups(model, recordings, feats, device):
    """Overall metrics plus the same metrics per age group (child / adult / unknown)."""
    probs = recording_probs(model, feats, device)
    result = {'all': evaluate(recordings, probs)}
    for group in sorted({r['age_group'] for r in recordings}):
        idx = [i for i, r in enumerate(recordings) if r['age_group'] == group]
        result[group] = evaluate([recordings[i] for i in idx], [probs[i] for i in idx])
    return result


def cap_per_class(recordings, max_per_class, seed):
    """Keep at most max_per_class recordings per class, sampling whole patients (training only)."""
    if not max_per_class:
        return recordings
    rng = random.Random(seed)
    kept = []
    for label in range(len(CLASSES)):
        by_patient = defaultdict(list)
        for r in recordings:
            if r['label'] == label:
                by_patient[r['patient']].append(r)
        patients = sorted(by_patient)
        rng.shuffle(patients)
        count = 0
        for p in patients:
            if count >= max_per_class:
                break
            kept.extend(by_patient[p])
            count += len(by_patient[p])
    return kept


def evaluate(recordings, probs):
    confusion = [[0] * len(CLASSES) for _ in CLASSES]
    unusable = 0
    for r, p in zip(recordings, probs):
        if p is None:
            unusable += 1
            continue
        confusion[r['label']][int(p.argmax())] += 1
    recall, precision = {}, {}
    for i, c in enumerate(CLASSES):
        row = sum(confusion[i])
        col = sum(confusion[j][i] for j in range(len(CLASSES)))
        recall[c] = confusion[i][i] / row if row else None
        precision[c] = confusion[i][i] / col if col else None
    valid = [v for v in recall.values() if v is not None]
    total = sum(map(sum, confusion))
    return {
        'recordings': total,
        'unusable_recordings': unusable,
        'patients': len({r['patient'] for r in recordings}),
        'accuracy': sum(confusion[i][i] for i in range(len(CLASSES))) / total if total else None,
        'macro_recall': sum(valid) / len(valid) if valid else None,
        'recall_sensitivity': recall,
        'precision': precision,
        'confusion_matrix': {'rows_true_cols_pred': CLASSES, 'matrix': confusion},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='../raw_data_phase4')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise-aug-dir', default='',
                        help='noise clips (one sub-folder per group) mixed into TRAINING recordings')
    parser.add_argument('--noise-copies', type=int, default=2, help='noisy copies per training recording')
    parser.add_argument('--sprsound-dir', default='', help='SPRSound checkout (paediatric data, pediatric_data.py)')
    parser.add_argument('--max-train-per-class', type=int, default=0,
                        help='cap training recordings per class (whole patients), 0 = no cap')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    recordings = collect_recordings(args.data)
    if args.sprsound_dir:
        recordings += sprsound_recordings(args.sprsound_dir, CLASS_DIRS)
    splits = split_by_patient(recordings, args.seed)
    splits['train'] = cap_per_class(splits['train'], args.max_train_per_class, args.seed)
    for name in ('train', 'val', 'test'):
        for group in sorted({r['age_group'] for r in splits[name]}):
            subset = [r for r in splits[name] if r['age_group'] == group]
            counts = Counter(CLASSES[r['label']] for r in subset)
            print(f"{name:5s} {group:7s}: {len(subset)} recordings, "
                  f"{len({r['patient'] for r in subset})} patients, {dict(counts)}")
    assert not ({r['patient'] for r in splits['train']} & {r['patient'] for r in splits['test']}), "patient leak"

    print("Featurising (decoding + spectrograms)...")
    feats = {'train': featurise(splits['train'], args.noise_aug_dir, args.noise_copies, args.seed),
             'val': featurise(splits['val']),
             'test': featurise(splits['test'])}
    x_train, y_train = windows_dataset(splits['train'], feats['train'])
    print(f"Training windows: {len(y_train)}")

    # Balance classes in the TRAINING set only; val/test stay untouched.
    class_counts = torch.bincount(y_train, minlength=len(CLASSES)).float()
    sample_weights = (1.0 / class_counts)[y_train]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(y_train), replacement=True)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=args.batch_size, sampler=sampler)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_score, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = spec_augment(xb.float()).to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        val_groups = evaluate_groups(model, splits['val'], feats['val'], device)
        # Model selection on CHILDREN when available: they are the target population
        val = val_groups.get('child', val_groups['all'])
        print(f"Epoch {epoch}/{args.epochs}  train_loss={total_loss / len(y_train):.4f}  "
              + "  ".join(f"val_{g}_macro={m['macro_recall']:.3f}" for g, m in val_groups.items() if m['macro_recall'] is not None)
              + f"  val_recall={ {k: round(v, 2) for k, v in val['recall_sensitivity'].items() if v is not None} }")
        if val['macro_recall'] > best_score:
            best_score, best_epoch = val['macro_recall'], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    metrics = {
        'pipeline_version': ap.PIPELINE_VERSION,
        'classes': CLASSES,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'noise_augmentation': {'dir': os.path.basename(os.path.normpath(args.noise_aug_dir)) if args.noise_aug_dir else None,
                               'copies': args.noise_copies if args.noise_aug_dir else 0, 'snr_db': [0, 20]},
        'sprsound': bool(args.sprsound_dir),
        'max_train_per_class': args.max_train_per_class,
        'note': ("Held-out patients only, reported overall and per age group. 'child' = SPRSound "
                 "(Shanghai, ages 1 month-18 years) plus KAUH patients under 18; 'unknown' = ICBHI and "
                 "clips without age information (mostly adults)."),
        'validation': evaluate_groups(model, splits['val'], feats['val'], device),
        'test': evaluate_groups(model, splits['test'], feats['test'], device),
    }
    torch.save({'state_dict': best_state, 'classes': CLASSES,
                'pipeline_version': ap.PIPELINE_VERSION, 'metrics': metrics}, MODEL_PATH)
    with open(METRICS_PATH, 'w', encoding='utf-8') as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps(metrics['test'], indent=2))
    print(f"Saved {MODEL_PATH} and {METRICS_PATH} (best epoch {best_epoch})")


if __name__ == '__main__':
    main()

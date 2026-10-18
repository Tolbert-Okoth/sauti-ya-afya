"""
SPRSound (SJTU Paediatric Respiratory Sound Database) loader.

    https://github.com/SJTU-YONGFU-RESEARCH-GRP/SPRSound   (CC-BY-4.0, cite the TBioCAS 2022 paper)

Children aged 1 month - 18 years, Shanghai Children's Medical Center, Yunting II stethoscope (8 kHz).
Recording file names: <patient>_<age>_<sex>_<location>_<number>.wav. Patient-level diagnoses come from
the "Patient Summary" CSVs; recordings rated "Poor Quality" by the annotating paediatricians are skipped.
"""
import csv
import glob
import json
import os

# SPRSound diagnosis -> engine class directory name (same names as raw_data_phase4/)
DIAGNOSIS_TO_CLASS = {
    "pneumonia (non-severe)": "pneumonia",
    "pneumonia (severe)": "pneumonia",
    "asthma": "asthma_wheeze",
    "bronchiolitis": "asthma_wheeze",   # grouped with wheeze, as in raw_data_phase4
    "control group": "normal",
}
# Everything else (bronchitis, URTI, bronchiectasis, ...) does not fit the 3 classes and is left out.

RECORDING_DIRS = [
    ("BioCAS2022/train2022_wav", "BioCAS2022/train2022_json"),
    ("BioCAS2022/test2022_wav", "BioCAS2022/test2022_json"),
    ("BioCAS2023/test2023_wav", "BioCAS2023/test2023_json"),
    ("BioCAS2024/test2024_wav", "BioCAS2024/test2024_json"),
    ("BioCAS2025/test2025_wav", "BioCAS2025/test2025_json"),
]


def load_diagnoses(root):
    """patient id -> lower-case diagnosis. Patients with conflicting diagnoses are dropped."""
    seen = {}
    for path in glob.glob(os.path.join(root, "Patient Summary", "*.csv")):
        with open(path, encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                pid = row["patient_num"].strip().zfill(8)
                seen.setdefault(pid, set()).add(row["disease"].strip().lower())
    return {pid: next(iter(d)) for pid, d in seen.items() if len(d) == 1}


def sprsound_recordings(root, class_dirs):
    """Recordings as dicts compatible with train_robust.collect_recordings()."""
    diagnoses = load_diagnoses(root)
    recordings, skipped = [], {"poor_quality": 0, "no_diagnosis": 0, "other_diagnosis": 0, "adult": 0}
    for wav_dir, json_dir in RECORDING_DIRS:
        annotations = {os.path.basename(p)[:-5]: p
                       for p in glob.glob(os.path.join(root, json_dir, "**", "*.json"), recursive=True)}
        for wav in sorted(glob.glob(os.path.join(root, wav_dir, "*.wav"))):
            name = os.path.basename(wav)[:-4]
            pid, age = name.split("_")[0].zfill(8), float(name.split("_")[1])
            if name in annotations:
                with open(annotations[name], encoding="utf-8") as fh:
                    if json.load(fh).get("record_annotation") == "Poor Quality":
                        skipped["poor_quality"] += 1
                        continue
            if pid not in diagnoses:
                skipped["no_diagnosis"] += 1
                continue
            cls = DIAGNOSIS_TO_CLASS.get(diagnoses[pid])
            if cls is None:
                skipped["other_diagnosis"] += 1
                continue
            if age >= 18:
                skipped["adult"] += 1
                continue
            recordings.append({"path": wav, "label": class_dirs.index(cls), "patient": f"sprs:{pid}",
                               "source": "sprsound", "age_group": "child", "age": age})
    print(f"SPRSound: {len(recordings)} recordings used, skipped {skipped}")
    return recordings


# ---------------------------------------------------------------------------
# Sound-pattern labels (what is AUDIBLE), from the paediatricians' event annotations.
# A child's diagnosis is not audible in most single recordings (70-80% of recordings
# of sick children sound normal), so the model learns sounds; diagnosis is per child.
# ---------------------------------------------------------------------------
CONTINUOUS_EVENTS = {"Wheeze", "Rhonchi", "Stridor"}        # CAS -> "asthma_wheeze" class
CRACKLE_EVENTS = {"Fine Crackle", "Coarse Crackle"}         # DAS -> "pneumonia" class
BOTH_EVENTS = {"Wheeze+Crackle"}
SEVERITY = {"normal": 0, "asthma_wheeze": 1, "pneumonia": 2}  # crackles outrank wheeze (pneumonia pathway)


def event_class(event_type):
    if event_type in CRACKLE_EVENTS or event_type in BOTH_EVENTS:
        return "pneumonia"
    if event_type in CONTINUOUS_EVENTS:
        return "asthma_wheeze"
    return "normal"


def worst(classes):
    return max(classes, key=SEVERITY.get, default="normal")


def sprsound_sound_recordings(root, class_dirs):
    """Every annotated SPRSound recording, labelled by audible sound pattern.

    Each recording dict carries 'events' [(start_s, end_s, class_dir)] so training can label
    each 5 s window by what is heard inside it; 'label' is the recording-level pattern and
    'patient_label' the child's worst pattern (used to stratify the split).
    """
    recordings, skipped = [], {"poor_quality": 0, "no_annotation": 0, "adult": 0}
    for wav_dir, json_dir in RECORDING_DIRS:
        annotations = {os.path.basename(p)[:-5]: p
                       for p in glob.glob(os.path.join(root, json_dir, "**", "*.json"), recursive=True)}
        for wav in sorted(glob.glob(os.path.join(root, wav_dir, "*.wav"))):
            name = os.path.basename(wav)[:-4]
            pid, age = name.split("_")[0].zfill(8), float(name.split("_")[1])
            if name not in annotations:
                skipped["no_annotation"] += 1
                continue
            with open(annotations[name], encoding="utf-8") as fh:
                ann = json.load(fh)
            if ann.get("record_annotation") == "Poor Quality":
                skipped["poor_quality"] += 1
                continue
            events = [(int(e["start"]) / 1000, int(e["end"]) / 1000, event_class(e["type"]))
                      for e in ann.get("event_annotation") or []]
            if not events:
                skipped["no_annotation"] += 1
                continue
            if age >= 18:
                skipped["adult"] += 1
                continue
            recordings.append({"path": wav, "label": class_dirs.index(worst(c for _, _, c in events)),
                               "events": [(s, e, class_dirs.index(c)) for s, e, c in events],
                               "patient": f"sprs:{pid}", "source": "sprsound", "age_group": "child", "age": age})
    by_patient = {}
    for r in recordings:
        by_patient[r["patient"]] = max(by_patient.get(r["patient"], 0), SEVERITY[class_dirs[r["label"]]])
    inverse = {v: k for k, v in SEVERITY.items()}
    for r in recordings:
        r["patient_label"] = class_dirs.index(inverse[by_patient[r["patient"]]])
    print(f"SPRSound (sound labels): {len(recordings)} recordings, {len(by_patient)} children, skipped {skipped}")
    return recordings


def window_label(events, start_s, end_s, class_dirs):
    """Worst sound class (index) among events overlapping [start_s, end_s); None if no event overlaps."""
    overlapping = [c for s, e, c in events if s < end_s and e > start_s]
    if not overlapping:
        return None
    return max(overlapping, key=lambda c: SEVERITY[class_dirs[c]])

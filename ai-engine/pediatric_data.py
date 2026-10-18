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

#!/usr/bin/env bash
# Rebuild the noise sets used for training augmentation and the noise benchmark.
#   bash tools/build_noise_sets.sh /data/noise
# -> /data/noise/noise_train/{voices,animals,impulsive,steady}   ESC-50 folds 1-3 (training augmentation)
# -> /data/noise/noise_eval/{voices,animals,impulsive,steady}    ESC-50 folds 4-5 + spoken digits (benchmark only)
# ESC-50 is CC BY-NC 3.0 (non-commercial); the Free Spoken Digit Dataset is CC BY-SA 4.0.
set -euo pipefail

OUT="${1:?usage: build_noise_sets.sh OUTPUT_DIR}"
ESC_RAW="https://raw.githubusercontent.com/karolpiczak/ESC-50/master"
FSDD_RAW="https://raw.githubusercontent.com/Jakobovski/free-spoken-digit-dataset/master/recordings"
CLIPS_PER_FOLD=4

mkdir -p "$OUT/esc50"
curl -sSfL -o "$OUT/esc50/esc50.csv" "$ESC_RAW/meta/esc50.csv"

group_of() {
  case "$1" in
    crying_baby|laughing) echo voices ;;
    rooster|dog) echo animals ;;
    door_wood_knock|footsteps|clock_tick) echo impulsive ;;
    engine|rain|wind) echo steady ;;
    *) echo "" ;;
  esac
}

for split in noise_train noise_eval; do
  mkdir -p "$OUT/$split"/{voices,animals,impulsive,steady}
done

# filename,fold,target,category,...  (first CLIPS_PER_FOLD clips of each category per fold)
tail -n +2 "$OUT/esc50/esc50.csv" | sort -t, -k4,4 -k2,2n -k1,1 | awk -F, -v n="$CLIPS_PER_FOLD" '
  { key = $4 "," $2; if (++seen[key] <= n) print $1 "," $2 "," $4 }' |
while IFS=, read -r file fold category; do
  group="$(group_of "$category")"
  [ -n "$group" ] || continue
  split=$([ "$fold" -le 3 ] && echo noise_train || echo noise_eval)
  [ -f "$OUT/$split/$group/$file" ] || curl -sSfL -o "$OUT/$split/$group/$file" "$ESC_RAW/audio/$file"
done

# Speech is used ONLY in the benchmark, so the voice test uses voices the model never trained with
for speaker in jackson nicolas theo yweweler george lucas; do
  for digit in 0 3 5 7 9; do
    for take in 0 1; do
      f="${digit}_${speaker}_${take}.wav"
      [ -f "$OUT/noise_eval/voices/$f" ] || curl -sSfL -o "$OUT/noise_eval/voices/$f" "$FSDD_RAW/$f"
    done
  done
done

for split in noise_train noise_eval; do
  for g in voices animals impulsive steady; do
    echo "$split/$g: $(ls "$OUT/$split/$g" | wc -l) clips"
  done
done

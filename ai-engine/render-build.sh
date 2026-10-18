#!/usr/bin/env bash
# Exit on error
set -o errexit

# 1. Update & Install BOTH Audio Libraries
# ffmpeg = Required for WebM decoding (Browser audio)
# libsndfile1 = Only needed by optional training tools
apt-get update && apt-get install -y libsndfile1 ffmpeg

# 2. Upgrade pip (Good practice)
pip install --upgrade pip

# 3. Install Python Dependencies
pip install -r requirements.txt
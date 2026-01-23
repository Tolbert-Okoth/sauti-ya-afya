#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install system dependency for audio
apt-get update && apt-get install -y libsndfile1

# Install Python dependencies
pip install -r requirements.txt
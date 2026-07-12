#!/usr/bin/env bash
# Fetch the AddBiomechanics core dataset (biomechanically-processed human motion,
# permissive license) — the pose source for our own ANNY 15-sensor calibration data.
# One HTTP archive (~1.2 GB of B3D files); resumable with curl -C -.
set -e
DEST="${1:-D:/addbiomechanics.zip}"
URL="http://archive.simtk.org/addbiomechanics/addbiomechanics.zip"
echo "fetching $URL -> $DEST"
curl -L -C - -o "$DEST" "$URL"
echo "extracting next to the zip..."
OUTDIR="$(dirname "$DEST")/addbiomechanics"
mkdir -p "$OUTDIR"
unzip -n "$DEST" -d "$OUTDIR"
echo "done: $OUTDIR  (B3D files under test/ … )"

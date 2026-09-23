#!/bin/bash
# Build the TinySoundFont per-track renderer.
set -e
cd "$(dirname "$0")"
gcc -O2 -o tsfrender tsfrender.c -lm
echo "built $(pwd)/tsfrender"

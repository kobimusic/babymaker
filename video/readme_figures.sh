#!/bin/bash
# The README's figures (docs/img/), cut from the long explainer. Run from the repo root
# after build.py long: video/readme_figures.sh [video/out/babymaker-explained-silent.mp4] [docs/img]
# Stills: the frame, its chapter label (top right) painted out, trimmed to the ink,
# a paper margin, 256 colours. GIFs: a fixed crop, sped up where the clip is long.
set -euo pipefail
V=${1:-video/out/babymaker-explained-silent.mp4}; O=${2:-docs/img}
mkdir -p "$O"; F=$(mktemp --suffix=.png); trap 'rm -f "$F"' EXIT
PAPER="rgb(252,252,249)"
still() {  # name time [bottom of the label box: 148 clears the descenders of "why" and "alignment"]
  ffmpeg -v error -y -ss "$2" -i "$V" -frames:v 1 -f image2 -c:v png "$F"
  convert "$F" -fill "$PAPER" -draw "rectangle 1440,0 1919,${3:-148}" \
    -fuzz 3% -trim +repage -bordercolor "$PAPER" -border 40 +dither -colors 256 -strip -define png:exclude-chunks=date,time "PNG8:$O/$1.png"
}
gif() {  # name start end crop(w:h:x:y) width speed fps [paint]
  local vf="setpts=PTS/$6,fps=$7,crop=$4,scale=$5:-1:flags=lanczos"
  [ -n "${8:-}" ] && vf="format=rgb24,drawbox=$8:color=0xFCFCF9:t=fill,$vf"  # in RGB, or the box is a shade off the paper
  ffmpeg -v error -y -ss "$2" -to "$3" -i "$V" -vf "$vf,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" -loop 0 "$O/$1.gif"
}
still result 404
still why 59
still mclt 190 135   # its waveform starts at y 140
still dtw 284
still ot 698
still resynth 399
still speed 528
gif satie 592 616 1810:872:60:48 900 2 12
gif dotabata 640 656 1850:872:20:48 800 2 10
gif triangle 0 14.5 970:683:480:158 560 1 10
gif envelope 213.5 238 1400:600:300:50 800 1.5 12 1440:0:480:135
ls -la "$O"

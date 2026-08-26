#!/bin/bash
set -e
MAP_DIR="$(rospack find raicom2026)/data/maps"
mkdir -p "$MAP_DIR"
OUT="$MAP_DIR/competition_map"
echo "[save_map] 正在抓取 /map -> $OUT.pgm/.yaml ..."
rosrun map_server map_saver map:=/map -f "$OUT"
echo "[save_map] 完成"
ls -la "$MAP_DIR"

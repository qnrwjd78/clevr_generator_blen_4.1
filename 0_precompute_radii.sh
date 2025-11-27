#!/bin/bash
#
# Helper script to generate per-shape radius metadata and preview renders.
# Usage:
#   sh 0_precompute_radii.sh [blender_path] [properties_json] [shape_dir] [base_scene] [preview_dir] [radius_json]
#

blender_path="${1:-$HOME/blender-4.1.0-linux-x64}"
properties_json="${2:-data/properties.json}"
shape_dir="${3:-data/shapes}"
base_scene="${4:-data/base_scene.blend}"
preview_dir="${5:-output/radius_previews}"
radius_json="${6:-data/shape_radii.json}"

if [ ! -x "$blender_path/blender" ] && [ ! -x "$blender_path" ]; then
  echo "Blender executable not found at $blender_path or $blender_path/blender"
  exit 1
fi

blender_exec="$blender_path"
if [ -d "$blender_path" ]; then
  blender_exec="$blender_path/blender"
fi

mkdir -p "$(dirname "$radius_json")"
mkdir -p "$preview_dir"

"$blender_exec" --background --python precompute_radii.py -- \
  --base_scene_blendfile "$base_scene" \
  --properties_json "$properties_json" \
  --shape_dir "$shape_dir" \
  --preview_dir "$preview_dir" \
  --output_radius_json "$radius_json"

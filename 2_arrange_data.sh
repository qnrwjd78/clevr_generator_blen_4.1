# Copyright 2020 Azade Farshad
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

# Original variable setup (same as your script)
num=0
res_dir='./MyClevr'

# Ensure directory structure exists
mkdir -p $res_dir/source/images
mkdir -p $res_dir/source/scenes
mkdir -p $res_dir/target/images
mkdir -p $res_dir/target/scenes

# Loop exactly like original script
for j in */; do

  # Find all target images exactly like original
  find "$(pwd)/${j}images" -iname "*_target.png" | sort -u | while read p; do

    # -----------------------------
    # TARGET COPY (same structure)
    # -----------------------------
    cp "$p" "$res_dir/target/images/$num.png"

    q="$(pwd)/${j}scenes/${p##*/}"
    q="${q%%.*}.json"
    cp "$q" "$res_dir/target/scenes/$num.json"

    # -----------------------------
    # SOURCE COPY (added)
    # -----------------------------
    base="${p##*/}"                     # CLEVR_new_000000_target.png
    prefix="${base%_target.png}"        # CLEVR_new_000000

    source_img="$(pwd)/${j}images/${prefix}_source.png"
    source_json="$(pwd)/${j}scenes/${prefix}_source.json"

    if [ -f "$source_img" ]; then
        cp "$source_img" "$res_dir/source/images/$num.png"
    else
        echo "WARNING: Missing source image: $source_img"
    fi

    if [ -f "$source_json" ]; then
        cp "$source_json" "$res_dir/source/scenes/$num.json"
    else
        echo "WARNING: Missing source JSON: $source_json"
    fi

    # -----------------------------
    # Increment safely (WSL compatible)
    # -----------------------------
    num=$((num+1))

    # Debug output (same as original)
    echo "$p"
    echo "$q"
    echo "$num"

  done

  # Recompute num like original script
  num=$(ls -1 "$res_dir/target/images/" | wc -l)

done

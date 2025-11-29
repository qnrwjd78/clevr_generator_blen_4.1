#!/bin/bash
#
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

# Load configuration
source "$(dirname "$0")/config.sh"

mode="$1" #removal, replacement, addition, rel_change
blender_path="${2:-$BLENDER_PATH}" # Use path from config if not provided
merge="$3"
NUM_IMAGES=${NUM_IMAGES:-10}  # Set the number of images to generate (default 10)
NUM_PROCS=${NUM_PROCS:-${NUM_GPUS:-1}} # Default to NUM_GPUS or 1 if not set

if [ "$merge" ]; then 
    out_folder="./output";
else
    out_folder="./output_$mode";
fi

# Function to count files safely
count_files() {
    if [ -d "$1" ]; then
        ls -1 "$1" 2>/dev/null | wc -l
    else
        echo 0
    fi
}

if [ -n "$START_IDX" ]; then
    start=$START_IDX
elif [ "$merge" ]; then 
    start=$(count_files "$out_folder/images/")
else
    c1=$(count_files "$out_folder/../output_rel_change/images/")
    c2=$(count_files "$out_folder/../output_removal/images/")
    c3=$(count_files "$out_folder/../output_replacement/images/")
    c4=$(count_files "$out_folder/../output_addition/images/")
    c5=$(count_files "$out_folder/../output_random/images/")
    start=$(expr $c1 + $c2 + $c3 + $c4 + $c5)
    start=$((start/2));
fi

echo "Generating $NUM_IMAGES images starting from index $start with $NUM_PROCS processes..."

# Log file for monitoring
LOG_FILE="parallel_render.log"
rm -f "$LOG_FILE"
touch "$LOG_FILE"

# Array to store PIDs
pids=()

# Calculate images per process
imgs_per_proc=$((NUM_IMAGES / NUM_PROCS))
remainder=$((NUM_IMAGES % NUM_PROCS))

current_start=$start

# Launch render processes
for ((i=0; i<NUM_PROCS; i++)); do
    count=$imgs_per_proc
    if [ $i -lt $remainder ]; then
        count=$((count + 1))
    fi

    if [ $count -gt 0 ]; then
        "$blender_path/blender" --background --python image_generation/render_clevr.py --             --width 320             --height 240             --num_images $count             --output_image_dir "$out_folder/images/"             --output_scene_dir "$out_folder/scenes/"             --output_scene_file "$out_folder/CLEVR_scenes.json"             --start_idx $current_start             --use_gpu 1             --check_visibility ${CHECK_VISIBILITY:-0}             --mode "$mode"             --min_dist 0.1             --margin 0.1             --max_objects 7             --min_objects 4             >> "$LOG_FILE" 2>&1 &
            
        pids+=($!)
        current_start=$((current_start + count))
    fi
done

# Start Monitor
python3 image_generation/monitor.py --total $NUM_IMAGES --log_file "$LOG_FILE" &
monitor_pid=$!

# Trap to kill processes on exit/interrupt
cleanup() {
    echo "Stopping all processes..."
    kill ${pids[@]} $monitor_pid 2>/dev/null
    exit
}
trap cleanup SIGINT SIGTERM

# Wait for all render processes to finish
wait ${pids[@]}

# Kill monitor after renders are done
kill $monitor_pid 2>/dev/null

echo "Done."

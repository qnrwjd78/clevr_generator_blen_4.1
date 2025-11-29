import sys
import time
import argparse
import os
import subprocess
from tqdm import tqdm

def get_gpu_memory_usage():
    """Returns a string describing GPU memory usage."""
    try:
        # Query memory used and total for all GPUs
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        lines = result.strip().split('\n')
        if not lines:
            return "N/A"
        
        # If single GPU
        if len(lines) == 1:
            used, total = lines[0].split(',')
            return f"{used.strip()}MiB / {total.strip()}MiB"
        
        # If multiple GPUs, show the first one (or customize as needed)
        used, total = lines[0].split(',')
        return f"GPU0: {used.strip()}MiB"
    except FileNotFoundError:
        return "No GPU"
    except Exception:
        return "Err"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total', type=int, required=True, help="Total number of images to generate")
    parser.add_argument('--log_file', type=str, required=True, help="Path to the log file to monitor")
    args = parser.parse_args()

    total_images = args.total
    log_file_path = args.log_file
    
    # Wait for log file to be created
    while not os.path.exists(log_file_path):
        time.sleep(0.1)

    # Print System GPU Info
    print("-" * 40)
    print("System GPU Information:")
    try:
        subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], check=True)
    except Exception:
        print("Could not retrieve GPU info (nvidia-smi not found?)")

    # Track processed images to avoid duplicates if multiple lines appear
    # (Though simple counting is usually enough for "Saved:" lines)
    saved_count = 0
    

    blender_device_info_printed = False

    with tqdm(total=total_images, unit='scene', dynamic_ncols=True) as pbar:
        last_gpu_check = 0
        
        with open(log_file_path, "r") as f:
            # Go to the end of file initially? No, we want to read from start 
            # because the script might have just started writing.
            # But if we restart, we might want to tail. 
            # Assuming clean start for each run.
            
            while saved_count < total_images:
                # Update GPU stats periodically (every 2 seconds)
                current_time = time.time()
                if current_time - last_gpu_check > 2.0:
                    mem_usage = get_gpu_memory_usage()
                    pbar.set_postfix(gpu_mem=mem_usage, refresh=False)
                    last_gpu_check = current_time
                    # If we are waiting for lines, force a refresh to show updated GPU stats
                    # But avoid flickering if lines are coming in fast
                
                line = f.readline()
                if not line:
                    # Force refresh to show GPU stats if idle
                    if time.time() - last_gpu_check < 0.1: # Just updated
                         pbar.refresh()
                    time.sleep(0.1)
                    continue
                
                # Check for Blender Device Info
                if not blender_device_info_printed and "Configured Cycles to use GPU with backend" in line:
                    tqdm.write(f"Blender Device Config: {line.strip()}")
                    blender_device_info_printed = True

                # Check for GPU info (optional, just to mimic old behavior)
                if "Enabled:" in line and "CPU" not in line:
                    try:
                        parts = line.split("Enabled:")
                        if len(parts) > 1:
                            gpu_name = parts[1].strip()
                            # Only print if we haven't seen it or just once
                            # tqdm.write(f"GPU detected: {gpu_name}") 
                            # (Disabled to avoid cluttering with multiple GPUs)
                            pass
                    except:
                        pass

                # Check for saved image
                # Blender script prints: Saved: '.../filename.png'
                # Only update progress on target image to count each scene once
                if "Saved:" in line and "target.png" in line:
                    saved_count += 1
                    pbar.update(1)

if __name__ == "__main__":
    main()

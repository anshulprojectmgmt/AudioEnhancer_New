import os
import shutil
import time

# Path to the temporary directory
TMP_DIR = "./Data/tmp"

# Max age in seconds (e.g., 7 days)
# 7 days * 24 hours/day * 60 minutes/hour * 60 seconds/minute
MAX_AGE_SECONDS = 1 * 24 * 60 * 60 

def cleanup():
    now = time.time()
    
    if not os.path.exists(TMP_DIR):
        print(f"Directory not found: {TMP_DIR}")
        return

    for uid_folder in os.listdir(TMP_DIR):
        folder_path = os.path.join(TMP_DIR, uid_folder)
        
        if not os.path.isdir(folder_path):
            continue
            
        try:
            # Get the time the folder was last modified
            folder_mod_time = os.path.getmtime(folder_path)
            
            if (now - folder_mod_time) > MAX_AGE_SECONDS:
                print(f"DELETING stale cache: {folder_path} (older than 7 days)")
                shutil.rmtree(folder_path)
            else:
                print(f"KEEPING active cache: {folder_path}")
                
        except Exception as e:
            print(f"Error processing {folder_path}: {e}")

if __name__ == "__main__":
    print("Starting stale cache cleanup...")
    cleanup()
    print("Cleanup complete.")

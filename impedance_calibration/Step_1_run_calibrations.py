import multiprocessing
import subprocess
from pathlib import Path
import sys
import signal
import tqdm
import threading
import time

# Constants
NUM_RUNS = 1  # Number of times to run each script
MAX_WORKERS = 10  # Maximum number of scripts to run concurrently

# List of scripts to run
scripts = list((Path.cwd() / 'calibration_scripts').glob('*.py'))

# # Exclude specific scripts
# exclude = ['calibrate0','calibrate1','calibrate2','calibrate3']
# scripts = [x for x in scripts if x.stem not in exclude]

# Global variables to track task status
completed_tasks = []
running_tasks = []
remaining_tasks = []

def run_script(args):
    script, run_num = args
    try:
        running_tasks.append(script.stem)  # Add to running tasks
        if script.stem in remaining_tasks:
            remaining_tasks.remove(script.stem)  # Remove from remaining tasks
        print(f"Running {script.stem} (Run {run_num + 1}/{NUM_RUNS})")
        subprocess.run([f"{sys.executable}", script], check=True)
        print(f"Completed {script.stem} (Run {run_num + 1}/{NUM_RUNS})")
        completed_tasks.append(script.stem)  # Add to completed tasks
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
    finally:
        if script.stem in remaining_tasks:
            running_tasks.remove(script.stem)  # Remove from running tasks when done

def init_worker():
    # Ignore the interrupt signals in the child workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def print_task_status():
    """Prints the status of completed, running, and remaining tasks."""
    print("\n----- Task Status -----")
    print(f"Completed tasks: {completed_tasks}")
    print(f"Currently running tasks: {running_tasks}")
    print(f"Remaining tasks: {remaining_tasks}")
    print("-----------------------\n")

def monitor_console_input():
    """Monitors console input for 's' key to print task status."""
    while True:
        user_input = input()  # Wait for console input
        if user_input.strip().lower() == 'status':
            print_task_status()

if __name__ == '__main__':
    print([x.stem for x in scripts])

    if not scripts:
        print("No scripts found. Exiting.")
        sys.exit(1)

    # Initialize global tracking lists
    remaining_tasks = [script.stem for script in scripts for _ in range(NUM_RUNS)]

    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(script, run_num) for script in scripts for run_num in range(NUM_RUNS)]

    print(f"Tasks created: {len(tasks)}")

    # Set up a signal handler for the main process
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Initialize the pool with worker processes
    pool = multiprocessing.Pool(processes=MAX_WORKERS, initializer=init_worker)

    # Restore the signal handler for the main process
    signal.signal(signal.SIGINT, original_sigint_handler)

    # Start a thread to monitor console input for 's' key press
    threading.Thread(target=monitor_console_input, daemon=True).start()

    try:
        # Start the pool of workers with a progress bar
        timeout_sec = 24 * 60 * 60  # 24 hours timeout
        pool.map_async(run_script, tasks).get(timeout_sec) 
        # with tqdm.tqdm(total=len(tasks), desc="Scripts Progress", unit="task") as pbar:
        #     for _ in pool.imap_unordered(run_script, tasks):
        #         pbar.update(1)

    except KeyboardInterrupt:
        print("\nCtrl + C detected. Terminating the pool...")
        pool.terminate()  # Terminate worker processes
        pool.join()       # Wait for processes to finish
        sys.exit(1)
    else:
        pool.close()  # Close the pool when done submitting tasks
        pool.join()   # Wait for all workers to finish

    print("Finished script execution.")
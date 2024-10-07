import multiprocessing
import subprocess
from pathlib import Path
import sys
import signal
import tqdm

#TODO would be nice if nfev and nit appeared every 30ish minutes or so
#TODO or just a status bar to now many more processes were running

# Constants
NUM_RUNS = 2  # Number of times to run each script
MAX_WORKERS = 8  # Maximum number of scripts to run concurrently

# List of scripts to run
scripts = list((Path.cwd() / 'calibration_scripts').glob('*.py'))

# Todo, have a more elagant way of doing this, i'm thinking just keep things seperated into different folders
# include = ['jaccard_buffer_mean','jaccard_buffer_total','jaccard_exact_mean','jaccard_exact_total']
exclude = ['calibrate0','calibrate1','calibrate2','calibrate3']
scripts = [x for x in scripts if x.stem not in exclude]

def run_script(args):
    script, run_num = args
    try:
        print(f"Running {script.stem} (Run {run_num + 1}/{NUM_RUNS})")
        subprocess.run([f"{sys.executable}", script], check=True)
        print(f"Completed {script.stem} (Run {run_num + 1}/{NUM_RUNS})")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")

def init_worker():
    # Ignore the interrupt signals in the child workers
    signal.signal(signal.SIGINT, signal.SIG_IGN)

if __name__ == '__main__':
    print([x.stem for x in scripts])

    if not scripts:
        print("No scripts found. Exiting.")
        sys.exit(1)

    # Create a list of (script, run_num) pairs for NUM_RUNS
    tasks = [(script, run_num) for script in scripts for run_num in range(NUM_RUNS)]

    print(f"Tasks created: {len(tasks)}")

    # Set up a signal handler for the main process
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Initialize the pool with worker processes
    pool = multiprocessing.Pool(processes=MAX_WORKERS, initializer=init_worker)

    # Restore the signal handler for the main process
    signal.signal(signal.SIGINT, original_sigint_handler)

    try:
        # Start the pool of workers
        timeout_sec = 24 * 60 * 60
        pool.map_async(run_script, tasks).get(timeout_sec) 
        # pool.map_async(run_script, tasks).get(9999999)  # A large timeout to allow for interrupt handling

    except KeyboardInterrupt:
        print("\nCtrl + C detected. Terminating the pool...")
        pool.terminate()  # Terminate worker processes
        pool.join()       # Wait for processes to finish
        sys.exit(1)
    else:
        pool.close()  # Close the pool when done submitting tasks
        pool.join()   # Wait for all workers to finish

    print("Finished script execution.")

    # TODO run the post processing code after that

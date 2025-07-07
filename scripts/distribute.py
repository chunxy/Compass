import os
import signal
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from fabric import Connection

# Directory on remote machines where experiments will run and produce output.
REMOTE_WORKSPACE = "~/repos/Compass"
PID_FILE = "/tmp/run.pid"

# Specify your SSH key file path here
SSH_KEY_FILE = os.path.expanduser("~/.ssh/id_rsa")

# Keep track of active connections to clean up on exit
active_connections = []


def cleanup_connections(signum=None, frame=None):
  """Clean up remote processes when the script is terminated."""
  print("\nCleaning up remote processes...")
  for conn in active_connections:
    try:
      conn.run(f"kill $(ps -s $(cat {PID_FILE}) -o pid=)", warn=True, hide=True)
      print(f"Cleaned up processes on {conn.host}")
    except Exception as e:
      print(f"Error cleaning up {conn.host}: {e}")
  sys.exit(0)


# Register signal handlers
signal.signal(signal.SIGINT, cleanup_connections)
signal.signal(signal.SIGTERM, cleanup_connections)

# ==============================================================================
# 1. CONFIGURATION - Modify this section for your experiments
# ==============================================================================

# List of remote machines to use.
# Fabric will use your ~/.ssh/config for usernames and keys if they exist.
# Otherwise, specify them directly, e.g., 'user@hostname'
# or using connect_kwargs: Connection(host="...", connect_kwargs={"key_filename": "..."})
GROUPS = OrderedDict([
  ("group0", ["ssd30", "ssd31", "ssd33"]),
  ("group1", ["ssd27", "ssd28", "ssd29"]),
  ("group2", ["ssd24", "ssd25", "ssd26"]),
  ("group3", ["ssd18", "ssd19", "ssd20"]),
  ("group4", ["ssd15", "ssd16", "ssd17"]),
  ("group5", ["ssd12", "ssd13", "ssd14"]),
])

HOSTS = [host for group in GROUPS.values() for host in group]

# --- Define the jobs you want to run ---
# This is a list of command strings. You can create as many as you need.
# The script will distribute these jobs across the available hosts.
PRE_EXP_SCRIPT = \
"""
bash pull_and_build.sh || exit 1
rm -rf logs_10/*
"""

POST_EXP_SCRIPT = \
"""
cp -r logs_10/* /opt/nfs_dcc/chunxy/logs_10
"""

ONED_EXPS = [
  "1d-k-exp",
  "1d-k-cg-exp",
  "1d-bikmeans-exp",
  "1d-bikmeans-cg-exp",
  "1d-pca-exp",
  "1d-pca-cg-exp",
]

TWOD_EXPS = [
  "2d-k-exp",
  "2d-k-cg-exp",
  "2d-bikmeans-exp",
  "2d-bikmeans-cg-exp",
  "2d-pca-exp",
  "2d-pca-cg-exp",
]

THREED_EXPS = [
  "3d-k-exp",
  "3d-k-cg-exp",
  "3d-bikmeans-exp",
  "3d-bikmeans-cg-exp",
  "3d-pca-exp",
  "3d-pca-cg-exp",
]

FOURD_EXPS = [
  "4d-k-exp",
  "4d-k-cg-exp",
  "4d-bikmeans-exp",
  "4d-bikmeans-cg-exp",
  "4d-pca-exp",
  "4d-pca-cg-exp",
]

EXP_GROUP_1 = [
  "1d-k-exp",
  "1d-k-cg-exp",
  "1d-bikmeans-exp",
  "1d-bikmeans-cg-exp",
  "2d-k-exp",
  "2d-k-cg-exp",
]

EXP_GROUP_2 = [
  "2d-bikmeans-exp",
  "2d-bikmeans-cg-exp",
  "3d-k-exp",
  "3d-k-cg-exp",
  "3d-bikmeans-exp",
  "3d-bikmeans-cg-exp",
]

EXP_GROUP_PCA = [
  "1d-pca-exp",
  "1d-pca-cg-exp",
  "2d-pca-exp",
  "2d-pca-cg-exp",
  "3d-pca-exp",
  "3d-pca-cg-exp",
]

EXP_GROUP_ICG = [
  "1d-k-icg-exp",
  "1d-bikmeans-icg-exp",
  "1d-pca-icg-exp",
  "2d-k-icg-exp",
  "2d-bikmeans-icg-exp",
  "2d-pca-icg-exp",
]

# ==============================================================================
# 2. REMOTE EXECUTION FUNCTION - This function runs on each remote machine
# ==============================================================================


def run_remote_job(args):
  """
    Connects to a remote host, runs a single job command, and returns the result.

    Args:
        args (tuple): A tuple containing (host_string, job_command, ssh_key_file).
                     ssh_key_file is optional and defaults to None.

    Returns:
        dict: A dictionary containing the execution result.
    """
  host, command, ssh_key_file = args
  result_summary = {'host': host, 'command': command, 'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1}

  try:
    print(f"CONNECTING: {host}")
    # Create connection with optional SSH key file
    connect_kwargs = {}
    if ssh_key_file:
      connect_kwargs['key_filename'] = ssh_key_file

    proxy_env = {
      'http_proxy': 'http://proxy.cse.cuhk.edu.hk:8000',
      'https_proxy': 'http://proxy.cse.cuhk.edu.hk:8000',
    }

    with Connection(host, connect_kwargs=connect_kwargs) as conn:
      # Add connection to active connections list
      active_connections.append(conn)

      # Set working directory
      with conn.cd(REMOTE_WORKSPACE):
        print(f"RUNNING on {host}: {command}")

        # Modify command to store its own PID
        cap_pid_run_cmd = f"echo $$ > {PID_FILE} && {command}"
        result = conn.run(cap_pid_run_cmd, warn=True, hide=False, env=proxy_env)
        result_summary.update({
          'success': result.ok,
          'stdout': result.stdout.strip(),
          'stderr': result.stderr.strip(),
          'exit_code': result.return_code,
          'ftime': datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        })

        if result.ok:
          print(f"SUCCESS: Job on {host} finished with exit code {result.return_code}.")
          # Optional: Download results here if needed
          # e.g., conn.get(f"{REMOTE_EXPERIMENT_DIR}/output.csv", f"{LOCAL_RESULTS_DIR}/{host}_output.csv")
        else:
          print(f"FAILED: Job on {host} failed with exit code {result.return_code}.")

  except Exception as e:
    print(f"ERROR: Could not connect or run job on {host}. Details: {e}")
    result_summary['stderr'] = str(e)
  finally:
    # Remove connection from active connections list
    if 'conn' in locals():
      active_connections.remove(conn)

  return result_summary


# ==============================================================================
# 3. LOCAL POST-PROCESSING FUNCTION - Runs on your machine after experiments
# ==============================================================================


def post_process():
  """
    This function is called after all remote jobs are complete.
    Replace this with your actual post-processing logic, like generating plots.
    """
  print("\n----------------------------------------------------")
  print("🎨 Starting local post-processing (e.g., drawing figures)...")
  print("----------------------------------------------------\n")

  # This is the line you correctly identified as needing more context.
  # It calls a local Python script, passing the results directory to it.
  # You would create 'draw_figures.py' to handle the plotting logic.
  try:
    os.chdir(os.path.expanduser("~/repos/Compass/"))

    summarize = "scripts/summarize.py"
    if os.path.exists(summarize):
      subprocess.run(["python", summarize], check=True)
      print(f"\n✅ Post-processing script '{summarize}' executed successfully.")
    else:
      print(f"NOTE: Post-processing script '{summarize}' not found. Skipping.")

    cherrypick = "scripts/cherrypick.py"
    if os.path.exists(cherrypick):
      subprocess.run(["python", cherrypick], check=True)
      print(f"\n✅ Post-processing script '{cherrypick}' executed successfully.")
    else:
      print(f"NOTE: Post-processing script '{cherrypick}' not found. Skipping.")

    notify = "scripts/notify.py"
    if os.path.exists(notify):
      subprocess.run(["python", notify], check=True)
      print(f"\n✅ Notification script '{notify}' executed successfully.")
    else:
      print(f"NOTE: Notification script '{notify}' not found. Skipping.")

  except subprocess.CalledProcessError as e:
    print(f"ERROR: The post-processing script failed with exit code {e.returncode}.")


# ==============================================================================
# 4. MAIN ORCHESTRATOR
# ==============================================================================


def run_grouped_exp(exp_set):
  os.chdir(os.path.expanduser("~/repos/Compass/"))

  # Generate experiment scripts.
  compose = "scripts/compose.py"
  if os.path.exists(compose):
    subprocess.run(["python", compose], check=True)
    print(f"\n✅ Compose script '{compose}' executed successfully.")
  else:
    print(f"NOTE: Compose script '{compose}' not found. Skipping.")

  # Create a list of (host, job) tuples for distribution.
  # This simple round-robin logic assigns jobs to hosts cyclically.
  tasks_to_run = []
  for group, exp in zip(GROUPS.keys(), exp_set):
    for i, host in enumerate(GROUPS[group]):
      exp_script = f"bash runs/exp/{exp}-{i + 1}.sh"
      task = PRE_EXP_SCRIPT + exp_script + POST_EXP_SCRIPT
      # Add the SSH key file as the third argument
      tasks_to_run.append((host, task, SSH_KEY_FILE))

  all_results = []
  print(f"Starting experiments. Distributing {len(exp_set)} jobs to {len(GROUPS)} groups.")

  # Use ThreadPoolExecutor to run jobs on all hosts in parallel.
  # The number of workers determines how many SSH connections are active at once.
  with ThreadPoolExecutor(max_workers=len(tasks_to_run)) as executor:
    # executor.map applies the function to each item in the tasks_to_run list
    all_results = list(executor.map(run_remote_job, tasks_to_run))

  print("\n====================================================")
  print("📋 EXPERIMENT SUMMARY")
  print("====================================================")

  run_log = open(f"scratches/run-log-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.txt", "w")

  successful_jobs = 0
  for res in all_results:
    status_word = "✅ SUCCESS" if res['success'] else "❌ FAILED"
    run_log.write(f"- Host: {res['host']}\n")
    run_log.write(f"  Command: {res['command']}\n")
    run_log.write(f"  Status: {status_word}\n")
    run_log.write(f"  Finish: {res['ftime']}\n")
    if not res['success']:
      run_log.write(f"  Stderr: {res['stderr']}\n")
    successful_jobs += 1 if res['success'] else 0

  run_log.write(f"\nTotal Jobs: {len(tasks_to_run)}, Successful: {successful_jobs}, Failed: {len(tasks_to_run) - successful_jobs}\n")
  # --- Conditionally run post-processing ---
  if successful_jobs == len(tasks_to_run):
    # post_process()
    print("\nGood job!", file=run_log)
  else:
    print("\nSkipping post-processing due to failed jobs.", file=run_log)
  run_log.close()


if __name__ == '__main__':
  # run_grouped_exp(ONED_EXPS)
  # run_grouped_exp(TWOD_EXPS)
  # run_grouped_exp(THREED_EXPS)
  # run_grouped_exp(FOURD_EXPS)

  run_grouped_exp(EXP_GROUP_1)
  run_grouped_exp(EXP_GROUP_2)
  run_grouped_exp(EXP_GROUP_PCA)
  run_grouped_exp(EXP_GROUP_ICG)
  post_process()

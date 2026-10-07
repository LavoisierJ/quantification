"""
Measuring the time observation of data recorded with at least 50 DUs. 
The script searches for files in the specified directory, filters them based on the number of DUs, 
and calculates the observation time for each run. 
The results are saved as numpy arrays for further analysis.
"""


import glob
import numpy as np
import re
from datetime import datetime
from os.path import basename

def extract_and_measure_observation_time(directory):
    # Step 1: Find all files matching the patterns (full paths)
    files = glob.glob(f"{directory}/*CD*-*dus-*.*") + glob.glob(f"{directory}/*CD*-*DUs-*.*")

    # Step 2: Filter files where the number in ** is > 50
    filtered_files = []
    for file in files:
        # Extract the basename (filename only) for processing
        filename = basename(file)
        parts = filename.split('-')
        for part in parts:
            if 'dus' in part.lower():
                numbers = re.findall(r'\d+', part)
                if numbers:
                    num = int(numbers[0])
                    if num > 50:
                        filtered_files.append(file)  # Store the full path
                        break  # Move to next file

    # Step 3: Group files by run and calculate observation time
    runs = {}
    for file in filtered_files:
        filename = basename(file)
        # Extract run identifier (e.g., _RUN001_)
        run_id = None
        for part in filename.split('_'):
            if part.startswith('RUN'):
                run_id = part
                break
        if not run_id:
            continue

        # Extract timestamp (e.g., _20251129_085215_)
        timestamp_str = filename.split('_')[1:3]
        if len(timestamp_str) != 2:
            continue
        date_str, time_str = timestamp_str
        try:
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
        except ValueError:
            continue

        # Add to runs dictionary
        if run_id not in runs:
            runs[run_id] = {'start': timestamp, 'end': timestamp}
        else:
            if timestamp < runs[run_id]['start']:
                runs[run_id]['start'] = timestamp
            if timestamp > runs[run_id]['end']:
                runs[run_id]['end'] = timestamp

    # Step 4: Calculate observation time for each run
    observation_times = {}
    for run_id, times in runs.items():
        observation_time = times['end'] - times['start']
        observation_times[run_id] = observation_time.total_seconds()  # in seconds

    return filtered_files, observation_times



months = np.array(['07', '08', '09', '10', 
                #    '11', '12', '01', '02', '03', '04', '06'
                   ])
years = np.array(['2025', '2025', '2025', '2025', 
                #   '2025', '2025', '2026', '2026', '2026', '2026', '2026'
                  ])

filtered_files = np.array([], dtype='str')
observation_times = {}

print(f"Number of files in each month:")
for i in range(len(months)):
    directory = f"/sps/grand/data/gp80/GrandRoot/{years[i]}/{months[i]}"
    files, obs_times = extract_and_measure_observation_time(directory)
    filtered_files = np.append(filtered_files, files)
    observation_times.update(obs_times)
    print(f"  {months[i]}/{years[i]}: {len(files)} files")
    print(f"    Observation time: {sum(obs_times.values()) / 86400:.2f} days")

print(f"Total observation time across all runs: {sum(observation_times.values())} seconds")
print(f"Observation time in days: {sum(observation_times.values()) / 86400} days")
print(f"Total number of filtered files: {len(filtered_files)}")


with open("/pbs/home/j/jlavoisier/pipeline_lab/check_data/filtered_files.txt", "w") as f:
    for filename in filtered_files:
        f.write(f"{filename}\n")

# np.save(f"/pbs/home/j/jlavoisier/pipeline_lab/check_data/filtered_files.npy", filtered_files)
np.save(f"/pbs/home/j/jlavoisier/pipeline_lab/check_data/observation_times.npy", observation_times)

# test = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/check_data/filtered_files.npy", allow_pickle=True)
# test = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/check_data/observation_times.npy", allow_pickle=True).item()

# print(test)
import numpy as np
import grand.dataio.data_handling as dh

files = sorted(np.loadtxt(f"/sps/grand/jlavoisier/code/quantification/check_data/filtered_files.txt", 
                   dtype=str
                   ))

nb_events = 0
nb_events_per_month = {'06': 0, '07': 0, '08': 0, '09': 0}

for filename in files:
    month = filename.split('/')[-2]
    file_root = dh.DataFile(str(filename))

    nb_events_per_month[month] += file_root.tadc.get_entries()

    nb_events += file_root.tadc.get_entries()

    file_root.close()

    print(f"File: {filename.split('/')[-1]}, Number of events: {file_root.tadc.get_entries()}")

print(f"Total number of events: {nb_events}")
print(f"Number of events per month: {nb_events_per_month}")
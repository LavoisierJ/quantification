"""
Extract the polarization of the events in different runs.

"""

import os
import subprocess
import numpy as np
from glob import glob

path = "/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts"

polar_parasites = {}

RUNs = np.load("/sps/grand/jlavoisier/code/quantification/check_data/observation_times.npy",
                allow_pickle=True
                ).item().keys()


for run in RUNs:
    repert_run = sorted(glob(f"{path}/*{run}*"))
    # Initialize empty arrays to store the data for the current run
    polar_parasites_run = np.array([])

    for i in range(len(repert_run)):
        try:
            data = np.load(f"{repert_run[i]}/polar_list.npy")
        except:
            fname = repert_run[i].split("/")[-1]
            month = fname.split("_")[1][4:6]
            subprocess.run(["python", "/sps/grand/jlavoisier/code/quantification/polarization_cut/1_parasite_polarization.py", f"/sps/grand/data/gp80/GrandRoot/2025/{month}/{fname}"])
            data = np.load(f"{repert_run[i]}/polar_list.npy")


        # Concatenate the data for the current run
        polar_parasites_run = np.concatenate((polar_parasites_run, data[0]))

	# Store the data for the current run in the dictionaries
    
    polar_parasites[run] = polar_parasites_run

# Save the data in .npy files
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/polar_parasites.npy",
        polar_parasites
        )
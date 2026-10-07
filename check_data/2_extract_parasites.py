"""

Gather the time of detection and the reconstructed zenith and azimuth of the events in different runs.

"""

import numpy as np
from glob import glob

path = "/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts"

time_parasites = {}
theta_parasites = {}
phi_parasites = {}
polar_parasites = {}
chi2_PWF_parasites = {}
chi2_SWF_parasites = {}
chi2_ADF_parasites = {}

RUNs = np.load("/sps/grand/jlavoisier/code/quantification/check_data/observation_times.npy",
                allow_pickle=True
                ).item().keys()


for run in RUNs:
    repert_run = sorted(glob(f"{path}/*{run}*"))
    # Initialize empty arrays to store the data for the current run
    time_parasites_run = np.array([])
    theta_parasites_run = np.array([])
    phi_parasites_run = np.array([])
    polar_parasites_run = np.array([])
    chi2_PWF_parasites_run = np.array([])
    chi2_SWF_parasites_run = np.array([])
    chi2_ADF_parasites_run = np.array([])

    for i in range(len(repert_run)):
        data = np.load(f"{repert_run[i]}/chi2_list.npy")
        # Concatenate the data for the current run
        time_parasites_run = np.concatenate((time_parasites_run, data[0]))
        theta_parasites_run = np.concatenate((theta_parasites_run, data[1]))
        phi_parasites_run = np.concatenate((phi_parasites_run, data[2]))
        polar_parasites_run = np.concatenate((polar_parasites_run, data[3]))
        chi2_PWF_parasites_run = np.concatenate((chi2_PWF_parasites_run, data[4]))
        chi2_SWF_parasites_run = np.concatenate((chi2_SWF_parasites_run, data[5]))
        chi2_ADF_parasites_run = np.concatenate((chi2_ADF_parasites_run, data[6]))

	# Store the data for the current run in the dictionaries
    
    time_parasites[run] = time_parasites_run
    theta_parasites[run] = theta_parasites_run
    phi_parasites[run] = phi_parasites_run
    polar_parasites[run] = polar_parasites_run
    chi2_PWF_parasites[run] = chi2_PWF_parasites_run
    chi2_SWF_parasites[run] = chi2_SWF_parasites_run
    chi2_ADF_parasites[run] = chi2_ADF_parasites_run

# Save the data in .npy files
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/time_parasites.npy",
        time_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/theta_parasites.npy",
        theta_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/phi_parasites.npy",
        phi_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/polar_parasites.npy",
        polar_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/chi2_PWF_parasites.npy",
        chi2_PWF_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/chi2_SWF_parasites.npy",
        chi2_SWF_parasites
        )
np.save("/sps/grand/jlavoisier/code/quantification/check_data/data/chi2_ADF_parasites.npy",
        chi2_ADF_parasites
        )

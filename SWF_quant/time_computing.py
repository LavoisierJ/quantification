import numpy as np
from glob import glob

repert = sorted(glob("/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts/*/time_list.npy"))


times = np.empty((3,0))
for i in repert:
    # print(i)
    data = np.load(f"{i}", allow_pickle=True)

    times = np.hstack((times, data))

times_PWF = times[0,:]

times_SWF = times[1,:]
times_ADF = times[2,:]

print(f"Mean time for PWF: {np.mean(times_PWF)}")
print(f"Std time for PWF: {np.std(times_PWF)}")

print(f"Mean time for SWF: {np.mean(times_SWF)}")
print(f"Std time for SWF: {np.std(times_SWF)}")

print(f"Mean time for ADF: {np.mean(times_ADF)}")
print(f"Std time for ADF: {np.std(times_ADF)}")
import numpy as np
from glob import glob


t_polar = np.array([])

repert = sorted(glob("/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts/GP*"))


times = np.empty((3,0))
for i in repert:
    # print(i)
    data = np.load(f"{i}/time_list.npy", allow_pickle=True)
#     print(data)
    try:
        polar = np.load(f"{i}/time_polar_list.npy", allow_pickle=True)
        t_polar = np.append(t_polar, polar)
    except:
        pass

    times = np.hstack((times, data))

times_PWF = times[0,:]
times_SWF = times[1,:]
times_ADF = times[2,:]


np.save(f"/sps/grand/jlavoisier/code/quantification/combine/CPU_time/time_PWF.npy", 
        times_PWF
        )
np.save(f"/sps/grand/jlavoisier/code/quantification/combine/CPU_time/time_SWF.npy", 
        times_SWF
        )
np.save(f"/sps/grand/jlavoisier/code/quantification/combine/CPU_time/time_ADF.npy", 
        times_ADF
        )
np.save(f"/sps/grand/jlavoisier/code/quantification/combine/CPU_time/time_polar.npy", 
        t_polar
        )
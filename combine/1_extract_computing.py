import numpy as np
from glob import glob


path = "/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts"

t_PWF = np.array([])
t_SWF = np.array([])
t_ADF = np.array([])
t_polar = np.array([])

repert_file = sorted(glob())
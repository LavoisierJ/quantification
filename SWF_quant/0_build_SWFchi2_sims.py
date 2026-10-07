"""
Quantifying the zenith angle dependence of the number of antennas triggered in the simulations. 
This is done by counting the number of antennas that are triggered for each zenith angle bin and plotting the results.
"""

import grand.dataio.data_handling as dh
import grand.analysis.fitting as fit
import grand.analysis.constants as cons
import numpy as np
import sys
from glob import glob
from scipy.signal import hilbert


array_dus = np.int32(np.loadtxt("/pbs/home/j/jlavoisier/pipeline_lab/check_sims/gp65_correspondance_ID_febID.txt").T) # so that it starts at 0 (ie use it for simulations)
array_dus[0] = array_dus[0] - 65000 - 1


data_dict = {103: [20, 30, 20, 30],
109: [40, 52, 46, 58],
1010: [34, 44, 53, 66],
1011: [34, 44, 39, 50],
1012: [35, 46, 46, 58],
1013: [39, 51, 42, 53],
1014: [35, 55, 35, 55],
1016: [31, 41, 40, 50],
1017: [35, 45, 40, 50],
1018: [46, 57, 43, 55],
1019: [55, 67, 42, 54],
1020: [41, 53, 43, 55],
1022: [48, 61, 43, 54],
1023: [48, 61, 43, 54],
1024: [39, 51, 45, 57],
1029: [13, 18, 31, 39],
1030: [36, 48, 47, 59],
1032: [36, 48, 47, 59],
1033: [44, 56, 43, 55],
1034: [40, 52, 46, 58],
1035: [37, 49, 51, 64],
1037: [35, 46, 51, 63],
1038: [35, 46, 51, 63],
1039: [35, 46, 51, 63],
1040: [40, 52, 42, 54],
1041: [35, 46, 51, 63],
1042: [40, 60, 60, 80],
1043: [39, 51, 43, 54],
1044: [35, 46, 51, 63],
1045: [33, 43, 45, 57],
1046: [36, 48, 41, 53],
1047: [41, 53, 40, 50],
1048: [29, 38, 48, 60],
1049: [35, 46, 51, 63],
1051: [33, 43, 41, 52],
1052: [39, 51, 46, 58],
1054: [42, 54, 40, 51],
1055: [44, 57, 48, 60],
1056: [39, 51, 46, 58],
1058: [39, 51, 44, 55],
1059: [35, 46, 51, 63],
1065: [44, 56, 38, 50],
1066: [38, 49, 43, 54],
1071: [26, 34, 44, 55],
1072: [44, 56, 46, 58],
1073: [38, 50, 44, 56],
1074: [44, 56, 51, 63],
1075: [37, 49, 49, 61],
1076: [35, 46, 47, 59],
1077: [30, 40, 43, 55],
1078: [28, 37, 43, 54],
1081: [40, 52, 42, 53],
1082: [40, 52, 42, 53],
1083: [38, 49, 51, 64],
1084: [40, 52, 53, 66],
1085: [36, 47, 38, 49],
1086: [24, 32, 44, 55],
1088: [35, 45, 75, 91],
1089: [35, 47, 45, 57],
1090: [46, 59, 48, 60],
1091: [37, 47, 55, 67],
1092: [48, 62, 62, 78],
1093: [48, 62, 62, 78],
1094: [42, 54, 45, 57]} # T2_X, T1_X, T2_Y, T1_Y

def extract_trigger_parameters(trace, trigger_config, baseline=0):
    # Extract the trigger infos from a trace

    # Parameters :
    # ------------
    # trace, numpy.ndarray: 
    # traces in ADC unit
    # trigger_config, dict:
    # the trigger parameters set in DAQ

    # Returns :
    # ---------
    # Index in the trace when the first T1 crossing happens
    # Indices in the trace of T2 crossing happens
    # Number of T2 crossings
    # Q, Peak/NC

    # Find the position of the first T1 crossing
    index_t1_crossing = np.where((trace) > trigger_config["th1"],
                                 np.arange(len(trace)), -1)
    dict_trigger_infos = dict()
    mask_T1_crossing = (index_t1_crossing != -1)
    if sum(mask_T1_crossing) == 0:
        # No T1 crossing 
        raise ValueError("No T1 crossing!")
    dict_trigger_infos['index_T1_crossing'] = None
    # Tquiet to decide the quiet time before the T1 crossing 
    for i in index_t1_crossing[mask_T1_crossing]:
       # Abs value not exceeds the T1 threshold
        if i - trigger_config["t_quiet"]//2 < 0:
            raise ValueError("Not enough data before T1 crossing!")
        if np.all((trace[np.max([0, i - trigger_config['t_quiet'] // 2]):i]) <= trigger_config["th1"]):
            dict_trigger_infos["index_T1_crossing"] = i
            # the first T1 crossing satisfying the quiet condition
            break
    if dict_trigger_infos['index_T1_crossing'] == None:
        raise ValueError("No T1 crossing with Tquiet satified!")
    # The trigger logic works for the timewindow given by T_period after T1 crossing.
    # Count number of T2 crossings, relevant pars: T2, NCmin, NCmax, T_sepmax
    # From ns to index, divided by two for 500MHz sampling rate
    
    period_after_T1_crossing = trace[dict_trigger_infos["index_T1_crossing"]:dict_trigger_infos["index_T1_crossing"]+trigger_config['t_period']//2]
    # All the points above +T2
    positive_T2_crossing = (np.array(period_after_T1_crossing) > trigger_config['th2']).astype(int)
    # Positive crossing, the point before which is below T2.
    mask_T2_crossing_positive = np.diff(positive_T2_crossing) == 1

    # Register the first T1 crossing as a T2 crossing
    mask_first_T1_crossing = np.zeros(len(period_after_T1_crossing), dtype=bool)
    mask_first_T1_crossing[0] = True

    mask_first_T1_crossing[1:] = (mask_T2_crossing_positive)
    index_T2_crossing = np.arange(len(period_after_T1_crossing))[mask_first_T1_crossing]
    n_T2_crossing = 1 # Starting from the first T1 crossing.
    dict_trigger_infos["index_T2_crossing"] = [0]
    if len(index_T2_crossing) > 1:
        for i, j in zip(index_T2_crossing[:-1], index_T2_crossing[1:]):
            # The separation between successive T2 crossings
            time_separation = (j - i) * 2
            if time_separation < trigger_config["t_sepmax"]:
                n_T2_crossing += 1
                dict_trigger_infos["index_T2_crossing"].append(j)
            else:
                # Violate the maximum separation, fail to trigger
                raise ValueError(f"Violating Tsepmax, the separation is {time_separation} ns.")
    else:
        n_T2_crossing = 1
        j = 1
    # Change the reference of indices of T2 crossing
    dict_trigger_infos["index_T2_crossing"] = np.array(dict_trigger_infos["index_T2_crossing"]) + dict_trigger_infos["index_T1_crossing"]
    dict_trigger_infos["NC"] = n_T2_crossing
    
    # Calulate the peak value
    dict_trigger_infos["Q"] = (np.max(np.abs(period_after_T1_crossing[:j])) - baseline) / dict_trigger_infos["NC"]
    return dict_trigger_infos

dict_trigger_parameter = dict([
  ("t_quiet", 512),
  ("t_period", 512),
  # ("t_sepmax", 20),
  ("t_sepmax", 50),
  ("nc_min", 2),
  ("nc_max", 7),
  ("q_min", 0),
  ("q_max", 255),
  # ("th1", 100),
  # ("th2", 50),
  ("th1",70),
  ("th2", 60),
  # Configs of readout timewindow
  ("t_pretrig", 960),
  ("t_overlap", 64),
  ("t_posttrig", 1024)
  ])

def pass_T1(list_traces,
            du_id,
            correspondance=array_dus,
            ):
    """
    Inputs
        list_traces: list of list, containing the traces of the 3 channels for an antenna (X,Y,Z), shape(3,1024)
        du_id: ID of the DU (Detector Unit)
    Output
        Boolean indicating whether the signal passes the trigger T1 (trigger tested over channels X and Y)
    """
    GP65_du = correspondance[1,np.isin(correspondance[0], du_id)][0]

    for v in range(2) :
        dict_trigger_parameter["th1"] = data_dict[GP65_du][v*2+1]
        dict_trigger_parameter["th2"] = data_dict[GP65_du][v*2]
        trace = list_traces[v]
        try:
            trigger_infos = extract_trigger_parameters(trace, dict_trigger_parameter)
            if trigger_infos["NC"] >= dict_trigger_parameter["nc_min"] and trigger_infos["NC"] <= dict_trigger_parameter["nc_max"]:
                # triggered_ant.append(i)
                indicator = True
                break
            else:
                indicator = False
        except ValueError as e:
            # No T1 crossing, no trigger
            # print(k, ": No trigger.")
            indicator = False
            pass
    return(indicator)


def pass_threshold(list_traces,
                   threshold=60,
                   ):
    """
    Inputs
        list_traces: list of list, containing the traces of the 3 channels for an antenna (X,Y,Z), shape(3,1024)
        threshold: threshold for the peak value of the signal
    Output
        Boolean indicating whether the signal passes the threshold (tested over channels X and Y)
    """
    # indicator = False

    # trace = np.sqrt(list_traces[0]**2 + list_traces[1]**2) # EW component of the signal
    # if np.max(np.abs(trace)) > threshold:
    #     indicator = True

    traces = np.asarray(list_traces)

    traces_xy = traces[:2, :]

    # Enveloppes de Hilbert : forme (2, n_samples)
    envelopes_xy = np.abs(hilbert(traces_xy, axis=-1))

    envelope_x = envelopes_xy[0]
    envelope_y = envelopes_xy[1]

    envelope_xy = np.sqrt((envelope_x**2 + envelope_y**2) / 2.0)
    max_xy = np.max(envelope_xy, axis=-1)

    indicator = max_xy > threshold
    return(indicator)


def select_GP65_antennas(tree_adc,
                         event_id,
                         correspondance,
                         ) :
    """
    Entries:
        tree_adc: the adc tree cintaining the list of du_id
        event_id: index of event to look through
        corresponce: 2d-array with correspondance between GP65 antennas and Coreas simulated antennas
    Output:
        list of indexes corresponding to GP65 antennas
    """
    tree_adc.get_entry(event_id)
    list_du_id = tree_adc.du_id

    list_valid_indexes = np.array([])
    for i in range(len(list_du_id)):
        if np.isin(list_du_id[i], correspondance[0]):
            list_valid_indexes = np.append(list_valid_indexes, i)
    
    return(list_valid_indexes)


# repert_sims = sorted(glob('/sps/grand/DC2.1rc4/GP300ZHAireS-AN/*/adc_*_L1_0000.root'))#[:100]
# repert_sims_NJ = sorted(glob('/sps/grand/DC2.1rc4/GP300ZHAireS-NJ/*/adc_*_L1_0000.root'))#[:100]

# repert_sims = sorted(glob('/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/*/adc_*_L1_0000.root'))[:1]

# repert_sims = sorted(glob('/sps/grand/DC2.1rc4/GP300ZHAireS-AN/sim_Xiaodushan_20221025_220000_RUN0_CD_GP300ZHAireS-AN_0012/adc_*_L1_0000.root'))

# repert_sims2 = sorted(glob('/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/*/adc_*_L1_0000.root'))#[:50]
# repert_sims = repert_sims + repert_sims2

# coord_DC2 = np.loadtxt('/sps/grand/jlavoisier/output/simu/DC2_antenna_coord.txt')

AN_file = sorted(glob(f"{sys.argv[1]}/adc_*_L1_0000.root"))
NJ_file = sorted(glob(f"{sys.argv[2]}/adc_*_L1_0000.root"))
shower_file = sorted(glob(f"{sys.argv[2]}/shower_*.root"))


chi2_swf = np.array([])
chi2_pwf = np.array([])
n_ant_trig = np.array([])
theta_list = np.array([])
phi_list = np.array([])

theta_true = np.array([])
phi_true = np.array([])

t = np.arange(0, 1024*2, 2)

for i in range(len(AN_file)):
    run_path = glob(AN_file[i].split("/adc_")[0] + "/run_*_L1_0000.root")[0]

    print("AN file : ", AN_file[i])
    print("NJ file : ", NJ_file[i])
    print("Run file : ", run_path)

    file_sims = dh.DataFile(AN_file[i])
    file_NJ = dh.DataFile(NJ_file[i])
    file_run = dh.DataFile(run_path)
    file_shower = dh.DataFile(shower_file[i])
    
    adc = file_sims.tadc
    adc_NJ = file_NJ.tadc
    run = file_run.trun
    shower = file_shower.tshower
    run.get_run(0)
    n_event_i = adc.get_number_of_entries()

    for j in range(n_event_i) :
        # print("Event ", j+1, " / ", n_event_i)

        adc.get_entry(j)
        adc_NJ.get_entry(j)
        shower.get_entry(j)
        # print("Theta : ", shower.zenith, " -- Phi : ", shower.azimuth)

        GP65_indexes = select_GP65_antennas(adc_NJ, j, array_dus)
        # measure the EW component of the voltage for this event, for each antenna that detected it, and save it for the maximum SNR value among all antennas
        n_ant_i = len(GP65_indexes)
        list_traces = np.zeros((n_ant_i,3,1024))
        list_traces_NJ = np.zeros((n_ant_i,3,1024))

        valid_ant = np.array([], dtype=int) # list of antennas that were triggered (passed T1)

        time_nanoseconds = np.array([])
        time_seconds = np.array([])

        for k in range(n_ant_i) :
            list_traces[k,0] = adc.trace_ch[int(GP65_indexes[k])][0]
            list_traces[k,1] = adc.trace_ch[int(GP65_indexes[k])][1]
            list_traces[k,2] = adc.trace_ch[int(GP65_indexes[k])][2]

            list_traces_NJ[k,0] = adc_NJ.trace_ch[int(GP65_indexes[k])][0]
            list_traces_NJ[k,1] = adc_NJ.trace_ch[int(GP65_indexes[k])][1]
            list_traces_NJ[k,2] = adc_NJ.trace_ch[int(GP65_indexes[k])][2]

            du_id = adc.du_id[int(GP65_indexes[k])]

            # if pass_T1(list_traces[k], du_id, correspondance=array_dus) :
            if pass_threshold(list_traces_NJ[k], threshold=60) :
                valid_ant = np.append(valid_ant, int(GP65_indexes[k]))
                # measure the 
                Emodulus = np.sqrt(list_traces[k,0]**2+list_traces[k,1]**2+list_traces[k,2]**2)
                hilbert_amp = np.abs(hilbert(Emodulus))
                peak_time = t[np.int32(np.argmax(hilbert_amp))]
                time_seconds = np.append(time_seconds, adc.du_seconds[int(GP65_indexes[k])])
                time_nanoseconds = np.append(time_nanoseconds, adc.du_nanoseconds[int(GP65_indexes[k])] + peak_time)
        
        if len(valid_ant) < 5 : # We want to have at least 5 ant that were triggered
            continue
        
        theta_true = np.append(theta_true, shower.zenith)
        phi_true = np.append(phi_true, shower.azimuth)

        print("Event ", j+1, " / ", n_event_i, " -- Number of triggered antennas : ", len(valid_ant))

        time_nanoseconds = time_nanoseconds - time_nanoseconds[np.argmin(time_seconds)]
        time_seconds = time_seconds - time_seconds.min() # we set the minimum time to 0 for better precision in the next steps
        time_ant = time_seconds + time_nanoseconds*1e-9
        du_id = np.asarray(adc.du_id)[valid_ant]
        print("Triggered antennas DU id : ", du_id)


        du_indices = np.asarray(adc.get_dus_indices_in_run(run))[valid_ant]
        x_ants = np.asarray(run.du_xyz)[du_indices]


        theta, phi = fit.PWF_semianalytical(x_ants, time_ant)
        chi2_pwf_reduced = fit.PWF_loss((theta, phi), 
                                        x_ants, 
                                        time_ant, 
                                        c=cons.c_light, 
                                        n=cons.n_atm, 
                                        sigma=5e-9)/(len(valid_ant) - 2)
        print("PWF chi2 reduced: ", chi2_pwf_reduced)

        theta_swf_rad, phi_swf_rad, r_xmax, t_s = fit.recons_swf(theta, phi, time_ant, x_ants, sigma = 5e-9)
        chi2_swf_reduced = fit.SWF_loss(theta_swf_rad, 
                                        phi_swf_rad, 
                                        r_xmax, 
                                        t_s, 
                                        x_ants, 
                                        time_ant, 
                                        sigma=5e-9)/(len(valid_ant) - 4)
        print("SWF chi2 reduced: ", chi2_swf_reduced)

        theta_list = np.append(theta_list, 
                               theta
                               )
        phi_list = np.append(phi_list, 
                             phi
                             )
        chi2_pwf = np.append(chi2_pwf, 
                             chi2_pwf_reduced
                             )
        chi2_swf = np.append(chi2_swf, 
                             chi2_swf_reduced
                             )

        n_ant_trig = np.append(n_ant_trig, len(valid_ant))
    print("Number of events processed : ", len(n_ant_trig))



print("Number of events processed : ", len(chi2_pwf))
np.save(f"/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/ZHAiReS/{AN_file[0].split('/')[-2].split('-')[-1]}_chi2_SWF_n_ant.npy", np.array([n_ant_trig, chi2_swf, theta_list, phi_list, theta_true, phi_true]))
np.save(f"/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/ZHAiReS/{AN_file[0].split('/')[-2].split('-')[-1]}_chi2_PWF_n_ant.npy", np.array([n_ant_trig, chi2_pwf, theta_list, phi_list, theta_true, phi_true]))

# np.save("/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/simsAN_test.npy", np.array([n_ant_trig, chi2, theta_list, phi_list]))


# with open(f"/pbs/home/j/jlavoisier/pipeline_lab/SWF_quant/out_files/batch/{AN_file[0].split('/')[-1].split('.')[0]}_specs.txt", "w") as f :
#     f.write("Sim files used for the study : \n")
#     for file in repert_sims :
#         f.write(file + "\n")
#     f.write("\nNumber of events processed : " + str(len(n_ant_trig)) + "\n")
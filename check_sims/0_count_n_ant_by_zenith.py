"""
Quantifying the zenith angle dependence of the number of antennas triggered in the simulations. 
This is done by counting the number of antennas that are triggered for each zenith angle bin and plotting the results.
"""

import grand.dataio.data_handling as dh
import numpy as np
import sys
from glob import glob

from scipy.signal import hilbert

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

def pass_T1(list_traces):
    """
    Inputs
        list_traces: list of list, containing the traces of the 3 channels for an antenna (X,Y,Z), shape(3,1024)
    Output
        Boolean indicating whether the signal passes the trigger T1 (trigger tested over channels X and Y)
    """
    for v in range(2) :
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


array_dus = np.int32(np.loadtxt("/sps/grand/jlavoisier/code/quantification/check_sims/gp65_correspondance_ID_febID.txt").T) # so that it starts at 0 (ie use it for simulations)
array_dus[0] = array_dus[0] - 65000 - 1



# repert_sims = sorted(glob('/sps/grand/DC2.1rc4/GP289ZHAireS-NJ/*/adc_*_L1_0000.root'))
# repert_shower = sorted(glob('/sps/grand/DC2.1rc4/GP289ZHAireS-NJ/*/shower_*_L0_0000.root'))
# repert_showersim = sorted(glob('/sps/grand/DC2.1rc4/GP289ZHAireS-NJ/*/showersim_*_L0_0000.root'))


repert_sims = sorted(glob('/sps/grand/DC2_Coreas/Coreas_nonoise/*/adc_*_L1_0000.root'))
repert_shower = sorted(glob('/sps/grand/DC2_Coreas/Coreas_nonoise/*/shower_*_L0_0000.root'))
repert_showersim = sorted(glob('/sps/grand/DC2_Coreas/Coreas_nonoise/*/showersim_*_L0_0000.root'))

repert_AN = sorted(glob('/sps/grand/DC2_Coreas/RFChain_v2/COREAS-AN/*/adc_*_L1_0000.root'))


theta = np.array([])
phi = np.array([])
energy = np.array([])
n_ant_trig = np.array([])
n_tested_core = np.array([])

for i in range(len(repert_sims)) :
    print("Processing file ", i+1, " / ", len(repert_sims))
    print("ADC file : ", repert_sims[i])
    # print("Shower file : ", repert_shower[i])
    # print("ShowerSim file : ", repert_showersim[i])
    # print("AN file : ", repert_AN[i])

    file_sims = dh.DataFile(repert_sims[i])
    file_shower = dh.DataFile(repert_shower[i])
    file_showersim = dh.DataFile(repert_showersim[i])
    file_AN = dh.DataFile(repert_AN[i])

    adc = file_sims.tadc
    shower = file_shower.tshower
    showersim = file_showersim.tshowersim


    n_event_i = adc.get_number_of_entries()
    if n_event_i != file_AN.tadc.get_number_of_entries() :
        print("Number of events in ADC and AN files do not match!")
        continue

    file_AN.close()
    print("Number of events in this file : ", n_event_i)

    valid_events_in_file = 0

    for j in range(n_event_i) :
        # print("Processing event ", j+1, " / ", n_event_i)

        adc.get_entry(j)
        shower.get_entry(j)
        showersim.get_entry(j)
        # print("Theta : ", shower.zenith, " -- Phi : ", shower.azimuth)

        GP65_indexes = select_GP65_antennas(adc, j, array_dus)

        # measure the EW component of the voltage for this event, for each antenna that detected it, and save it for the maximum SNR value among all antennas
        n_ant_i = len(GP65_indexes)
        # print("Number of antennas in this event : ", n_ant_i)
        list_traces = np.zeros((n_ant_i,3,1024))

        valid_ant = np.array([], dtype=int) # list of antennas that were triggered (passed T1)

        # if n_ant_i <= 1 :
        #     print("Only one antenna or less in this event, skipping...")
        #     continue

        for k in range(n_ant_i) :
            list_traces[k, 0] = adc.trace_ch[int(GP65_indexes[k])][0]
            list_traces[k, 1] = adc.trace_ch[int(GP65_indexes[k])][1]
            list_traces[k, 2] = adc.trace_ch[int(GP65_indexes[k])][2]

            if pass_threshold(list_traces[k], 
                              threshold=60
                              ) :
                valid_ant = np.append(valid_ant, k)
            
        
        if len(valid_ant) >= 5 : # We want to have at least 5 ant that were triggered
            valid_events_in_file += 1
        #     continue

        theta = np.append(theta, shower.zenith)
        phi = np.append(phi, shower.azimuth)
        n_ant_trig = np.append(n_ant_trig, len(valid_ant))
        energy = np.append(energy, shower.energy_primary)
        n_tested_core = np.append(n_tested_core, np.shape(showersim.tested_cores)[0]) # number of tested cores for this event
        # print("Shapes of arrays : ", len(theta), len(phi), len(n_ant_trig), len(energy), len(n_tested_core))
    # print("Number of events processed : ", len(theta))
    # print("Number of events with at least 5 triggered antennas : ", np.sum(n_ant_trig >= 5))
    print("Number of valid events in this file (with at least 5 triggered antennas) : ", valid_events_in_file)

print("Number of events processed : ", len(theta))
np.save("/sps/grand/jlavoisier/code/quantification/check_sims/out_files/simsCOREASNJ_theta_phi_nant_energy_ntestedcore_SNR5.npy", 
        np.array([theta, phi, n_ant_trig, energy, n_tested_core])
        )

with open("/sps/grand/jlavoisier/code/quantification/check_sims/out_files/simsCOREASNJ_specs.txt", "w") as f :
    f.write("Sim files used for the study : \n")
    for file in repert_sims :
        f.write(file + "\n")
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



repert_sims = sorted(glob('/sps/grand/DC2.1rc4/GP*ZHAireS-AN/*/adc_*_L1_0000.root'))[:50]
repert_shower = sorted(glob('/sps/grand/DC2.1rc4/GP*ZHAireS-AN/*/shower_*_L0_0000.root'))[:50]

theta = np.array([])
phi = np.array([])
energy = np.array([])
n_ant_trig = np.array([])

for i in range(len(repert_sims)) :
    print("Processing file ", i+1, " / ", len(repert_sims))
    print("ADC file : ", repert_sims[i])
    print("Shower file : ", repert_shower[i])

    file_sims = dh.DataFile(repert_sims[i])
    file_shower = dh.DataFile(repert_shower[i])
    adc = file_sims.tadc
    shower = file_shower.tshower
    n_event_i = adc.get_number_of_entries()

    for j in range(n_event_i) :
        # print("Event ", j+1, " / ", n_event_i)

        adc.get_entry(j)
        shower.get_entry(j)
        # print("Theta : ", shower.zenith, " -- Phi : ", shower.azimuth)

        # measure the EW component of the voltage for this event, for each antenna that detected it, and save it for the maximum SNR value among all antennas
        n_ant_i = len(adc.du_id)
        list_traces = np.zeros((n_ant_i,3,1024))
        valid_ant = np.array([], dtype=int) # list of antennas that were triggered (passed T1)

        for k in range(n_ant_i) :
            list_traces[k,0] = adc.trace_ch[int(k)][0]
            list_traces[k,1] = adc.trace_ch[int(k)][1]
            list_traces[k,2] = adc.trace_ch[int(k)][2]

            if pass_threshold(list_traces[k]) :
                valid_ant = np.append(valid_ant, k)
        
        # if len(valid_ant) < 5 : # We want to have at least 5 ant that were triggered
        #     continue

        theta = np.append(theta, shower.zenith)
        phi = np.append(phi, shower.azimuth)
        n_ant_trig = np.append(n_ant_trig, len(valid_ant))
        energy = np.append(energy, shower.energy_primary)
    print("Number of events processed : ", len(theta))

print("Number of events processed : ", len(theta))
np.save("/sps/grand/jlavoisier/code/quantification/check_sims/out_files/simsAN_theta_phi_nant_energy.npy", np.array([theta, phi, n_ant_trig, energy]))

with open("/sps/grand/jlavoisier/code/quantification/check_sims/out_files/simsAN_specs.txt", "w") as f :
    f.write("Sim files used for the study : \n")
    for file in repert_sims :
        f.write(file + "\n")
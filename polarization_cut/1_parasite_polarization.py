import time

import grand.dataio.data_handling as dh
import grand.analysis.fitting as fit
import grand.analysis.signals.extraction as ext
import grand.analysis.constants as cons

import numpy as np
import awkward as ak
import os
import sys
import warnings
from scipy.signal import hilbert
from scipy.optimize import differential_evolution
import psutil

import sqlite3
import argparse

from grand import ECEF, Geodetic, GRANDCS, LTP



file = sys.argv[1]

fname = file.split('/')[-1]



# ---------------------- Constants ----------------------

c_light = 299792458.0 # in m/s, the speed of light in vacuum
c_air = c_light/1.0003 # in m/s, the speed of light in air at sea level
coord_DAQ = Geodetic(latitude=40.99434, longitude=93.94177, height=1262)
coord_origin = coord_DAQ
sigma_time_antennas = 5 # in ns, timing error of antennas
magnetic_field = np.array([26579.3, +34.0, - 50127.7])/np.linalg.norm(np.array([26579.3, +34.0, - 50127.7]))

antenna_path = '/pbs/home/j/jlavoisier/flaginator/flaginator/basics/list_antennas_GP80_xyz_origin_0_rtk.txt'


if "512trace" in fname:
    nb_points_trace = 512
else:
    nb_points_trace = 1024

# nb_points_trace = 1024
# nb_points_trace = 512

def polarization(list_trace,
                 magnetic_field
                 ) :
    """
    Entries:
        trace : shape (3, 1024) trace of the antenna in voltage
        magnetic_field : shape (3,) magnetic field vector at the location of the antenna
    Output:
        polarization of one triggered antenna
    """

    voltage_norm = np.sqrt((list_trace[0])**2 + (list_trace[1])**2 + (list_trace[2])**2)

    volt_pulse = np.argwhere(voltage_norm > 3*np.mean(voltage_norm[nb_points_trace//2:]))
    voltpulse = np.full(nb_points_trace, False)
    if len(volt_pulse) != 0 :
        voltpulse[np.int32(volt_pulse[0][0]):np.int32(volt_pulse[-1][0])] = True
    
    V_vect = np.zeros(3)
    # print(np.sum(pulse))
    if np.sum(voltpulse) > 1 :
        for k in range(3) :
            V_vect[k] = np.max(list_trace[k][voltpulse]) - np.min(list_trace[k][voltpulse])
            e_vect = V_vect / np.sqrt(np.sum(V_vect**2)) # Unitary vector in the direction of V
            scalar_product = np.dot(magnetic_field, e_vect)
    else :
        scalar_product = 2

    return(scalar_product)

# -------------------------------------------------------------

def causal_antennas(file_root,
                    entry_index,
                    list_antennas_xyz) :
    file_root.tadc.get_entry(entry_index)
    file_root.trawvoltage.get_entry(entry_index)

    time_seconds = np.array(file_root.tadc.du_seconds)
    time_nanoseconds = np.array(file_root.tadc.du_nanoseconds)

    du_id_events = file_root.tadc.du_id
    du_id_events_index = np.array([[i, du_id_events[i]] for i in range(len(du_id_events))])

    # Check for empty antennas, if there is, delete them from the time arrays and the du_id_events array
    if 0 in time_seconds :
        index_empty = np.where(time_seconds == 0)[0]
        time_seconds = np.delete(time_seconds, index_empty)
        time_nanoseconds = np.delete(time_nanoseconds, index_empty)
        du_id_events = np.delete(du_id_events, index_empty)
        du_id_events_index = np.delete(du_id_events_index, index_empty, axis=0)

    du_trigg, idx, dupp_du = np.unique(du_id_events, 
                                  return_index=True, 
                                  return_counts=True
                                  )
    du_trigg, dupp_du = du_trigg[np.argsort(idx)], dupp_du[np.argsort(idx)]

    if np.sum(dupp_du) == len(du_trigg) :
        return np.arange(len(du_trigg))
    du_id_dupp_trigg = du_trigg[dupp_du != 1]
    du_id_once_trigg = du_trigg[dupp_du == 1]

    # Time reference is the minimum time in the event
    time_seconds0 = time_seconds - np.min(time_seconds)
    time_nanoseconds0 = time_nanoseconds - time_nanoseconds[np.argmin(time_seconds)]
    time_trigger0 = time_seconds0 + time_nanoseconds0 / 1e9

    # Create an array to store the indices of causal antennas
    causal_du_indices = np.array([], dtype=int)

    # Appending to causal_du_indices the indices of antennas triggered only once
    for k in range(len(du_id_once_trigg)) :
        antenna_index = np.where(du_id_events == du_id_once_trigg[k])[0][0]
        causal_du_indices =  np.append(causal_du_indices, antenna_index)

    # Time stamps of antennas triggered only once
    time_stamp_du_once = time_trigger0[[np.where(du_id_events == du_id_once_trigg[j])[0][0] for j in range(len(du_id_once_trigg))]]

    # Loop over all DUs triggered multiple times
    for du_id in du_id_dupp_trigg :
        list_index = du_id_events_index[du_id_events_index[:,1] == du_id]
        list_times_i = time_trigger0[list_index[:,0]]
        time_diff_sum = np.zeros(len(list_times_i))
        mask_pair_antennas_causal = np.full(len(list_times_i), True, dtype=bool) # check if antennas are causal with each other (if they are not, they will be removed from the list of causal antennas)
        position_du_i = list_antennas_xyz[list_antennas_xyz[:,0] == du_id, 1:4][0]
        
        # Loop over all times this DU was triggered
        for i in range(len(list_times_i)) :
            # Loop over all DUs triggered once
            for j in range(len(time_stamp_du_once)) :
                position_du_j = list_antennas_xyz[list_antennas_xyz[:,0] == du_id_once_trigg[j], 1:4][0]
                dist_i_j = np.linalg.norm(position_du_i - position_du_j)
                # The causal antenna should minimize the sum of |t_i - t_j| - d_ij/c
                time_diff_j = np.abs(time_stamp_du_once[j] - list_times_i[i]) - (dist_i_j / c_air)
                mask_pair_antennas_causal[i] &= (time_diff_j < 0) # if time_diff_j is positive, the two antennas cannot be causal with each other
                time_diff_sum[i] += time_diff_j
                
        if not np.any(mask_pair_antennas_causal) : # if no antenna is causal with any of the antennas triggered once, we cannot determine which one is the causal antenna, so we skip this DU
            continue
        
        antenna_index_min = list_index[mask_pair_antennas_causal][np.argmin(time_diff_sum[mask_pair_antennas_causal]),0]
        causal_du_indices = np.append(causal_du_indices, antenna_index_min)
    
    return causal_du_indices

# ---------------------- Time of triggering in the trace -----------------------

def trigger_time_in_trace(list_trace
                          ):
    """
    Entry:
        list_trace : shape (3, 1024) trace of the antenna in adc
    Output:
        peaktime : time of the peak in the trace (in ns)
    """
    t = np.arange(0, nb_points_trace*2, 2)
    Emodulus = np.sqrt(list_trace[0]**2+list_trace[1]**2+list_trace[2]**2)
    hilbert_amp = np.abs(hilbert(Emodulus))
    peaktime = t[np.int32(np.argmax(hilbert_amp))]

    return(peaktime)

# -------------------------------------------------------

def treat_one_event(root_file_CD,
                    index,
                    list_antennas_xyz
                    ) :
    """
    Entries :
        root_file_CD : the root file being studied
        index : root index of the event studied
    Output :
        True if the event is to be flagged
        False otherwise
        along with info of event
    """
    nb_events = root_file_CD.tadc.get_entries()

    root_file_CD.trawvoltage.get_entry(index)
    root_file_CD.tadc.get_entry(index)
    
    list_ant_indices = causal_antennas(root_file_CD, 
                                        index,
                                        list_antennas_xyz
                                        )
    n_ant = len(list_ant_indices)

    if n_ant < 5 :
        print(f"Event {index} has only {n_ant} antennas triggered, skipping...")
        raise ValueError("Not enough antennas in the event")
    
    
    print("\n------------------ Event ", index, "/", nb_events-1, " ------------------")
    print("Number of antennas in the event: ", n_ant, root_file_CD.tadc.du_id)
    list_traces = np.zeros((n_ant, 3, nb_points_trace), dtype=np.int64)

    # list_antenna_positions = np.zeros((n_ant, 3), dtype=np.float32)
    list_antenna_pos_cartesian = np.zeros((n_ant, 3), dtype=np.float32)
    list_trigger_time_seconds = np.zeros(n_ant)
    list_trigger_time_nanoseconds = np.zeros(n_ant)
    list_du_id = np.zeros(n_ant)

    # Store the traces and the time of trigger of each antenna to use them later
    for i in range(n_ant) :
        # Trace storage in voltage
        try :
            for j in range(3) :
                list_traces[i][j] = root_file_CD.trawvoltage.trace_ch[int(list_ant_indices[i])][int(j+1)]
        except IndexError:
            print(f"Error occurred while accessing trace for antenna {i}")

        du_id = root_file_CD.tadc.du_id[int(list_ant_indices[i])]
        list_du_id = np.append(list_du_id, du_id)
            
        # Position of each antenna
        list_antenna_pos_cartesian[i] = list_antennas_xyz[list_antennas_xyz[:,0] == du_id][0,1:4]

        # list_antenna_positions[i][0] = root_file_CD.trawvoltage.gps_lat[int(list_ant_indices[i])]
        # list_antenna_positions[i][1] = root_file_CD.trawvoltage.gps_long[int(list_ant_indices[i])]
        # list_antenna_positions[i][2] = root_file_CD.trawvoltage.gps_alt[int(list_ant_indices[i])]

        # Time of triggering of antennas
        # For more precision, add the timing of highest amplitude
        peaktime = trigger_time_in_trace(list_traces[i])

        list_trigger_time_nanoseconds[i] = root_file_CD.tadc.du_nanoseconds[int(list_ant_indices[i])] + peaktime
        list_trigger_time_seconds[i] = root_file_CD.tadc.du_seconds[int(list_ant_indices[i])]

    argmin_time_trigger = np.argmin(list_trigger_time_seconds)
    ref_sec = np.min(list_trigger_time_seconds)
    ref_ns = list_trigger_time_nanoseconds[argmin_time_trigger]
    list_trigger_time_seconds = list_trigger_time_seconds - ref_sec
    list_trigger_time_nanoseconds = list_trigger_time_nanoseconds - ref_ns

    list_trigger_time = list_trigger_time_seconds + list_trigger_time_nanoseconds / 1e9


    timing = ref_sec

    t_1 = time.time()
    polar_event = np.zeros(n_ant)
    
    for i in range(n_ant) :
        polar_event[i] = np.abs(polarization(list_traces[i], 
                                      np.array([26579.3, +34.0, - 50127.7])/np.linalg.norm(np.array([26579.3, +34.0, - 50127.7]))))

    med_polar = np.median(polar_event[polar_event < 2])

    t_2 = time.time()
    t_polar = t_2 - t_1

    return(med_polar, t_polar)




write_path = f'/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/quality_cuts/{fname}'
os.makedirs(write_path, exist_ok=True)

file_root = dh.DataFile(file)

print('--------------------------------------------------------------------')
print("Treating file: ", file)
print('--------------------------------------------------------------------')

list_antennas_xyz = np.loadtxt(antenna_path, dtype=np.float32)

polar_list = np.array([])

t_polar_list = np.array([])

nb_events = file_root.tadc.get_entries()

for i in range(nb_events):
# for i in range(5):
    try :
        med_polar, t_polar = treat_one_event(file_root, 
                                            i, 
                                            list_antennas_xyz
                                            )
        polar_list = np.append(polar_list, med_polar)

        t_polar_list = np.append(t_polar_list, t_polar)
    except Exception as e:
        warnings.warn(f"Event {i} could not be treated: {e}")
        continue


np.save(f"{write_path}/polar_list.npy", np.array([
                                                 polar_list
                                                 ]))
np.save(f"{write_path}/time_polar_list.npy", np.array([t_polar_list]))
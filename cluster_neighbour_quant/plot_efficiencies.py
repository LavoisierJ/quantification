import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
from scipy.integrate import trapz

import matplotlib as mpl
# Set default figuresize and font size and labelsize for plots
plt.rcParams['figure.figsize'] = (14, 11)
plt.rcParams['font.size'] = 25
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'


# Extract data

mockCR_n_CR = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/mock_CR/n_CR_RUNs.npy",
                    allow_pickle=True
                    ).item()
mockCR_phi = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/mock_CR/phi_RUNs.npy",
                    allow_pickle=True
                    ).item()
mockCR_theta = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/mock_CR/theta_RUNs.npy",
                    allow_pickle=True
                    ).item()
mockCR_time = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/mock_CR/time_RUNs.npy",
                    allow_pickle=True
                    ).item()

RUNs = np.array(list(mockCR_n_CR.keys()))
RUN_valid = RUNs[np.array([len(mockCR_time[run]) > 0 for run in RUNs])]

# Load extracted data
parasite_time_s = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/noise/t_data.npy", 
                 allow_pickle=True
                 ).item()
parasite_theta = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/noise/theta_data.npy", 
                     allow_pickle=True
                     ).item()
parasite_phi = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/noise/phi_data.npy", 
                    allow_pickle=True
                    ).item()

obs_time_RUNs = np.load(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_quant/preliminary/noise/observation_times.npy", 
                    allow_pickle=True).item()


# Results of cluster algorithm

mockCR_cluster = np.load(f"/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/cluster_param/cluster_RUNs_5s_5deg.npy", 
                 allow_pickle=True
                 ).item()

parasite_cluster = np.load(f"/sps/grand/jlavoisier/output/data_treatment/pipeline_eff/cluster_param/cluster_data_5s_5deg.npy", 
                 allow_pickle=True
                 ).item()



# ----------------------------------------------

# Apply cluster on available runs


# loop_time_window = np.linspace(2, 30, 30)
loop_time_window = np.logspace(np.log10(.1), np.log10(1e5), 45)
loop_angle_window = np.arange(2, 47, 2)*np.pi/180

cut_efficiency_grid_w_zenith = np.zeros((len(loop_time_window), len(loop_angle_window)))
loss_CR_grid_w_zenith = np.zeros((len(loop_time_window), len(loop_angle_window)))

cut_efficiency_grid_or = np.zeros((len(loop_time_window), len(loop_angle_window)))
loss_CR_grid_or = np.zeros((len(loop_time_window), len(loop_angle_window)))




cut_by_zenith = 0 # contains the number of events excluded by zenith cut


consecutive_events_cluster_w_zenith = np.array([])
consecutive_events_cluster_or = np.array([])

total_events= 0
total_CR = 0
indicator_full = np.array([])
zenith_pass_indicator_full = np.array([])


count_parasites_w_zenith = np.zeros((len(loop_time_window), len(loop_angle_window)))
count_parasites_or = np.zeros((len(loop_time_window), len(loop_angle_window)))

count_CR_w_zenith = np.zeros((len(loop_time_window), len(loop_angle_window)))
count_CR_or = np.zeros((len(loop_time_window), len(loop_angle_window)))



for run in RUN_valid:
    time_event = np.array(parasite_time_s[run]) - np.min(parasite_time_s[run])
    theta_event = np.array(parasite_theta[run])
    phi_event = np.array(parasite_phi[run])



    zenith_cut_event = (theta_event > 60 *np.pi/180) & (theta_event < 88 *np.pi/180)  # Example cut, adjust as needed
    # time_event = time_event[~zenith_cut]
    # theta_event = theta_event[~zenith_cut]
    # phi_event = phi_event[~zenith_cut]

    cut_by_zenith += np.sum(~zenith_cut_event)


    indicator = np.zeros(len(time_event))

    time_event = np.append(time_event, np.floor(np.array(mockCR_time[run])))
    theta_event = np.append(theta_event, np.array(mockCR_theta[run]*np.pi/180))
    phi_event = np.append(phi_event, np.array(mockCR_phi[run]*np.pi/180))

    zenith_cut_CR = (np.array(mockCR_theta[run]*np.pi/180) > 60 *np.pi/180) & (np.array(mockCR_theta[run]*np.pi/180) < 88 *np.pi/180)
    cut_by_zenith += np.sum(~zenith_cut_CR)


    zenith_pass_indicator = np.concatenate([zenith_cut_event, zenith_cut_CR])


    indicator = np.append(indicator, np.ones(len(mockCR_time[run])))

    sort = np.argsort(time_event)
    time_event = time_event[sort]
    theta_event = theta_event[sort]
    phi_event = phi_event[sort]

    indicator = indicator[sort]
    zenith_pass_indicator = zenith_pass_indicator[sort]
    zenith_CR_indicator = indicator[zenith_pass_indicator]


    total_events += len(time_event)
    total_CR += len(mockCR_time[run])


    consecutive_mask = np.ones(len(time_event)-1, dtype=bool)
    consecutive_mask_zenith = np.ones(len(time_event[zenith_pass_indicator])-1, dtype=bool)

    time_between_consecutive = np.abs(time_event[1:]-time_event[:-1])
    theta_between_consecutive = np.abs(theta_event[1:]-theta_event[:-1])
    phi_between_consecutive = np.abs((phi_event[1:]-phi_event[:-1]+np.pi) % (2*np.pi) - np.pi)


    time_between_consecutive_w_zenith = np.abs(time_event[zenith_pass_indicator][1:]-time_event[zenith_pass_indicator][:-1])
    theta_between_consecutive_w_zenith = np.abs(theta_event[zenith_pass_indicator][1:]-theta_event[zenith_pass_indicator][:-1])
    phi_between_consecutive_w_zenith = np.abs((phi_event[zenith_pass_indicator][1:]-phi_event[zenith_pass_indicator][:-1]+np.pi) % (2*np.pi) - np.pi)


    for i in range(len(loop_time_window)):
        for j in range(len(loop_angle_window)):
            consecutive_mask[np.all([time_between_consecutive < loop_time_window[i], theta_between_consecutive < loop_angle_window[j], phi_between_consecutive < loop_angle_window[j]], axis=0)] = False
            consecutive_mask_zenith[np.all([time_between_consecutive_w_zenith < loop_time_window[i], theta_between_consecutive_w_zenith < loop_angle_window[j], phi_between_consecutive_w_zenith < loop_angle_window[j]], axis=0)] = False
    

            test_cluster_neighbours_w_zenith = np.ones(len(time_event[zenith_pass_indicator]), dtype=bool) # False if either neighbours are in a cluster
            test_cluster_neighbours_or = np.ones(len(time_event), dtype=bool) # False if both neighbour is in a cluster

            for k in range(len(time_event)):
                if k == 0:
                    test_cluster_neighbours_or[k] = consecutive_mask[k]
                elif k == len(time_event)-1:
                    test_cluster_neighbours_or[k] = consecutive_mask[k-1]
                else:
                    test_cluster_neighbours_or[k] = consecutive_mask[k] or consecutive_mask[k-1] 

            for k in range(len(time_event[zenith_pass_indicator])):
                if k == 0:
                    test_cluster_neighbours_w_zenith[k] = consecutive_mask_zenith[k]
                elif k == len(time_event[zenith_pass_indicator])-1:
                    test_cluster_neighbours_w_zenith[k] = consecutive_mask_zenith[k-1]
                else:
                    test_cluster_neighbours_w_zenith[k] = consecutive_mask_zenith[k] or consecutive_mask_zenith[k-1]

            count_parasites_w_zenith[i, j] += np.sum(test_cluster_neighbours_w_zenith[zenith_CR_indicator==0])
            count_parasites_or[i, j] += np.sum(test_cluster_neighbours_or[indicator==0])
            count_CR_w_zenith[i, j] += np.sum(test_cluster_neighbours_w_zenith[zenith_CR_indicator==1])
            count_CR_or[i, j] += np.sum(test_cluster_neighbours_or[indicator==1])
    
    # consecutive_events_cluster_and = np.append(consecutive_events_cluster_and, test_cluster_neighbours_and)
    # consecutive_events_cluster_or = np.append(consecutive_events_cluster_or, test_cluster_neighbours_or)
    # indicator_full = np.append(indicator_full, indicator)


for i in range(len(loop_time_window)):
    for j in range(len(loop_angle_window)):
        cut_efficiency_grid_w_zenith[i, j] = 100 - count_parasites_w_zenith[i, j]/total_events*100
        loss_CR_grid_w_zenith[i, j] = 100 - count_CR_w_zenith[i, j]/total_CR*100

        cut_efficiency_grid_or[i, j] = 100 - count_parasites_or[i, j]/total_events*100
        loss_CR_grid_or[i, j] = 100 - count_CR_or[i, j]/total_CR*100


# ----------------------------------------

# Plots


# total cut with zenith included



cmap_angle = plt.colormaps['winter']
norm = mpl.colors.Normalize(vmin=0, vmax=45)

plt.figure(figsize=(10, 6))
ax = plt.gca()

for i in range(len(loop_angle_window)):
    color = cmap_angle(norm(loop_angle_window[i] * 180 / np.pi))
    plt.plot(loop_time_window,
             cut_efficiency_grid_or[:, i],
             color=color,
             marker='o',
             label=f'ClusterCut Efficiency' if i == 0 else ""
             )

    plt.plot(loop_time_window,
             loss_CR_grid_or[:, i],
             color=color,
             marker='x',
             label=f'CR Loss Rate' if i == 0 else "")

    plt.plot(loop_time_window,
             cut_efficiency_grid_w_zenith[:, i],
             color=color,
             marker='s',
             linestyle='--',
             label=f'Cluster Cut Efficiency after Zenith cut' if i == 0 else ""
             )

    plt.plot(loop_time_window,
            np.zeros(len(loop_time_window)) + cut_by_zenith / (total_events) * 100 + cut_efficiency_grid_w_zenith[:, i]*(100 - cut_by_zenith / (total_events) * 100)/100,
            color='k',
            linestyle=':',
            label='Cut Efficiency of Zenith cut'
            )
    
# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap_angle, norm=norm)
sm.set_array([])  # This is required for the colorbar to work

fig = plt.gcf()
fig.colorbar(sm, ax=ax, label="Angle window (deg)")

plt.xscale('log')
plt.xlabel('Time Window (s)')
plt.yticks(np.array([0, 20, 40, 60, 80, 100]))
plt.ylabel('Percentage (%)')
# plt.title(f'Total Cut Efficiency and CR Loss Rate vs Time Window at fixed angle when preceeded by a zenith cut',
#           wrap=True)
plt.legend()
plt.grid(True)
plt.savefig(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_neighbour_quant/plots/cut_eff_loss_w_zenith.png", dpi=300)
plt.close()





Xmin = 0.1
index_plot_min = np.argmin(np.abs(loop_angle_window*180/np.pi - Xmin))
Xmax = 100
index_plot_max = np.argmin(np.abs(loop_angle_window*180/np.pi - Xmax))



fig, ax = plt.subplots(1, 2, figsize=(28, 12))
# Plot using pcolor

X0, Y0 = np.meshgrid(loop_angle_window*180/np.pi, loop_time_window[index_plot_min:index_plot_max])

im0 = ax[0].pcolor(X0,
                   Y0,
                   cut_efficiency_grid_w_zenith[index_plot_min:index_plot_max, :], 
                   cmap='copper', 
                #    aspect='auto',
                   # origin='lower',
                   norm=mpl.colors.Normalize(
                                    #  vmin=cut_efficiency_grid_or[index_plot_min:index_plot_max, :].min(), 
                                     vmin=0,
                                    #  vmax=100,
                                     vmax= cut_efficiency_grid_w_zenith[index_plot_min:index_plot_max, :].max(),
                                     ),
           )

CS = ax[0].contour(X0,
                   Y0,
                   cut_efficiency_grid_w_zenith[index_plot_min:index_plot_max, :], 
              np.array([50, 65, 75, 95]), 
              colors='k', 
              origin='lower', 
            #   extent=extent
              )
ax[0].clabel(CS, fontsize=25)

        
fig.colorbar(im0, ax=ax[0], label='Cut efficiency (%) ')
ax[0].set_yscale('log')
ax[0].set_xlabel(r'Angle window (deg)',
           fontsize=30
           )
ax[0].set_ylabel('Time window (s)', 
           fontsize=30
           )



X1, Y1 = np.meshgrid(loop_angle_window*180/np.pi, loop_time_window[index_plot_min:index_plot_max])

im1 = ax[1].pcolor(X1,
                   Y1,
                   loss_CR_grid_w_zenith[index_plot_min:index_plot_max, :], 
                    cmap='viridis', 
                    # aspect='auto',
                    # origin='lower',
                    norm=mpl.colors.Normalize(
                                                vmin=0,
                                                #  vmax=100,
                                                vmax=loss_CR_grid_w_zenith[index_plot_min:index_plot_max, :].max()
                                                ),
                    )

CS = ax[1].contour(X1,
                   Y1,
                   loss_CR_grid_w_zenith[index_plot_min:index_plot_max, :], 
              np.array([0.5, 1, 2, 5, 10]), 
              colors='k', 
            #   origin='lower', 
            #   extent=extent
              )
ax[1].clabel(CS, fontsize=25)


        
fig.colorbar(im1, ax=ax[1], label='CR Loss rate (%) ')

ax[1].set_yscale('log')
ax[1].set_xlabel(r'Angle window (deg)',
           fontsize=30
           )
ax[1].set_ylabel('Time window (s)', 
           fontsize=30
           )

plt.savefig(f"/pbs/home/j/jlavoisier/pipeline_lab/cluster_neighbour_quant/plots/cut_eff_loss_w_zenith_2D.png", dpi=300)
plt.close()
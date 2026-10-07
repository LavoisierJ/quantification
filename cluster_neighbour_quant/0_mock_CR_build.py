"""
Building mock cosmic ray data for testing the cluster quantification pipeline.
We extract the observation times of the specified RUNs, and build a mock dataset with the same observation times, but with random noise instead of real data.
The mock dataset is saved in the same directory as the real data, with the same structure, but with a different prefix in the filename to indicate that it is mock data.
"""

import numpy as np
from glob import glob
from scipy.integrate import trapz

def calculate_PAO_spectrum(e_eV):
    """
    Calculate the CR flux measured by PAO in 10 ** 17 eV < E < 10 ** 20 eV
    with the SD-750 and SD-1500.
    The formula of the spectrum is Equation (13) of
    Abreu et al., Eur. Phys. J. C 81, 966 (2021)
    (https://doi.org/10.1140/epjc/s10052-021-09700-w)
    """


    j_0 = 1.309e-18 # km^-2 yr^-1 sr^-1 eV^-1
    e_0 = 10 ** 18.5 # eV
    omega_01, omega_12, omega_23, omega_34 = 0.43, 0.05, 0.05, 0.05
    g_0, g_1, g_2, g_3, g_4 = 2.64, 3.298, 2.52, 3.08, 5.2
    e_01, e_12, e_23, e_34 = 1.24e17, 4.9e18, 1.4e19, 4.7e19 # eV

    j = j_0 * (e_eV/e_0) ** (-g_0) \
            * (1 + (e_eV/e_01) ** (1/omega_01)) ** ((g_0-g_1) * omega_01) \
            * (1 + (e_eV/e_12) ** (1/omega_12)) ** ((g_1-g_2) * omega_12) \
            * (1 + (e_eV/e_23) ** (1/omega_23)) ** ((g_2-g_3) * omega_23) \
            * (1 + (e_eV/e_34) ** (1/omega_34)) ** ((g_3-g_4) * omega_34) \
            / (1 + (e_0/e_01)  ** (1/omega_01)) ** ((g_0-g_1) * omega_01) \
            / (1 + (e_0/e_12)  ** (1/omega_12)) ** ((g_1-g_2) * omega_12) \
            / (1 + (e_0/e_23)  ** (1/omega_23)) ** ((g_2-g_3) * omega_23) \
            / (1 + (e_0/e_34)  ** (1/omega_34)) ** ((g_3-g_4) * omega_34)

    return j

def number_CR(energy,
              energy_tot,
              theta,
              theta_tot,
              energy_bins,
              theta_bins,
              array_area,
              time_observation
              ):
    """
    Inputs:
    - energy: array of energies of the events that triggered the array, in eV
    - energy_tot: array of energies of all events in the simulation, in eV
    - theta: array of zenith angles of the events that triggered the array, in degrees
    - theta_tot: array of zenith angles of all events in the simulation, in degrees
    - energy_bins: array of energy bin edges in eV
    - theta_bins: array of zenith angle bin edges in degrees
    - array_area: area of the array in km^2
    - time_observation: observation time in seconds
    Output:
        number of CR events seen by the array, computed the "N_CR2" way:
        n_CR = int_E J(E) * exposure_E(E) dE
        with exposure_E(E) = 2*pi * t_obs * S * int_theta tau(E, theta) sin(theta) dtheta
        where tau(E, theta) is the 2D trigger efficiency obtained from
        sin(theta)-weighted histograms. The extra cos(theta) on the triggered
        sample makes efficiency_2d an *effective* (projection-weighted)
        efficiency, so that sin(theta) * efficiency_2d integrates
        tau(E, theta) cos(theta) sin(theta) over the solid angle, exactly as
        in the original exposure_E / N_CR2 computation.
    """

    years_to_s = 365.25 * 24 * 3600

    # ----------- Convert units -----------
    energy_eV = np.asarray(energy)
    energy_tot_eV = np.asarray(energy_tot)
    theta_rad = np.radians(np.asarray(theta))     # deg -> rad
    theta_tot_rad = np.radians(np.asarray(theta_tot))

    theta_bins = np.radians(theta_bins)

    total_surface = array_area * 1e6              # km^2 -> m^2
    total_time = time_observation                  # s

    # ----------- Weights flattening the simulation distribution -----------
    # log_10(1 / cos) : to get a uniform distribution in cos(zenith);
    # triggered events additionally carry the cos(zenith) projection factor
    weights_triggered = energy_eV / np.log10(1 / np.cos(theta_rad))
    weights_all = energy_tot_eV / np.log10(1 / np.cos(theta_tot_rad))

    # ----------- 2D histograms and trigger efficiency -----------
    hist_triggered_2d = np.histogram2d(energy_eV, 
                                       theta_rad,
                                       bins=[energy_bins, theta_bins],
                                       weights=weights_triggered
                                       )[0]
    
    hist_all_2d = np.histogram2d(energy_tot_eV, 
                                 theta_tot_rad,
                                 bins=[energy_bins, theta_bins],
                                 weights=weights_all
                                 )[0]

    # empty denominator bins -> efficiency set to 0
    efficiency_2d = np.divide(hist_triggered_2d, 
                              hist_all_2d,
                              out=np.zeros_like(hist_triggered_2d, dtype=float),
                              where=hist_all_2d > 0
                              )
    # print(efficiency_2d)


    # ----------- Exposure per energy bin (integrated over zenith) -----------
    mid_bins_energy = np.sqrt(energy_bins[:-1] * energy_bins[1:])
    mid_bins_zenith = 0.5 * (theta_bins[:-1] + theta_bins[1:])

    # exposure_E = (2 * np.pi * total_time * total_surface
    #               * np.trapz(np.sin(mid_bins_zenith)[None, :] * np.cos(mid_bins_zenith)[None, :] * efficiency_2d,
    #                          mid_bins_zenith, axis=1))  # m^2 s sr, per energy bin

    exposure_E = 2 * np.pi * total_time * total_surface * np.sin(mid_bins_zenith)[None, :] * np.cos(mid_bins_zenith)[None, :] * efficiency_2d

    exposure_E = np.trapz(exposure_E, mid_bins_zenith, axis=1)  # m^2 s sr, per energy bin

    # ----------- Number of CR events -----------
    # J in km^-2 yr^-1 sr^-1 eV^-1 -> m^-2 s^-1 sr^-1 eV^-1
    flux = calculate_PAO_spectrum(mid_bins_energy) * 1e-6 / years_to_s

    n_CR = np.trapz(flux * exposure_E, mid_bins_energy)

    return int(np.floor(n_CR))

def draw_mock_CR(n_CR, 
                 obs_time,
                 bins_theta, 
                 prob_dens_theta
                 ):
    """
    Draw n_CR mock cosmic rays from the distributions of energy and zenith angle.
    Inputs:
        n_CR: number of cosmic rays to draw
        obs_time: observation time in seconds
        bins_theta: array of zenith angle bin edges for histogramming
        prob_dens_theta: array of probability densities for each zenith angle bin
    Output:
        drawn times, zenith and azimuth angles
    """
    time_CR = np.array([np.random.uniform(low=0, high=obs_time) for i in range(n_CR)])

    # Zenith angle are generated according to the array's sensibility in zenith
    bin_indices_theta = np.random.choice(len(bins_theta)-1, 
                                        size=n_CR, 
                                        p=prob_dens_theta[0]
                                        )
    theta_CR = np.array([np.random.uniform(low=bins_theta[bin_indices_theta[i]], 
                                        high=bins_theta[bin_indices_theta[i] + 1])
                                        for i in range(n_CR)])

    # Azimuth is generated uniformly since it doesn't have particular sensibility to azimuth arrival
    phi_CR = np.array([np.random.uniform(low=0, high=360) for i in range(n_CR)])

    return time_CR, theta_CR, phi_CR


# load the information from the simulations to get the distributions of energy and zenith angle
name_sims = "GP300NJ"

obs_time = np.load(f"/sps/grand/jlavoisier/code/quantification/check_data/observation_times.npy", 
                    allow_pickle=True)

RUNs = list(obs_time.item().keys())
times = list(obs_time.item().values())

print(f"Building mock cosmic ray data for the following RUNs: {RUNs}")

print(f"Observation times (in seconds): {times}")

# ---------------------------------------------
path = f"/sps/grand/jlavoisier/code/quantification/check_sims/out_files/sims{name_sims}_theta_n_ant_energy.npy"
file_sims = np.load(path)

theta_tot = file_sims[0]
phi_tot = file_sims[1]
n_ant_trig_tot = file_sims[2]
energy_tot = file_sims[3]

theta = theta_tot[n_ant_trig_tot >= 5]
phi = phi_tot[n_ant_trig_tot >= 5]
energy = energy_tot[n_ant_trig_tot >= 5]


log_bins_energy = np.logspace(np.log10(energy.min()),
                       np.log10(energy.max()),
                       20)

bins_theta = np.linspace(theta.min(), theta.max()+1, 15)

total_surface = (11.3 - (-5.2)) * (6.5 - (-4.5)) # in km^2
# ---------------------------------------------


# number of CR events seen by GP65 in the observation time of the RUNs
n_CR_RUNs = {}
# n_CR_RUNs.item().keys() = RUNs

for i, run in enumerate(RUNs):
    obs_time = times[i]
    n_CR_RUNs[run] = number_CR(energy*1e9, # in eV
                               energy_tot*1e9,
                               theta,
                               theta_tot,
                               log_bins_energy*1e9,
                               bins_theta,
                               array_area=total_surface, # km^2
                               time_observation=obs_time # s
                               )

# Drawing n_CR mock CR for each RUN

distrib_theta = np.load(f"/sps/grand/jlavoisier/code/quantification/check_sims/out_files/sims{name_sims}_theta_dens.npy")
prob_dens_theta = distrib_theta / np.sum(distrib_theta)
bins_theta = np.load(f"/sps/grand/jlavoisier/code/quantification/check_sims/out_files/sims{name_sims}_theta_bins.npy")[0]



time_CR_RUNs = {}
theta_CR_RUNs = {}
phi_CR_RUNs = {}

for i, run in enumerate(RUNs):
    print(f"Drawing mock cosmic rays for RUN {run} with observation time {times[i]} seconds: {n_CR_RUNs[run]} events")
    obs_time = times[i]
    n_CR = n_CR_RUNs[run]
    time_CR, theta_CR, phi_CR = draw_mock_CR(n_CR, 
                                             obs_time, 
                                             bins_theta, 
                                             prob_dens_theta
                                             )
    time_CR_RUNs[run] = time_CR
    theta_CR_RUNs[run] = theta_CR
    phi_CR_RUNs[run] = phi_CR

print(f"Mock cosmic ray data built for the following RUNs: {n_CR_RUNs}")

np.save(f"/sps/grand/jlavoisier/code/quantification/cluster_neighbour_quant/mockCR_dataset/n_CR_RUNs.npy", n_CR_RUNs)
np.save(f"/sps/grand/jlavoisier/code/quantification/cluster_neighbour_quant/mockCR_dataset/time_RUNs.npy", time_CR_RUNs)
np.save(f"/sps/grand/jlavoisier/code/quantification/cluster_neighbour_quant/mockCR_dataset/theta_RUNs.npy", theta_CR_RUNs)
np.save(f"/sps/grand/jlavoisier/code/quantification/cluster_neighbour_quant/mockCR_dataset/phi_RUNs.npy", phi_CR_RUNs)
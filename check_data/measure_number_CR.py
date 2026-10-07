"""

Calculate the number of cosmic rays falling on GP65 depneding on an observation time

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

# first measure the number of CRs detected by/falling on the array to know how many "CRs" to create

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

# def integrate_2d_histogram(histogram, energy_bins, angle_bins):
#     """
#     Integrate a 2D histogram over both dimensions.

#     Parameters:
#     - histogram: 2D numpy array, shape (n_energy_bins, n_angle_bins)
#     - energy_bins: 1D array, edges of the logarithmic energy bins
#     - angle_bins: 1D array, edges of the linear angle bins

#     Returns:
#     - integral: float, the integral of the histogram over both dimensions
#     """
#     # Calculate energy bin widths (logarithmic)
#     energy_bin_widths = np.diff(energy_bins)
#     # print(f"Energy bin widths: {energy_bin_widths}")


#     # Calculate angle bin widths (linear)
#     angle_bin_widths = np.diff(angle_bins)
#     # print(f"Angle bin widths: {angle_bin_widths}")

#     # Reshape bin widths for broadcasting
#     energy_widths_2d = energy_bin_widths[:, np.newaxis]  # shape: (n_energy_bins, 1)
#     angle_widths_2d = angle_bin_widths[np.newaxis, :]    # shape: (1, n_angle_bins)
#     # print(f"Histogram shape: {histogram.shape}, Energy widths 2D shape: {energy_widths_2d.shape}, Angle widths 2D shape: {angle_widths_2d.shape}")

#     # Calculate the integral
#     # print(histogram * energy_widths_2d * angle_widths_2d)
#     integral = np.sum(histogram * energy_widths_2d * angle_widths_2d)

#     return integral

def integrate_2d_histogram_log(histogram, 
                               energy_bins, 
                               angle_bins
                               ):
    """
    Integrate a 2D histogram over both dimensions.

    Parameters:
    - histogram: 2D numpy array, shape (n_energy_bins, n_angle_bins)
    - energy_bins: 1D array, edges of the logarithmic energy bins
    - angle_bins: 1D array, edges of the linear angle bins

    Returns:
    - integral: float, the integral of the histogram over both dimensions
    """
    # Calculate energy bin widths (logarithmic)
    log_energy_bin_widths = np.diff(np.log10(energy_bins))[0]
    energies = energy_bins[:-1, np.newaxis]  # shape: (n_energy_bins-1, 1)
    histogram = energies * histogram  # Convert to dN/dlogE

    # print(f"Energy bin widths: {energy_bin_widths}")


    # Calculate angle bin widths (linear)
    angle_bin_widths = np.diff(angle_bins)[0]
    # print(f"Angle bin widths: {angle_bin_widths}")

    # Reshape bin widths for broadcasting
    # log_energy_widths_2d = log_energy_bin_widths[:, np.newaxis]  # shape: (n_energy_bins, 1)
    # angle_widths_2d = angle_bin_widths[np.newaxis, :]    # shape: (1, n_angle_bins)
    # print(f"Histogram shape: {histogram.shape}, Energy widths 2D shape: {energy_widths_2d.shape}, Angle widths 2D shape: {angle_widths_2d.shape}")

    # Calculate the integral
    # print(histogram * energy_widths_2d * angle_widths_2d)
    integral = np.sum(histogram) * log_energy_bin_widths * angle_bin_widths

    return integral

def integrate_2d_histogram(histogram, 
                           energy_bins, 
                           angle_bins
                           ):
    """
    Integrate a 2D histogram over both dimensions.
    """
    # Calculate energy bin widths (logarithmic)
    energy_bin_widths = np.diff(energy_bins)
    # Calculate angle bin widths (linear)
    angle_bin_widths = np.diff(angle_bins)

    # Reshape bin widths for broadcasting
    energy_widths_2d = energy_bin_widths[:, np.newaxis]  # shape: (n_energy_bins, 1)
    angle_widths_2d = angle_bin_widths[np.newaxis, :]    # shape: (1, n_angle_bins)

    # Calculate the integral
    integral = np.sum(histogram * energy_widths_2d * angle_widths_2d)

    return integral


def integration_AAfragpy(E, Y):

    E1 = E[:-1]
    E2 = E[1:]
    Y1 = Y[:-1]
    Y2 = Y[1:]
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(Y2 / Y1) / np.log(E2 / E1) + 1.0
        INT = (Y2 * E2 - Y1 * E1) / log_term
        INT[np.isnan(INT)] = 0.0
    return np.sum(INT)

def number_CR_old(energy,
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
    - energy: array of energies of the events that triggered the array
    - energy_tot: array of energies of all events in the simulation
    - theta: array of zenith angles of the events that triggered the array
    - theta_tot: array of zenith angles of all events in the simulation
    - energy_bins: array of energy bin edges for histogramming
    - theta_bins: array of zenith angle bin edges for histogramming
    - array_area: area of the array in m^2
    - time_observation: observation time in seconds
    Output:
        number of CR events seen by GP65
    """
    clem_weights_trigg = calculate_PAO_spectrum(energy * 1e9) * np.cos(np.deg2rad(theta)) * np.sin(np.deg2rad(theta))
    clem_weights_all = calculate_PAO_spectrum(energy_tot * 1e9) * np.sin(np.deg2rad(theta_tot))

    clem_hist_trigg = np.histogram(energy, 
                                   bins=energy_bins, 
                                   weights=clem_weights_trigg)[0]
    clem_hist_all = np.histogram(energy_tot, 
                                 bins=energy_bins, 
                                 weights=clem_weights_all)[0]

    mid_bins_energy = (energy_bins[:-1] + energy_bins[1:]) / 2

    exposure = clem_hist_trigg / clem_hist_all * time_observation * 2 * np.pi * (-np.cos(np.max(theta*np.pi/180)) + np.cos(np.min(theta*np.pi/180))) * array_area # in m^2 s sr

    years_to_s = 365.25 * 24 * 3600

    # return(int(np.floor(trapz(calculate_PAO_spectrum(mid_bins_energy*1e9) * exposure, mid_bins_energy*1e9) * 1e-6 / years_to_s)))
    return(int(np.floor(integration_AAfragpy(mid_bins_energy*1e9, 
                                             calculate_PAO_spectrum(mid_bins_energy*1e9) * exposure) * 1e-6 / years_to_s)))


def number_CR_no_work(energy,
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
    - energy: array of energies of the events that triggered the array in eV
    - energy_tot: array of energies of all events in the simulation in eV
    - theta: array of zenith angles of the events that triggered the array
    - theta_tot: array of zenith angles of all events in the simulation
    - energy_bins: array of energy bin edges for histogramming
    - theta_bins: array of zenith angle bin edges for histogramming
    - array_area: area of the array in km^2
    - time_observation: observation time in seconds
    Output:
        number of cosmic ray events seen by the array
        contrary to the other function, we here interagte a 2d-histogram in energy and zenith
        number of CR events seen by GP65:
        n_CR = 2π t_obs S_GP65 \iint J(E) τ(E, θ) cos(θ) sin(θ) dE dθ
    """
    
    # ----------- Trigger efficiency of the array -----------
    trigger_rate_num = np.histogram2d(energy, 
                                      theta, 
                                      bins=[energy_bins, theta_bins],
                                      weights=energy / np.log10(1 / np.cos(theta*np.pi/180)) # weights due to the simulaion distribution
                                      )[0]

    trigger_rate_den = np.histogram2d(energy_tot, 
                                          theta_tot, 
                                          bins=[energy_bins, theta_bins],
                                          weights=energy_tot / np.log10(1 / np.cos(theta_tot*np.pi/180)) 
                                          )[0]

    trigger_rate = trigger_rate_num / trigger_rate_den


    # ----------- Natural weights due to the PAO spectrum and the solid angle -----------

    mid_bins_E = np.sqrt(energy_bins[:-1] * energy_bins[1:]) # logarithmic binning, so we take the geometric mean
    mid_bins_theta = 0.5 * (theta_bins[:-1] + theta_bins[1:]) # linear binning, so we take the arithmetic mean

    table_mid_bins = np.meshgrid(mid_bins_E, mid_bins_theta, indexing='ij')
    mid_bins_E_tot = table_mid_bins[0].ravel()
    mid_bins_theta_tot = table_mid_bins[1].ravel()

    omega_E = calculate_PAO_spectrum(mid_bins_E) # in km^-2 yr^-1 sr^-1 eV^-1
    omega_theta = np.cos(np.deg2rad(mid_bins_theta)) * np.sin(np.deg2rad(mid_bins_theta))
    print(omega_theta)

    table_omega = np.matmul(omega_E[:, None], omega_theta[None, :]).ravel()


    # print(f"mid_bins_E_tot : {mid_bins_E_tot}\n mid_bins_theta_tot: {mid_bins_theta_tot}")

    omega_E_theta = np.histogram2d(mid_bins_E_tot,
                                    mid_bins_theta_tot,
                                    bins=[energy_bins, theta_bins],
                                    weights=table_omega
                                    )[0]

    integrand = omega_E_theta * trigger_rate
    # integrand = omega_E_theta


    X, Y = np.meshgrid(np.sqrt(energy_bins[:-1] * energy_bins[1:]), 0.5 * (theta_bins[:-1] + theta_bins[1:]), indexing='ij')
    plt.pcolor(X, Y, 2 * np.pi * (time_observation / (365.25 * 24 * 3600)) * array_area * integrand, shading='auto')
    plt.colorbar(label='Trigger Efficiency')
    plt.xlabel('Energy (eV)')
    plt.ylabel('Zenith Angle (rad)')
    plt.title('2D Histogram of Trigger Efficiency')
    plt.savefig('/pbs/home/j/jlavoisier/pipeline_lab/check_data/trigger_rate_moi.png')
    plt.close()


    # ----------- Integrate the 2D histogram over both dimensions -----------
    integral = integrate_2d_histogram(integrand, 
                                          energy_bins, 
                                          theta_bins
                                          )

    # ----------- Calculate the number of CR events seen by GP65 -----------

    years_to_s = 365.25 * 24 * 3600

    n_CR = 2 * np.pi * (time_observation / years_to_s) * array_area * integral

    # return int(np.floor(n_CR))
    return n_CR


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

    return n_CR


# # data obtained after looking at sims passing at least 5 antennas with a threshold 
# # (or T1 depending on the analysis)

# load the information from the simulations to get the distributions of energy and zenith angle
name_sims = "GP300NJ"

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
# obs_time = 846672 # in seconds
obs_time = 34.4*24*3600 # in seconds
# obs_time = 1*24*3600 # in seconds

print(energy.min(), energy.max())

nb_CR_obstime = number_CR(energy*1e9, # in eV
                          energy_tot*1e9,
                          theta,
                          theta_tot,
                          log_bins_energy*1e9,
                          bins_theta,
                          array_area=total_surface, # km^2
                          time_observation=obs_time # s
                          )

print(f"Number of CRs falling on GP65 in {obs_time/3600} hours : {nb_CR_obstime}")


"""
物理常数与单位转换模块
Heavy-ion collision physics: natural units with GeV, fm, c=1
"""
import numpy as np

# Fundamental constants in natural units (hbar = c = k_B = 1)
HBAR_C = 0.1973269804  # GeV·fm
BOLTZMANN = 1.0  # In natural units
PROTON_MASS = 0.938272  # GeV/c^2
NEUTRON_MASS = 0.939565  # GeV/c^2
PION_MASS = 0.13957  # GeV/c^2
KAON_MASS = 0.49368  # GeV/c^2

# Collision system parameters
NUCLEON_CROSS_SECTION = 4.2  # mb (nucleon-nucleon inelastic at sqrt(s)=200 GeV)
NUCLEON_CROSS_SECTION_FM2 = NUCLEON_CROSS_SECTION * 0.1  # Convert mb to fm^2

# QGP critical temperature
TC_QCD = 0.154  # GeV (~154 MeV, lattice QCD crossover)
TC_QCD_MEV = 154.0

# Typical heavy-ion collision scales
TYPICAL_INITIAL_TEMP = 0.400  # GeV (400 MeV at LHC)
TYPICAL_FREEZEOUT_TEMP = 0.120  # GeV (120 MeV chemical freeze-out)

# Transport coefficients
ETA_S_MIN = 1.0 / (4.0 * np.pi)  # KSS bound: η/s ≥ 1/(4π)
ETA_S_QGP = 0.08  # Typical QGP shear viscosity over entropy density
ZETA_S_PEAK = 0.04  # Bulk viscosity peak near Tc

# Nuclear geometry
GOLDEN_RATIO = (1.0 + np.sqrt(5.0)) / 2.0

# Wood-Saxon parameters for Au (Gold nucleus)
AU_RADIUS = 6.38  # fm
AU_DIFFUSENESS = 0.535  # fm
AU_MASS_NUMBER = 197
AU_ATOMIC_NUMBER = 79

# Wood-Saxon parameters for Pb (Lead nucleus)
PB_RADIUS = 6.62  # fm
PB_DIFFUSENESS = 0.546  # fm
PB_MASS_NUMBER = 208
PB_ATOMIC_NUMBER = 82

def convert_energy_to_temperature(energy_gev):
    """
    Convert energy scale to temperature: T = E/k_B (natural units)

    Parameters:
    -----------
    energy_gev : float
        Energy in GeV

    Returns:
    --------
    float
        Temperature in GeV
    """
    return energy_gev / BOLTZMANN

def convert_fm_to_gev_inv(length_fm):
    """
    Convert length from fm to GeV^{-1}: 1 fm = 1/0.197 GeV^{-1}

    Parameters:
    -----------
    length_fm : float
        Length in femtometers

    Returns:
    --------
    float
        Length in GeV^{-1}
    """
    return length_fm / HBAR_C

def convert_gev_inv_to_fm(length_gev_inv):
    """
    Convert length from GeV^{-1} to fm

    Parameters:
    -----------
    length_gev_inv : float
        Length in GeV^{-1}

    Returns:
    --------
    float
        Length in femtometers
    """
    return length_gev_inv * HBAR_C

def compute_de_broglie_wavelength(temperature_gev):
    """
    Thermal de Broglie wavelength: λ = ħc/(2πT)

    Parameters:
    -----------
    temperature_gev : float
        Temperature in GeV

    Returns:
    --------
    float
        Wavelength in fm
    """
    if temperature_gev <= 0:
        return np.inf
    return HBAR_C / (2.0 * np.pi * temperature_gev)

def compute_knudsen_number(mean_free_path_fm, system_size_fm):
    """
    Knudsen number: Kn = λ_mfp / L

    Parameters:
    -----------
    mean_free_path_fm : float
        Mean free path in fm
    system_size_fm : float
        System size in fm

    Returns:
    --------
    float
        Dimensionless Knudsen number
    """
    if system_size_fm <= 0:
        return np.inf
    return mean_free_path_fm / system_size_fm

def compute_reynolds_number(velocity_fm_per_fm, length_fm, kinematic_viscosity_fm2_per_fm):
    """
    Reynolds number for QGP: Re = vL/ν

    Parameters:
    -----------
    velocity_fm_per_fm : float
        Flow velocity (dimensionless in natural units)
    length_fm : float
        Characteristic length in fm
    kinematic_viscosity_fm2_per_fm : float
        Kinematic viscosity ν = η/(ε+P) in fm

    Returns:
    --------
    float
        Dimensionless Reynolds number
    """
    if kinematic_viscosity_fm2_per_fm <= 0:
        return np.inf
    return velocity_fm_per_fm * length_fm / kinematic_viscosity_fm2_per_fm

def stefan_boltzmann_constant_nb_flavors(nb_flavors=3):
    """
    Stefan-Boltzmann constant for QGP with N_f flavors:
    ε_SB = (π^2/30) * (16 + 21*N_f/2) * T^4

    Parameters:
    -----------
    nb_flavors : int
        Number of quark flavors (default 3: u, d, s)

    Returns:
    --------
    float
        Coefficient in GeV^4 such that ε = σ * T^4
    """
    gluon_dof = 16  # 8 colors × 2 polarizations
    quark_dof = 7.0 / 8.0 * 2.0 * 3.0 * nb_flavors  # particle/antiparticle × spin × color × flavors
    return (np.pi**2 / 30.0) * (gluon_dof + quark_dof)

def ideal_entropy_density(temperature_gev, nb_flavors=3):
    """
    Ideal QGP entropy density: s = (4/3) * ε/T = (4ε_SB/3) * T^3

    Parameters:
    -----------
    temperature_gev : float
        Temperature in GeV

    Returns:
    --------
    float
        Entropy density in fm^{-3}
    """
    sigma = stefan_boltzmann_constant_nb_flavors(nb_flavors)
    energy_density = sigma * temperature_gev**4
    return (4.0 / 3.0) * energy_density / temperature_gev

def ideal_pressure(temperature_gev, nb_flavors=3):
    """
    Ideal QGP pressure: P = ε/3 = (σ/3) * T^4

    Parameters:
    -----------
    temperature_gev : float
        Temperature in GeV

    Returns:
    --------
    float
        Pressure in GeV/fm^3
    """
    sigma = stefan_boltzmann_constant_nb_flavors(nb_flavors)
    return sigma * temperature_gev**4 / 3.0

def compute_sound_speed_squared(temperature_gev, nb_flavors=3):
    """
    Speed of sound squared: c_s^2 = dP/dε

    For ideal massless gas: c_s^2 = 1/3
    For interacting QGP near Tc: c_s^2 dips below 1/3

    Parameters:
    -----------
    temperature_gev : float
        Temperature in GeV

    Returns:
    --------
    float
        c_s^2 (dimensionless)
    """
    # Simple parameterization with softening near Tc
    T_ratio = temperature_gev / TC_QCD
    if T_ratio < 1.0:
        # Hadronic phase: c_s^2 ≈ 0.15
        return 0.15
    elif T_ratio < 1.5:
        # Crossover region: smooth interpolation
        x = (T_ratio - 1.0) / 0.5
        return 0.15 + (1.0/3.0 - 0.15) * (3*x**2 - 2*x**3)
    else:
        # QGP phase: approaches ideal value
        return 1.0 / 3.0

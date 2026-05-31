"""
Monte Carlo Neutron Transport for ICF Burn Diagnosis

Based on:
- niederreiter2 (Project 803): Quasi-random sequences for QMC variance reduction
- rng_cliff (Project 1039): Cliff random number generator
- histogramize (Project 543): Energy spectrum binning

Models:
- Neutron transport through spherical DT plasma
- Quasi-Monte Carlo path sampling
- Energy spectrum histogramization
- Escape probability and deposition profile
"""

import numpy as np

# Physical constants
M_N = 1.67492749804e-27  # Neutron mass [kg]
C_LIGHT = 2.99792458e8
E_CHARGE = 1.602176634e-19
SIGMA_DT = 5.0e-28  # Approximate DT cross-section [m^2] at 14.1 MeV


def niederreiter2_generate(dim_num, n, key=0):
    """
    Generate Niederreiter quasi-random sequence in [0,1]^dim_num.
    Based on niederreiter2_generate from Project 803.
    Simplified implementation using Sobol-like properties.
    """
    # Use a simple quasi-random sequence based on radical inverse
    r = np.zeros((dim_num, n))
    for j in range(n):
        for d in range(dim_num):
            # Van der Corput sequence with base 2 for d=0, base 3 for d=1, etc.
            base = 2 + d
            k = j + key + 1
            x = 0.0
            f = 1.0 / base
            while k > 0:
                x += f * (k % base)
                k //= base
                f /= base
            r[d, j] = x
    return r


def rng_cliff_next(x):
    """
    Cliff random number generator.
    Based on rng_cliff_next from Project 1039.
    """
    if x <= 0.0 or x >= 1.0:
        return 0.5
    return (-100.0 * np.log(x)) % 1.0


def rng_cliff_sequence(n, seed=0.5):
    """
    Generate sequence of Cliff random numbers.
    """
    seq = np.zeros(n)
    x = seed
    for i in range(n):
        x = rng_cliff_next(x)
        seq[i] = x
    return seq


def sample_neutron_source_spherical(n_neutrons, r_fuel, energy_MeV=14.1):
    """
    Sample initial neutron positions uniformly in sphere,
    and isotropic directions.
    """
    # Use quasi-random for positions
    qr = niederreiter2_generate(4, n_neutrons)

    # r = r_fuel * u^(1/3)
    r = r_fuel * qr[0]**(1.0 / 3.0)

    # theta = arccos(2*v - 1)
    theta = np.arccos(2.0 * qr[1] - 1.0)
    phi = 2.0 * np.pi * qr[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z_pos = r * np.cos(theta)

    # Isotropic directions
    theta_d = np.arccos(2.0 * qr[2] - 1.0)
    phi_d = 2.0 * np.pi * qr[3]

    ux = np.sin(theta_d) * np.cos(phi_d)
    uy = np.sin(theta_d) * np.sin(phi_d)
    uz = np.cos(theta_d)

    energy = np.full(n_neutrons, energy_MeV * 1e6 * E_CHARGE)  # J

    return x, y, z_pos, ux, uy, uz, energy


def macroscopic_cross_section(rho, Z_bar):
    """
    Approximate total macroscopic cross section [m^-1].
    """
    m_u = 1.66053906660e-27
    n_ion = rho / (2.5 * m_u)
    n_DT = n_ion / 2.0  # D and T each
    sigma_tot = SIGMA_DT
    return n_DT * sigma_tot


def transport_neutron(x0, y0, z0, ux, uy, uz, energy, r_outer, rho_func,
                      max_steps=1000):
    """
    Track single neutron through spherical target.
    Returns: escaped (bool), path_length, final_energy, deposition_profile.
    """
    x, y, z = x0, y0, z0
    path_length = 0.0
    deposition = 0.0

    for _ in range(max_steps):
        r = np.sqrt(x**2 + y**2 + z**2)
        if r > r_outer:
            return True, path_length, energy, deposition

        rho = rho_func(r)
        if rho <= 0.0:
            return True, path_length, energy, deposition

        sigma_t = macroscopic_cross_section(rho, 1.0)
        if sigma_t <= 1e-30:
            return True, path_length, energy, deposition

        # Sample free flight distance
        xi = np.random.random()
        if xi <= 0.0:
            xi = 1e-10
        s = -np.log(xi) / sigma_t

        # Move neutron
        x += s * ux
        y += s * uy
        z += s * uz
        path_length += s

        # Collision: deposit energy (simplified)
        deposition += energy * 0.01  # 1% energy deposit per collision
        energy *= 0.99

        # Isotropic scatter
        theta = np.arccos(2.0 * np.random.random() - 1.0)
        phi = 2.0 * np.pi * np.random.random()
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)

        if energy < 1e3 * E_CHARGE:
            break

    r = np.sqrt(x**2 + y**2 + z**2)
    escaped = r > r_outer
    return escaped, path_length, energy, deposition


def run_mc_neutron_transport(n_neutrons, r_fuel, r_outer, rho_func,
                              energy_MeV=14.1, max_steps=1000,
                              use_quasi_random=True):
    """
    Run Monte Carlo neutron transport simulation.
    """
    x, y, z, ux, uy, uz, energy = sample_neutron_source_spherical(
        n_neutrons, r_fuel, energy_MeV
    )

    if use_quasi_random:
        # Replace direction cosines with quasi-random
        qr = niederreiter2_generate(2, n_neutrons)
        theta = np.arccos(2.0 * qr[0] - 1.0)
        phi = 2.0 * np.pi * qr[1]
        ux = np.sin(theta) * np.cos(phi)
        uy = np.sin(theta) * np.sin(phi)
        uz = np.cos(theta)

    results = {
        'n_escaped': 0,
        'n_absorbed': 0,
        'path_lengths': [],
        'depositions': [],
        'final_energies': []
    }

    for i in range(n_neutrons):
        escaped, pl, fe, dep = transport_neutron(
            x[i], y[i], z[i], ux[i], uy[i], uz[i], energy[i],
            r_outer, rho_func, max_steps
        )

        if escaped:
            results['n_escaped'] += 1
        else:
            results['n_absorbed'] += 1

        results['path_lengths'].append(pl)
        results['depositions'].append(dep)
        results['final_energies'].append(fe)

    results['path_lengths'] = np.array(results['path_lengths'])
    results['depositions'] = np.array(results['depositions'])
    results['final_energies'] = np.array(results['final_energies'])

    return results


def histogramize_spectrum(energies, bin_min, bin_max, bin_num):
    """
    Bin neutron energies into histogram.
    Based on histogramize from Project 543.
    """
    n = len(energies)
    if n == 0:
        return np.zeros(bin_num), np.zeros(bin_num)

    energies = np.asarray(energies)
    # Convert to MeV
    energies_MeV = energies / (1e6 * E_CHARGE)

    bindex = 1 + np.floor(bin_num * (energies_MeV - bin_min) / (bin_max - bin_min))
    bindex = bindex.astype(int)

    for i in range(n):
        if bindex[i] == bin_num + 1 and energies_MeV[i] < bin_max + 1e-12 * (bin_max - bin_min):
            bindex[i] = bin_num

    bin_ave = np.zeros(bin_num)
    bin_count = np.zeros(bin_num)

    for i in range(bin_num):
        bin_ave[i] = bin_min + (2.0 * i + 1.0) * (bin_max - bin_min) / (2.0 * bin_num)
        bin_count[i] = np.sum(bindex == (i + 1))

    return bin_ave, bin_count


def compute_neutron_diagnostics(mc_results, r_fuel, r_outer):
    """
    Compute neutron diagnostic quantities.
    """
    n_total = mc_results['n_escaped'] + mc_results['n_absorbed']
    if n_total == 0:
        return {}

    escape_fraction = mc_results['n_escaped'] / n_total
    avg_path = np.mean(mc_results['path_lengths'])
    avg_deposition = np.mean(mc_results['depositions'])
    total_deposition = np.sum(mc_results['depositions'])

    bin_ave, bin_count = histogramize_spectrum(
        mc_results['final_energies'], 0.0, 15.0, 50
    )

    return {
        'escape_fraction': escape_fraction,
        'avg_path_length': avg_path,
        'avg_deposition': avg_deposition,
        'total_deposition': total_deposition,
        'energy_spectrum_bins': bin_ave,
        'energy_spectrum_counts': bin_count,
        'n_total': n_total
    }

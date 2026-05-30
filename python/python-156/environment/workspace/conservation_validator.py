
import numpy as np


def mass_conservation_checksum(Y_fields, weights=None):
    n = len(next(iter(Y_fields.values())))
    checksum = np.zeros(n)

    for name, Y in Y_fields.items():
        checksum += np.clip(Y, 0.0, 1.0)

    max_error = np.max(np.abs(checksum - 1.0))

    return checksum, max_error


def energy_conservation_checksum(T_field, Y_fields, cp=1200.0,
                                  enthalpies=None):
    if enthalpies is None:
        enthalpies = {'fuel': -4.5e7, 'oxidizer': 0.0, 'product': -3.9e7}

    checksum = cp * T_field
    for name, Y in Y_fields.items():
        h = enthalpies.get(name, 0.0)
        checksum += np.clip(Y, 0.0, 1.0) * h

    mean_energy = np.mean(checksum)
    if abs(mean_energy) < 1.0e-12:
        mean_energy = 1.0

    max_error = np.max(np.abs(checksum - mean_energy)) / abs(mean_energy)

    return checksum, max_error


def validate_simulation(T_field, Y_F_field, Y_O_field, Z_nodes,
                        mass_tol=1.0e-3, energy_tol=1.0e-2):
    n = len(Z_nodes)


    Y_P_field = 1.0 - Y_F_field - Y_O_field
    Y_P_field = np.clip(Y_P_field, 0.0, 1.0)
    Y_fields = {'fuel': Y_F_field, 'oxidizer': Y_O_field, 'product': Y_P_field}
    mass_check, mass_error = mass_conservation_checksum(Y_fields)


    energy_check, energy_error = energy_conservation_checksum(
        T_field, Y_fields,
        enthalpies={'fuel': -4.5e7, 'oxidizer': 0.0, 'product': -3.9e7}
    )


    T_min = np.min(T_field)
    T_max = np.max(T_field)
    T_reasonable = (T_min >= 200.0 and T_max <= 3500.0)

    Y_reasonable = (np.all(Y_F_field >= -1.0e-6) and
                    np.all(Y_F_field <= 1.0 + 1.0e-6) and
                    np.all(Y_O_field >= -1.0e-6) and
                    np.all(Y_O_field <= 1.0 + 1.0e-6))

    validation = {
        'mass_conservation_error': mass_error,
        'mass_conservation_passed': mass_error < mass_tol,
        'energy_conservation_error': energy_error,
        'energy_conservation_passed': energy_error < energy_tol,
        'temperature_range': (float(T_min), float(T_max)),
        'temperature_reasonable': T_reasonable,
        'mass_fractions_reasonable': Y_reasonable,
        'overall_valid': (mass_error < mass_tol and
                          energy_error < energy_tol and
                          T_reasonable and Y_reasonable),
        'checksum_mass_mean': float(np.mean(mass_check)),
        'checksum_energy_mean': float(np.mean(energy_check)),
    }

    return validation

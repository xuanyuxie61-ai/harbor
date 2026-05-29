"""
ICF Implosion Symmetry Diagnostics and Reconstruction

Based on:
- theodolite (Project 1258): Multi-point triangulation/optimization for 3D reconstruction

Models:
- Multi-angle line-of-sight diagnostic arrays
- 3D implosion shape reconstruction from chordal measurements
- Residual minimization for symmetry quantification
- Hot spot radius and shape inference
"""

import numpy as np
from scipy.optimize import minimize


def line_par_point_dist_3d(direction_cosines, x0, y0, z0, point):
    """
    Distance from point to line through (x0,y0,z0) with direction cosines (l,m,n).
    Based on line_par_point_dist_3d from Project 1258.
    """
    l, m, n = direction_cosines
    px, py, pz = point

    # Vector from line origin to point
    dx = px - x0
    dy = py - y0
    dz = pz - z0

    # Cross product magnitude / |direction|
    cross_x = dy * n - dz * m
    cross_y = dz * l - dx * n
    cross_z = dx * m - dy * l

    dist = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)
    dir_mag = np.sqrt(l**2 + m**2 + n**2)
    if dir_mag > 1e-15:
        dist /= dir_mag

    return dist


def generate_diagnostic_array(n_diags=10, radius=5.0):
    """
    Generate diagnostic camera positions on a sphere around target.
    Based on theodolite data setup from Project 1258.
    """
    # Distribute cameras uniformly on sphere
    phi = np.linspace(0.0, 2.0 * np.pi, n_diags, endpoint=False)
    theta = np.arccos(1.0 - 2.0 * np.arange(n_diags) / n_diags)

    xyz = np.zeros((n_diags, 3))
    for i in range(n_diags):
        xyz[i, 0] = radius * np.sin(theta[i]) * np.cos(phi[i])
        xyz[i, 1] = radius * np.sin(theta[i]) * np.sin(phi[i])
        xyz[i, 2] = radius * np.cos(theta[i])

    return xyz


def chordal_transmission_diagnostic(camera_pos, target_center, target_radius,
                                     emission_profile):
    """
    Simulate chordal transmission measurement through imploded target.
    Returns effective chord length and integrated emission.
    """
    # Line from camera through target center
    direction = target_center - camera_pos
    dist = np.linalg.norm(direction)
    if dist < 1e-15:
        return 0.0, 0.0

    direction = direction / dist

    # For spherical target, chord length = 2 * sqrt(R^2 - b^2)
    # where b is impact parameter (0 for on-axis line)
    chord = 2.0 * target_radius

    # Integrated emission (simplified)
    integrated = emission_profile * chord

    return chord, integrated


def symmetry_residual(xyz_star, camera_positions, chord_measurements):
    """
    Residual function for 3D implosion reconstruction.
    Based on theodolite_f from Project 1258.

    xyz_star : candidate implosion center/shape parameters
    camera_positions : array of diagnostic positions
    chord_measurements : measured chord lengths
    """
    n = len(camera_positions)
    f = np.zeros(n)

    # xyz_star = [x_center, y_center, z_center, radius, ellipticity_e]
    if len(xyz_star) < 4:
        xyz_star = np.append(xyz_star, [1e-4, 0.0])

    center = xyz_star[:3]
    R_eff = xyz_star[3]
    e_ellip = xyz_star[4] if len(xyz_star) > 4 else 0.0

    for j in range(n):
        cam = camera_positions[j]

        # Direction toward center
        direction = center - cam
        dir_mag = np.linalg.norm(direction)
        if dir_mag < 1e-15:
            f[j] = abs(chord_measurements[j])
            continue

        direction = direction / dir_mag

        # Distance from point (center) to line through camera
        dist_to_line = line_par_point_dist_3d(direction, cam[0], cam[1], cam[2], center)

        # Effective radius in this direction (ellipsoidal)
        # R_eff(theta) = R_eff * (1 - e * cos^2(theta))
        cos_theta = np.dot(direction, np.array([0.0, 0.0, 1.0]))
        R_dir = R_eff * (1.0 - e_ellip * cos_theta**2)

        # Theoretical chord for this line of sight
        if R_dir > dist_to_line:
            chord_theory = 2.0 * np.sqrt(R_dir**2 - dist_to_line**2)
        else:
            chord_theory = 0.0

        f[j] = abs(chord_theory - chord_measurements[j])

    return f


def reconstruct_implosion_shape(camera_positions, chord_measurements,
                                 initial_guess=None):
    """
    Reconstruct 3D implosion shape from multi-angle chord measurements.
    Based on theodolite optimization from Project 1258.
    """
    n = len(camera_positions)
    if n == 0:
        return np.zeros(5)

    if initial_guess is None:
        initial_guess = np.array([0.0, 0.0, 0.0, 1e-4, 0.0])

    def objective(params):
        residuals = symmetry_residual(params, camera_positions, chord_measurements)
        return np.sum(residuals**2)

    # Bounds for physical parameters
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0),
              (1e-6, 1.0), (-0.5, 0.5)]

    result = minimize(objective, initial_guess, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 1000, 'ftol': 1e-12})

    return result.x


def compute_symmetry_metrics(reconstructed_params, camera_positions,
                              chord_measurements):
    """
    Compute symmetry metrics from reconstruction.
    """
    center = reconstructed_params[:3]
    R_eff = reconstructed_params[3]
    e_ellip = reconstructed_params[4]

    residuals = symmetry_residual(reconstructed_params, camera_positions,
                                   chord_measurements)
    rms_residual = np.sqrt(np.mean(residuals**2))

    # P2/P0 Legendre mode (ellipticity proxy)
    p2_mode = e_ellip

    # Shell uniformity metric
    uniformity = 1.0 - abs(e_ellip)

    return {
        'center': center,
        'radius': R_eff,
        'ellipticity': e_ellip,
        'rms_residual': rms_residual,
        'p2_mode': p2_mode,
        'uniformity': uniformity
    }


def simulate_implosion_measurements(true_center, true_radius, true_ellipticity,
                                    camera_positions, noise_level=0.01):
    """
    Simulate chord measurements with noise for testing reconstruction.
    """
    n = len(camera_positions)
    chords = np.zeros(n)

    for j in range(n):
        cam = camera_positions[j]
        direction = true_center - cam
        dir_mag = np.linalg.norm(direction)
        if dir_mag < 1e-15:
            chords[j] = 0.0
            continue

        direction = direction / dir_mag
        dist_to_line = line_par_point_dist_3d(direction, cam[0], cam[1], cam[2],
                                               true_center)

        cos_theta = np.dot(direction, np.array([0.0, 0.0, 1.0]))
        R_dir = true_radius * (1.0 - true_ellipticity * cos_theta**2)

        if R_dir > dist_to_line:
            chord = 2.0 * np.sqrt(R_dir**2 - dist_to_line**2)
        else:
            chord = 0.0

        # Add noise
        noise = noise_level * true_radius * (2.0 * np.random.random() - 1.0)
        chords[j] = max(chord + noise, 0.0)

    return chords

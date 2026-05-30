
import numpy as np
from math import gamma as math_gamma


G_CONST = 6.67430e-11


def keast_tetrahedron_nodes_weights(order):
    if order == 1:

        nodes = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1.0 / 6.0])
    elif order == 2:

        a = 0.58541020
        b = 0.13819660
        nodes = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ])
        weights = np.ones(4) * (1.0 / 24.0)
    elif order == 3:

        nodes = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0/6.0, 1.0/6.0],
            [1.0/6.0, 0.5, 1.0/6.0],
            [1.0/6.0, 1.0/6.0, 0.5],
            [1.0/6.0, 1.0/6.0, 1.0/6.0]
        ])
        weights = np.array([-2.0/15.0, 3.0/40.0, 3.0/40.0, 3.0/40.0, 3.0/40.0]) * (1.0 / 6.0) / (1.0/6.0)

        weights = weights / np.sum(weights) * (1.0 / 6.0)
    elif order == 4:


        nodes_list = [[0.25, 0.25, 0.25]]
        weights_list = [-0.013155555555555556]

        a = 0.7857142857142857
        b = 0.07142857142857142
        perm = [
            [a, a, a, b], [a, a, b, a], [a, b, a, a], [b, a, a, a]
        ]

        for p in perm:
            x, y, z, w = p

            s = x + y + z + w
            x, y, z, w = x/s, y/s, z/s, w/s
            nodes_list.append([x, y, z])
            weights_list.append(0.007622222222222222)

        a = 0.1005964283272147
        b = 0.3994035716727853
        perm2 = [
            [a, a, b, b], [a, b, a, b], [a, b, b, a],
            [b, a, a, b], [b, a, b, a], [b, b, a, a]
        ]
        for p in perm2:
            x, y, z, w = p
            s = x + y + z + w
            x, y, z, w = x/s, y/s, z/s, w/s
            nodes_list.append([x, y, z])
            weights_list.append(0.024888888888888888)
        nodes = np.array(nodes_list)
        weights = np.array(weights_list)

        weights = weights / np.sum(weights) * (1.0 / 6.0)
    else:
        raise ValueError("keast_tetrahedron_nodes_weights: unsupported order {}".format(order))
    

    assert nodes.shape[1] == 3, "Nodes must have 3 columns"
    assert nodes.shape[0] == weights.shape[0], "Nodes and weights length mismatch"
    assert np.all(nodes >= -1e-12) and np.all(nodes.sum(axis=1) <= 1.0 + 1e-12), "Nodes outside reference tetrahedron"
    vol_sum = np.sum(weights)
    assert abs(vol_sum - 1.0/6.0) < 1e-10, "Weights do not sum to reference tetrahedron volume"
    
    return nodes, weights


def reference_to_physical_tetrahedron(nodes_ref, verts):
    verts = np.asarray(verts, dtype=float)
    if verts.shape != (4, 3):
        raise ValueError("verts must be of shape (4, 3)")
    J = np.column_stack([
        verts[1] - verts[0],
        verts[2] - verts[0],
        verts[3] - verts[0]
    ])
    detJ = np.linalg.det(J)
    if abs(detJ) < 1e-15:
        raise ValueError("Degenerate tetrahedron encountered: det(J) = {}".format(detJ))
    
    nodes_ref = np.asarray(nodes_ref, dtype=float)

    nodes_phys = verts[0] + nodes_ref @ J.T
    return nodes_phys, abs(detJ)


def prism_gravity_anomaly(prism_bounds, density, obs_points):
    x1, x2, y1, y2, z1, z2 = prism_bounds
    if x1 >= x2 or y1 >= y2 or z1 >= z2:
        raise ValueError("Prism bounds must satisfy x1<x2, y1<y2, z1<z2")
    if density == 0:
        return np.zeros(obs_points.shape[0])
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    
    N = obs.shape[0]
    dg = np.zeros(N)
    

    signs = np.array([1, -1])
    











    raise NotImplementedError("Hole_1: prism_gravity_anomaly Nagy公式核心求和待实现")
    return dg


def tetrahedron_gravity_anomaly(verts, density, obs_points, keast_order=3):
    if density == 0:
        return np.zeros(obs_points.shape[0])
    
    nodes_ref, weights_ref = keast_tetrahedron_nodes_weights(keast_order)
    nodes_phys, detJ = reference_to_physical_tetrahedron(nodes_ref, verts)
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    N = obs.shape[0]
    dg = np.zeros(N)
    

    for q in range(nodes_phys.shape[0]):
        r_q = nodes_phys[q]
        w_q = weights_ref[q]

        dx = obs[:, 0] - r_q[0]
        dy = obs[:, 1] - r_q[1]
        dz = obs[:, 2] - r_q[2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.maximum(dist, 1e-12)

        dV = detJ * w_q

        dg += G_CONST * density * (obs[:, 2] - r_q[2]) / (dist**3) * dV
    
    dg *= 1e5
    return dg


def hammersley_sequence_3d(n_points, offset=0):
    if n_points < 0:
        raise ValueError("n_points must be non-negative")
    
    primes = [2, 3, 5]
    seq = np.zeros((n_points, 3))
    
    for idx in range(n_points):
        i = idx + offset

        if n_points > 0:
            seq[idx, 0] = (i % (n_points + 1)) / max(n_points, 1)
        else:
            seq[idx, 0] = 0.0
        

        p = primes[1]
        val = 0.0
        inv_p = 1.0 / p
        f = inv_p
        t = i
        while t > 0:
            d = t % p
            val += d * f
            f *= inv_p
            t //= p
        seq[idx, 1] = val
        

        p = primes[2]
        val = 0.0
        inv_p = 1.0 / p
        f = inv_p
        t = i
        while t > 0:
            d = t % p
            val += d * f
            f *= inv_p
            t //= p
        seq[idx, 2] = val
    

    seq = np.clip(seq, 0.0, 1.0 - 1e-15)
    return seq


def qmc_gravity_anomaly(volume_bounds, density_func, obs_points, n_samples=5000):
    xmin, xmax, ymin, ymax, zmin, zmax = volume_bounds
    if xmin >= xmax or ymin >= ymax or zmin >= zmax:
        raise ValueError("Invalid volume bounds")
    
    seq = hammersley_sequence_3d(n_samples)

    samples = np.zeros_like(seq)
    samples[:, 0] = xmin + seq[:, 0] * (xmax - xmin)
    samples[:, 1] = ymin + seq[:, 1] * (ymax - ymin)
    samples[:, 2] = zmin + seq[:, 2] * (zmax - zmin)
    
    rho = density_func(samples)
    if np.any(np.isnan(rho)) or np.any(np.isinf(rho)):
        raise ValueError("density_func returned NaN or Inf")
    
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    M = obs.shape[0]
    dg = np.zeros(M)
    
    dV = (xmax - xmin) * (ymax - ymin) * (zmax - zmin) / n_samples
    
    for q in range(n_samples):
        dx = obs[:, 0] - samples[q, 0]
        dy = obs[:, 1] - samples[q, 1]
        dz = obs[:, 2] - samples[q, 2]
        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        dist = np.maximum(dist, 1e-12)
        dg += G_CONST * rho[q] * dz / (dist**3) * dV
    
    dg *= 1e5
    return dg


def spherical_harmonic_gravity(clm, slm, radius, colat, lon, r_obs, max_degree):
    GM = 3.986004418e14
    
    cos_theta = np.cos(colat)
    dg = 0.0
    
    for l in range(2, max_degree + 1):

        plm = _associated_legendre(l, cos_theta)
        for m in range(0, l + 1):
            if m < clm.shape[1] and l < clm.shape[0]:
                coeff = (l - 1) * (radius / r_obs)**l
                harm = plm[m] * (clm[l, m] * np.cos(m * lon) + slm[l, m] * np.sin(m * lon))
                dg += coeff * harm
    
    dg *= GM / (r_obs**2) * 1e5
    return dg


def _associated_legendre(l, x):
    x = float(np.clip(x, -1.0, 1.0))
    plm = np.zeros(l + 1)
    

    plm[0] = 1.0
    if l == 0:
        return plm
    

    plm[0] = np.sqrt(3.0) * x

    if l >= 1:
        plm[1] = -np.sqrt(3.0 * max(1.0 - x**2, 0.0))
    



    if l >= 2:
        p_mm = plm[1]
        p_mmp1 = plm[0]
        for ll in range(2, l + 1):


            p_mm_new = np.sqrt((2.0 * ll + 1.0) / (2.0 * ll)) * x * p_mmp1 - np.sqrt((2.0 * ll + 1.0) / (2.0 * (ll - 1.0))) * p_mm
            p_mm = p_mmp1
            p_mmp1 = p_mm_new
            plm[0] = p_mmp1
    
    return plm


def composite_forward_model(prisms, tetras, density_func, obs_points, qmc_samples=2000):
    obs = np.asarray(obs_points, dtype=float)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    N = obs.shape[0]
    dg_total = np.zeros(N)
    

    for prism in prisms:
        bounds = prism[:6]
        rho = prism[6]
        dg_total += prism_gravity_anomaly(bounds, rho, obs)
    

    for tetra in tetras:
        verts = tetra[0]
        rho = tetra[1]
        dg_total += tetrahedron_gravity_anomaly(verts, rho, obs, keast_order=3)
    

    if density_func is not None:

        if len(prisms) > 0:
            xs = [p[0] for p in prisms] + [p[1] for p in prisms]
            ys = [p[2] for p in prisms] + [p[3] for p in prisms]
            zs = [p[4] for p in prisms] + [p[5] for p in prisms]
            vb = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        elif len(tetras) > 0:
            allv = np.vstack([t[0] for t in tetras])
            vb = (allv[:,0].min(), allv[:,0].max(),
                  allv[:,1].min(), allv[:,1].max(),
                  allv[:,2].min(), allv[:,2].max())
        else:
            vb = (-1e4, 1e4, -1e4, 1e4, -3e4, 0)
        dg_total += qmc_gravity_anomaly(vb, density_func, obs, n_samples=qmc_samples)
    
    return dg_total

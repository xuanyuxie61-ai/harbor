# -*- coding: utf-8 -*-

import numpy as np


def components_3d(A):
    A = np.asarray(A, dtype=int)
    nx, ny, nz = A.shape
    C = np.zeros_like(A, dtype=int)
    component_index = 0

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if A[i, j, k] != 0 and C[i, j, k] == 0:
                    component_index += 1
                    plist = [(i, j, k)]

                    while plist:
                        ci, cj, ck = plist.pop()
                        if C[ci, cj, ck] != 0:
                            continue
                        C[ci, cj, ck] = component_index


                        neighbors = []
                        if ci > 0 and A[ci - 1, cj, ck] != 0 and C[ci - 1, cj, ck] == 0:
                            neighbors.append((ci - 1, cj, ck))
                        if ci < nx - 1 and A[ci + 1, cj, ck] != 0 and C[ci + 1, cj, ck] == 0:
                            neighbors.append((ci + 1, cj, ck))
                        if cj > 0 and A[ci, cj - 1, ck] != 0 and C[ci, cj - 1, ck] == 0:
                            neighbors.append((ci, cj - 1, ck))
                        if cj < ny - 1 and A[ci, cj + 1, ck] != 0 and C[ci, cj + 1, ck] == 0:
                            neighbors.append((ci, cj + 1, ck))
                        if ck > 0 and A[ci, cj, ck - 1] != 0 and C[ci, cj, ck - 1] == 0:
                            neighbors.append((ci, cj, ck - 1))
                        if ck < nz - 1 and A[ci, cj, ck + 1] != 0 and C[ci, cj, ck + 1] == 0:
                            neighbors.append((ci, cj, ck + 1))

                        plist.extend(neighbors)

    return C, component_index


def q_criterion(u, v, w, dx, dy, dz):
    def ddx(f):
        result = np.zeros_like(f)
        result[1:-1, :, :] = (f[2:, :, :] - f[:-2, :, :]) / (2.0 * dx)
        return result

    def ddy(f):
        result = np.zeros_like(f)
        result[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        return result

    def ddz(f):
        result = np.zeros_like(f)
        result[:, :, 1:-1] = (f[:, :, 2:] - f[:, :, :-2]) / (2.0 * dz)
        return result

    dudx = ddx(u)
    dudy = ddy(u)
    dudz = ddz(u)
    dvdx = ddx(v)
    dvdy = ddy(v)
    dvdz = ddz(v)
    dwdx = ddx(w)
    dwdy = ddy(w)
    dwdz = ddz(w)


    O12 = 0.5 * (dudy - dvdx)
    O13 = 0.5 * (dudz - dwdx)
    O23 = 0.5 * (dvdz - dwdy)


    S11 = dudx
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S22 = dvdy
    S23 = 0.5 * (dvdz + dwdy)
    S33 = dwdz


    norm_Omega_sq = 2.0 * (O12 ** 2 + O13 ** 2 + O23 ** 2)
    norm_S_sq = S11 ** 2 + S22 ** 2 + S33 ** 2 + 2.0 * S12 ** 2 + 2.0 * S13 ** 2 + 2.0 * S23 ** 2

    Q = 0.5 * (norm_Omega_sq - norm_S_sq)
    return Q


def percolation_analysis_3d(field, thresholds=None):
    field = np.asarray(field, dtype=float)
    nx, ny, nz = field.shape

    if thresholds is None:
        qmin, qmax = np.min(field), np.max(field)
        thresholds = np.linspace(qmin, qmax, 11)

    results = []

    for thresh in thresholds:

        binary = (field > thresh).astype(int)


        labels, n_components = components_3d(binary)


        component_sizes = []
        for c in range(1, n_components + 1):
            size = np.sum(labels == c)
            if size > 0:
                component_sizes.append(size)

        component_sizes = np.array(component_sizes, dtype=int)


        p_occupied = np.mean(binary)


        spanning_x = 0
        spanning_y = 0
        spanning_z = 0

        for c in range(1, n_components + 1):
            mask = (labels == c)
            if np.any(mask[0, :, :]) and np.any(mask[-1, :, :]):
                spanning_x += 1
            if np.any(mask[:, 0, :]) and np.any(mask[:, -1, :]):
                spanning_y += 1
            if np.any(mask[:, :, 0]) and np.any(mask[:, :, -1]):
                spanning_z += 1


        if len(component_sizes) > 0:
            max_size = np.max(component_sizes)

            r_eq = (max_size / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
            fractal_dim = np.log(max_size) / (np.log(r_eq) + 1e-15) if r_eq > 1 else 0.0
        else:
            fractal_dim = 0.0

        results.append({
            'threshold': thresh,
            'p_occupied': p_occupied,
            'n_components': n_components,
            'mean_size': float(np.mean(component_sizes)) if len(component_sizes) > 0 else 0.0,
            'max_size': int(np.max(component_sizes)) if len(component_sizes) > 0 else 0,
            'spanning_x': spanning_x,
            'spanning_y': spanning_y,
            'spanning_z': spanning_z,
            'fractal_dim': float(fractal_dim)
        })

    return results


def energy_cascade_topology(u, v, w, dx, dy, dz):
    nx, ny, nz = u.shape
    scales = [1, 2, 4]
    topology_metrics = []

    for scale in scales:
        if nx // scale < 3 or ny // scale < 3 or nz // scale < 3:
            continue


        u_coarse = u[::scale, ::scale, ::scale]
        v_coarse = v[::scale, ::scale, ::scale]
        w_coarse = w[::scale, ::scale, ::scale]


        Q = q_criterion(u_coarse, v_coarse, w_coarse,
                        dx * scale, dy * scale, dz * scale)


        perc = percolation_analysis_3d(Q, thresholds=[np.percentile(Q, 75)])

        topology_metrics.append({
            'scale': scale,
            'n_components': perc[0]['n_components'],
            'mean_size': perc[0]['mean_size'],
            'fractal_dim': perc[0]['fractal_dim'],
            'spanning': perc[0]['spanning_x'] + perc[0]['spanning_y'] + perc[0]['spanning_z']
        })

    return scales, topology_metrics

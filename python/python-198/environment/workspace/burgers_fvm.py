
import numpy as np


def godunov_flux(u_left, u_right):
    uL = np.asarray(u_left, dtype=float)
    uR = np.asarray(u_right, dtype=float)
    

    M = 1e3
    uL = np.clip(uL, -M, M)
    uR = np.clip(uR, -M, M)
    
    flux = np.zeros_like(uL)
    

    mask_sparse = uL < uR
    flux[mask_sparse & (uR <= 0)] = 0.5 * uR[mask_sparse & (uR <= 0)] ** 2
    flux[mask_sparse & (uL >= 0)] = 0.5 * uL[mask_sparse & (uL >= 0)] ** 2
    flux[mask_sparse & (uL < 0) & (uR > 0)] = 0.0
    

    mask_shock = uL >= uR
    mid = 0.5 * (uL + uR)
    flux[mask_shock & (mid > 0)] = 0.5 * uL[mask_shock & (mid > 0)] ** 2
    flux[mask_shock & (mid < 0)] = 0.5 * uR[mask_shock & (mid < 0)] ** 2
    flux[mask_shock & np.isclose(mid, 0)] = 0.0
    

    flux = np.clip(flux, -M**2, M**2)
    return flux


def lax_friedrichs_flux(u_left, u_right, alpha):
    fL = 0.5 * u_left ** 2
    fR = 0.5 * u_right ** 2
    return 0.5 * (fL + fR) - 0.5 * alpha * (u_right - u_left)


def upwind_flux(u_left, u_right):
    f = np.zeros_like(u_left)
    mask = u_left >= 0
    f[mask] = 0.5 * u_left[mask] ** 2
    f[~mask] = 0.5 * u_right[~mask] ** 2
    return f


def build_fvm_operators(nodes, elements):
    n_elem = elements.shape[0]
    n_nodes = nodes.shape[0]
    

    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    area = 0.5 * np.abs(
        p1[:, 0] * (p2[:, 1] - p3[:, 1])
        + p2[:, 0] * (p3[:, 1] - p1[:, 1])
        + p3[:, 0] * (p1[:, 1] - p2[:, 1])
    )
    area = np.clip(area, 1e-14, None)
    
    centroid = (p1 + p2 + p3) / 3.0
    

    edge_to_tri = {}
    for e_idx, tri in enumerate(elements):
        edges = [
            tuple(sorted((tri[0], tri[1]))),
            tuple(sorted((tri[1], tri[2]))),
            tuple(sorted((tri[2], tri[0]))),
        ]
        for ed in edges:
            if ed not in edge_to_tri:
                edge_to_tri[ed] = []
            edge_to_tri[ed].append(e_idx)
    

    internal_edges = []
    boundary_edges = []
    for ed, tri_list in edge_to_tri.items():
        n0, n1 = nodes[ed[0]], nodes[ed[1]]
        ex = n1[0] - n0[0]
        ey = n1[1] - n0[1]
        edge_len = np.sqrt(ex ** 2 + ey ** 2)

        nx = ey / (edge_len + 1e-15)
        ny = -ex / (edge_len + 1e-15)
        
        if len(tri_list) == 2:

            tL, tR = tri_list[0], tri_list[1]

            dx = centroid[tR][0] - centroid[tL][0]
            dy = centroid[tR][1] - centroid[tL][1]
            if dx * nx + dy * ny < 0:
                nx, ny = -nx, -ny
                tL, tR = tR, tL
            internal_edges.append((tL, tR, edge_len, nx, ny))
        else:
            boundary_edges.append((tri_list[0], ed[0], ed[1], edge_len, nx, ny))
    
    return area, centroid, internal_edges, boundary_edges


def burgers_fvm_step(u, area, internal_edges, boundary_edges,
                     dt, flux_type='godunov', boundary_value=0.0):
    n_elem = len(u)
    rhs = np.zeros(n_elem)
    

    for tL, tR, elen, nx, ny in internal_edges:
        uL = u[tL]
        uR = u[tR]
        
        if flux_type == 'godunov':
            flux = godunov_flux(uL, uR)
        elif flux_type == 'lax_friedrichs':
            alpha = max(abs(uL), abs(uR), 1e-8)
            flux = lax_friedrichs_flux(uL, uR, alpha)
        elif flux_type == 'upwind':
            flux = upwind_flux(uL, uR)
        else:
            flux = godunov_flux(uL, uR)
        
        sign = np.sign(nx + 1e-15)
        f_val = flux * sign
        
        rhs[tL] -= f_val * elen / area[tL]
        rhs[tR] += f_val * elen / area[tR]
    

    for tL, n0, n1, elen, nx, ny in boundary_edges:
        uL = u[tL]
        uR = boundary_value
        
        if flux_type == 'godunov':
            flux = godunov_flux(uL, uR)
        elif flux_type == 'lax_friedrichs':
            alpha = max(abs(uL), abs(uR), 1e-8)
            flux = lax_friedrichs_flux(uL, uR, alpha)
        else:
            flux = godunov_flux(uL, uR)
        
        sign = np.sign(nx + 1e-15)
        f_val = flux * sign
        rhs[tL] -= f_val * elen / area[tL]
    

    u_new = u + dt * rhs

    u_new = np.clip(u_new, -1e3, 1e3)

    u_new = np.where(np.isfinite(u_new), u_new, 0.0)
    return u_new


def solve_burgers_fvm(nodes, elements, u0_func, t_max, nt,
                      flux_type='godunov', boundary_value=0.0, cfl=0.15):
    area, centroid, internal_edges, boundary_edges = build_fvm_operators(nodes, elements)
    n_elem = elements.shape[0]
    

    u = u0_func(centroid[:, 0], centroid[:, 1])
    u = np.clip(u, -1e2, 1e2)
    
    dt = t_max / nt
    

    max_speed = np.max(np.abs(u))
    min_h = np.min(np.sqrt(area))
    dt_cfl = cfl * min_h / (max_speed + 1e-8)
    if dt > dt_cfl:
        nt = int(np.ceil(t_max / dt_cfl)) + 1
        dt = t_max / nt
    
    U = np.zeros((nt + 1, n_elem))
    U[0] = u
    
    for i in range(nt):
        u = burgers_fvm_step(u, area, internal_edges, boundary_edges,
                             dt, flux_type, boundary_value)
        U[i + 1] = u

        if i % 20 == 0:
            max_speed = np.max(np.abs(u))
            if not np.isfinite(max_speed) or max_speed > 1e4:

                U[i+1:] = np.nan
                break
    
    t = np.linspace(0, t_max, nt + 1)
    return t, U

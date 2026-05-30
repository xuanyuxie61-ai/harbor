
import numpy as np


MANTLE_DENSITY = 3300.0
CHEM_DIFFUSIVITY = 1e-6
GRAVITY_ACC = 9.81


def lax_wendroff_density_convection(nz, dz, dt, n_steps,
                                     rho_init, u_field, D_diff, source,
                                     bc_type='periodic'):
    if nz < 3:
        raise ValueError("nz must be >= 3")
    
    rho = np.asarray(rho_init, dtype=float).copy()
    u = np.asarray(u_field, dtype=float)
    if len(rho) != nz or len(u) != nz:
        raise ValueError("Field length mismatch with nz")
    

    umax = np.max(np.abs(u))
    if umax > 0:
        cfl_adv = dz / umax
    else:
        cfl_adv = 1e10
    if D_diff > 0:
        cfl_diff = dz**2 / (2.0 * D_diff)
    else:
        cfl_diff = 1e10
    
    cfl = min(cfl_adv, cfl_diff)
    if dt > cfl:

        sub_steps = int(np.ceil(dt / (0.5 * cfl)))
        actual_dt = dt / sub_steps
    else:
        sub_steps = 1
        actual_dt = dt
    
    history = []
    
    for step in range(n_steps):
        for _ in range(sub_steps):
            rho_half = np.zeros(nz - 1)
            

            for j in range(nz - 1):


                flux_adv = actual_dt / dz * (u[j + 1] * rho[j + 1] - u[j] * rho[j])

                if D_diff > 0 and j > 0:
                    flux_diff = actual_dt * D_diff / dz**2 * (rho[j + 1] - 2.0 * rho[j] + rho[j - 1])
                else:
                    flux_diff = 0.0
                
                val = 0.5 * (rho[j] + rho[j + 1]) - 0.5 * flux_adv + 0.5 * flux_diff

                rho_half[j] = np.clip(val, -1e4, 1e4)
            

            rho_old = rho.copy()
            for j in range(1, nz - 1):

                u_half_j = 0.5 * (u[j] + u[j + 1])
                u_half_jm1 = 0.5 * (u[j - 1] + u[j])
                flux = actual_dt / dz * (u_half_j * rho_half[j] - u_half_jm1 * rho_half[j - 1])
                

                if D_diff > 0:
                    diff = actual_dt * D_diff / dz**2 * (rho_old[j + 1] - 2.0 * rho_old[j] + rho_old[j - 1])
                else:
                    diff = 0.0
                

                if callable(source):
                    src = actual_dt * source(j * dz, step * dt)
                else:
                    src = actual_dt * source[j]
                
                val = rho_old[j] - flux + diff + src
                rho[j] = np.clip(val, -1e4, 1e4)
            

            rho = _apply_bc(rho, bc_type)
        
        if step % max(1, n_steps // 20) == 0:
            history.append(rho.copy())
    
    return rho, history


def _apply_bc(rho, bc_type):
    nz = len(rho)
    if bc_type == 'periodic':
        rho[0] = rho[-2]
        rho[-1] = rho[1]
    elif bc_type == 'fixed':

        pass
    elif bc_type == 'reflective':
        rho[0] = rho[1]
        rho[-1] = rho[-2]
    elif bc_type == 'zero_gradient':
        rho[0] = 2.0 * rho[1] - rho[2]
        rho[-1] = 2.0 * rho[-2] - rho[-3]
    return rho


def porous_medium_barenblatt(x, t, m=2.0, C=1.0, delta=0.1):
    if m <= 1.0:
        raise ValueError("m must be > 1 for porous medium equation")
    
    x = np.asarray(x, dtype=float)
    t = float(t)
    
    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))
    
    bot = (t + delta)**beta
    xi = x / bot
    factor = C - gamma * xi**2
    
    u = np.zeros_like(x)
    ut = np.zeros_like(x)
    ux = np.zeros_like(x)
    uxx = np.zeros_like(x)
    
    mask = factor > 0.0
    if np.any(mask):
        u[mask] = (t + delta)**(-beta) * factor[mask]**alpha
        ut[mask] = (2.0 * alpha * beta * gamma * (t + delta)**(-1.0 - 3.0 * beta) * x[mask]**2 * factor[mask]**(alpha - 1.0)
                    - beta * (t + delta)**(-1.0 - beta) * factor[mask]**alpha)
        ux[mask] = (-2.0 * alpha * gamma * (t + delta)**(-3.0 * beta) * x[mask] * factor[mask]**(alpha - 1.0))
        uxx[mask] = (4.0 * (alpha - 1.0) * alpha * gamma**2 * (t + delta)**(-5.0 * beta) * x[mask]**2 * factor[mask]**(alpha - 2.0)
                     - 2.0 * alpha * gamma * (t + delta)**(-3.0 * beta) * factor[mask]**(alpha - 1.0))
    
    return u, ut, ux, uxx


def porous_medium_verification(nz, z_max, t_test, m=2.0, C=1.0, delta=0.1):
    dz = z_max / (nz - 1)
    z = np.linspace(0, z_max, nz)
    
    u_exact, _, _, _ = porous_medium_barenblatt(z, t_test, m, C, delta)
    

    dt = 0.1 * dz**2
    n_steps = int(t_test / dt)
    u_num = porous_medium_barenblatt(z, 0.0, m, C, delta)[0]
    
    for _ in range(n_steps):
        u_old = u_num.copy()
        for j in range(1, nz - 1):

            um_jp = u_old[j + 1]**m
            um_j = u_old[j]**m
            um_jm = u_old[j - 1]**m
            u_num[j] = u_old[j] + dt * (um_jp - 2.0 * um_j + um_jm) / dz**2
        u_num[0] = u_num[1]
        u_num[-1] = u_num[-2]
    
    diff = np.abs(u_num - u_exact)
    l2_error = np.sqrt(np.mean(diff**2))
    linf_error = np.max(diff)
    
    return l2_error, linf_error, u_num, u_exact


def stokes_velocity_profile(z, eta, delta_rho, L_scale):
    z = np.asarray(z, dtype=float)
    u_max = delta_rho * GRAVITY_ACC * L_scale**2 / (eta + 1e-30)

    u_max = np.clip(u_max, -1e-2, 1e-2)
    
    u = -u_max * (z / L_scale) * (1.0 - z / L_scale)
    return u


def density_anomaly_evolution_full(z, nz, dt, n_steps,
                                    rho_background, delta_rho_init,
                                    eta, D_diff, source,
                                    L_scale=1e5):
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    

    u = stokes_velocity_profile(z, eta, delta_rho_init, L_scale)
    

    rho_init = np.ones(nz) * rho_background

    z_center = L_scale / 2.0
    sigma = L_scale / 10.0
    rho_init += delta_rho_init * np.exp(-((z - z_center)**2) / (2.0 * sigma**2))
    
    rho_final, history = lax_wendroff_density_convection(
        nz, dz, dt, n_steps, rho_init, u, D_diff, source, bc_type='zero_gradient'
    )
    
    return rho_final, history

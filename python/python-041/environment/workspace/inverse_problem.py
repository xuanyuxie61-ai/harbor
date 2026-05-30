
import numpy as np
from wave_propagation import rk4_integrate
from helmholtz_solver import gmres_solve


def compute_misfit(d_obs, d_calc):
    residual = d_calc - d_obs
    misfit = 0.5 * np.sum(residual ** 2)
    return misfit, residual


def compute_gradient_fd(objective_func, m, h=1e-4):
    n = len(m)
    grad = np.zeros(n)
    j0 = objective_func(m)
    for i in range(n):
        m_plus = m.copy()
        m_minus = m.copy()
        m_plus[i] += h
        m_minus[i] -= h
        grad[i] = (objective_func(m_plus) - objective_func(m_minus)) / (2.0 * h)
    return grad


def adjoint_state_gradient_1d(m, d_obs, dx, dt, nt, source_pos, source_fn,
                               boundary='absorbing'):
    nx = len(m)
    c = np.asarray(m, dtype=float)
    

    from wave_propagation import seismic_wave_rk4_1d
    u_hist, t = seismic_wave_rk4_1d(nx, dx, nt, dt, c, source_fn, source_pos,
                                     boundary=boundary)
    

    misfit, residual = compute_misfit(d_obs, u_hist)
    


    def adjoint_deriv(tau, y):


        lam = y[:nx]
        vlam = y[nx:]
        dlam_dtau = vlam.copy()
        dvlam_dtau = np.zeros(nx)
        t_forward = nt * dt - tau
        t_idx = min(int(t_forward / dt), nt)







        pass

        if 0 <= t_idx <= nt:
            dvlam_dtau += residual[t_idx, :]
        return np.concatenate([dlam_dtau, dvlam_dtau])
    
    y0 = np.zeros(2 * nx)
    tau, y_adj = rk4_integrate(adjoint_deriv, (0.0, nt * dt), y0, nt)

    lambda_field = y_adj[::-1, :nx]
    

    grad = np.zeros(nx)
    for ti in range(nt + 1):
        uxx = np.zeros(nx)
        u_t = u_hist[ti, :]
        for i in range(1, nx - 1):
            uxx[i] = (u_t[i - 1] - 2 * u_t[i] + u_t[i + 1]) / dx ** 2
        grad += uxx * lambda_field[ti, :]
    grad = -2.0 * grad / (c ** 3) * dt
    

    grad_smooth = grad.copy()
    for i in range(1, nx - 1):
        grad_smooth[i] = 0.25 * grad[i - 1] + 0.5 * grad[i] + 0.25 * grad[i + 1]
    grad_smooth[0] = grad[0]
    grad_smooth[-1] = grad[-1]
    
    return grad_smooth, misfit


def fwi_gradient_descent_1d(m_init, d_obs, dx, dt, nt, source_pos, source_fn,
                             n_iter=20, step_length=1e6, boundary='absorbing',
                             verbose=True):
    m = m_init.copy()
    m_history = [m.copy()]
    misfit_history = []
    beta_reg = 1e-4
    
    for k in range(n_iter):
        grad, misfit = adjoint_state_gradient_1d(
            m, d_obs, dx, dt, nt, source_pos, source_fn, boundary=boundary
        )
        

        reg_term = np.zeros_like(m)
        for i in range(1, len(m) - 1):
            reg_term[i] = -2 * m[i] + m[i - 1] + m[i + 1]
        reg_term[0] = 0.0
        reg_term[-1] = 0.0
        grad += beta_reg * reg_term / dx ** 2
        
        alpha = step_length / (1.0 + 0.1 * k)
        m = m - alpha * grad

        m = np.clip(m, 1000.0, 8000.0)
        
        m_history.append(m.copy())
        misfit_history.append(misfit)
        
        if verbose and k % 5 == 0:
            print(f"  FWI Iteration {k}: misfit = {misfit:.6e}, alpha = {alpha:.3e}")
    
    return m_history, misfit_history


def tomography_traveltime_1d(m, dx, source_pos, receiver_positions):
    c = np.asarray(m, dtype=float)
    traveltimes = np.zeros(len(receiver_positions))
    for ir, rec_pos in enumerate(receiver_positions):
        i1 = min(source_pos, rec_pos)
        i2 = max(source_pos, rec_pos)
        if i1 == i2:
            traveltimes[ir] = 0.0
        else:
            traveltimes[ir] = np.sum(dx / c[i1:i2])
    return traveltimes
